"""
app.py — DrowseGuard (fixed)
============================
FIXES IN THIS VERSION
─────────────────────
1. DISTRACTION: YAW_THRESH raised 15 → 25 deg. Head must be turned CLEARLY
   to the side, not just slightly off-centre. Also requires DISTRACTION_FRAMES
   (50) of sustained gaze-away before firing, same as before — but the higher
   angle means normal head movement no longer triggers it.

2. EAR CALIBRATION: First 60 frames collect the person's own open-eye EAR.
   Threshold = personal_baseline × 0.75  (instead of a hardcoded 0.21).
   People with naturally narrow/small eyes will no longer be wrongly flagged.
   A "CALIBRATING…" banner shows during those first 60 frames.

3. REAL BEEP: Uses a background thread so the video stream is never blocked.
   - Windows  → winsound.Beep(880, 400)
   - macOS    → afplay /System/Library/Sounds/Funk.aiff
   - Linux    → aplay /usr/share/sounds/alsa/Front_Left.wav  (or paplay)
   Beep fires every 1.5 s while DROWSY state is active.
   Stops immediately when driver opens eyes.

4. YAWN ≠ DROWSY: Yawning only sets is_yawning=True and increments yawn
   counter. It does NOT push status to DROWSY on its own. Only sustained
   eye closure or a high drowsiness score does that.

5. CLOSED EYES = DROWSY (not yawn): Eye closure and mouth opening are
   evaluated independently. A wide-open mouth with open eyes = yawn.
   Closed eyes = drowsiness path, separate from yawn path.
"""

import sys, os, platform, threading, time
from flask import Flask, request, render_template, jsonify, Response
import cv2, numpy as np, mediapipe as mp, json, base64
from scipy.spatial import distance as dist
from collections import deque
import urllib.request
import joblib

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# ── MediaPipe model ──────────────────────────────────────────────────────────
MODEL_PATH = 'face_landmarker.task'
MODEL_URL  = ('https://storage.googleapis.com/mediapipe-models/'
              'face_landmarker/face_landmarker/float16/1/face_landmarker.task')
if not os.path.exists(MODEL_PATH):
    print('Downloading face_landmarker.task (~30 MB)…')
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print('Done.')

# ── GBM model ────────────────────────────────────────────────────────────────
GBM_MODEL_PATH = 'model.pkl'
if not os.path.exists(GBM_MODEL_PATH):
    raise FileNotFoundError(
        f"'{GBM_MODEL_PATH}' not found — run train_drowsiness.py first."
    )
gbm_model = joblib.load(GBM_MODEL_PATH)
print(f'Loaded GBM model from {GBM_MODEL_PATH}')

# ── Config ────────────────────────────────────────────────────────────────────
with open('config.json') as f:
    cfg = json.load(f)

# These are now FALLBACK defaults only.
# The live stream uses a per-person calibrated threshold instead.
EAR_THRESHOLD_DEFAULT = cfg['ear_threshold']          # 0.21  fallback
MAR_THRESHOLD         = cfg['mar_threshold']           # 0.60
CONSEC_FRAMES         = cfg['consec_frames']           # 48
MIN_FACE_WIDTH_PX     = cfg.get('min_face_width_px', 120)
DISTRACTION_FRAMES    = cfg.get('distraction_frames', 50)
LEFT_EYE_IDX          = cfg['left_eye_idx']
RIGHT_EYE_IDX         = cfg['right_eye_idx']

MOUTH_TOP    = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291
YAWN_CONSEC  = 30
NOSE_TIP     = 1
CHIN         = 199
LEFT_EAR_LM  = 234
RIGHT_EAR_LM = 454

# ── FIX 1: wider yaw threshold so normal head movement ≠ distracted ──────────
YAW_THRESH   = 25.0   # degrees — must clearly turn head to fire (was 15)
PITCH_THRESH = 18.0   # degrees — looking clearly up/down

# ── FIX 2: EAR calibration constants ─────────────────────────────────────────
CALIB_FRAMES        = 60    # collect open-eye EAR for 2 s at 30 fps
EAR_CALIB_RATIO     = 0.75  # threshold = baseline × 0.75

# Feature order for GBM (must match training)
ALL_FEATURES = [
    'ear', 'left_ear', 'right_ear', 'mar',
    'head_yaw', 'head_pitch', 'face_width',
    'ear_diff', 'ear_mar_ratio', 'yaw_abs', 'pitch_abs'
]


# ═══════════════════════════════════════════════════════════════════════════════
#  BEEP (FIX 3)
# ═══════════════════════════════════════════════════════════════════════════════

_beep_active = False
_beep_thread  = None

def _beep_worker():
    """Plays a system beep every 1.5 s while _beep_active is True."""
    while _beep_active:
        _play_one_beep()
        time.sleep(1.5)

def _play_one_beep():
    sys_name = platform.system()
    try:
        if sys_name == 'Windows':
            import winsound
            winsound.Beep(880, 400)          # 880 Hz, 400 ms
        elif sys_name == 'Darwin':           # macOS
            os.system('afplay /System/Library/Sounds/Funk.aiff &')
        else:                                # Linux
            # Try paplay first, fall back to aplay, then bell
            if os.path.exists('/usr/bin/paplay'):
                os.system('paplay /usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga &')
            elif os.path.exists('/usr/bin/aplay'):
                os.system('aplay /usr/share/sounds/alsa/Front_Left.wav &')
            else:
                sys.stdout.write('\a'); sys.stdout.flush()
    except Exception:
        sys.stdout.write('\a'); sys.stdout.flush()

def start_beep():
    global _beep_active, _beep_thread
    if _beep_active:
        return
    _beep_active = True
    _beep_thread = threading.Thread(target=_beep_worker, daemon=True)
    _beep_thread.start()

def stop_beep():
    global _beep_active
    _beep_active = False


# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE STREAM STATE
# ═══════════════════════════════════════════════════════════════════════════════

stream_state = {
    'ear': 0.0, 'mar': 0.0, 'score': 0,
    'status': 'ALERT', 'left_ear': 0.0, 'right_ear': 0.0,
    'consec_eye': 0, 'consec_yawn': 0, 'consec_distract': 0,
    'alerts': 0, 'yawns': 0, 'distractions': 0,
    'blink_count': 0, 'frame_count': 0,
    'ear_history': deque(maxlen=90),
    'is_yawning': False, 'is_drowsy_eye': False, 'is_distracted': False,
    'too_far': False, 'face_width': 0,
    'gaze': 'CENTER', 'head_yaw': 0.0, 'head_pitch': 0.0,
    # FIX 2: calibration
    'calib_ears': [],           # raw EAR readings during calibration window
    'ear_threshold': EAR_THRESHOLD_DEFAULT,  # starts as fallback, updated after calib
    'calibrated': False,
    'calibrating': True,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  MEDIAPIPE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def make_detector(running_mode):
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=running_mode, num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)

def get_pts(landmarks, indices, w, h):
    return np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in indices],
        dtype=np.float64
    )

def compute_ear(pts):
    v1 = dist.euclidean(pts[1], pts[5])
    v2 = dist.euclidean(pts[2], pts[4])
    h_ = dist.euclidean(pts[0], pts[3])
    return round((v1 + v2) / (2.0 * h_), 4) if h_ > 0 else 0.0

def compute_mar(landmarks, w, h):
    def pt(i): return np.array([landmarks[i].x * w, landmarks[i].y * h])
    v  = dist.euclidean(pt(MOUTH_TOP),  pt(MOUTH_BOTTOM))
    hz = dist.euclidean(pt(MOUTH_LEFT), pt(MOUTH_RIGHT))
    return round(v / hz, 4) if hz > 0 else 0.0

def get_face_width(landmarks, w, h):
    l = np.array([landmarks[LEFT_EAR_LM].x * w, landmarks[LEFT_EAR_LM].y * h])
    r = np.array([landmarks[RIGHT_EAR_LM].x * w, landmarks[RIGHT_EAR_LM].y * h])
    return dist.euclidean(l, r)

def compute_head_pose(landmarks, w, h):
    nose  = np.array([landmarks[NOSE_TIP].x * w,    landmarks[NOSE_TIP].y * h])
    chin  = np.array([landmarks[CHIN].x * w,         landmarks[CHIN].y * h])
    l_ear = np.array([landmarks[LEFT_EAR_LM].x * w,  landmarks[LEFT_EAR_LM].y * h])
    r_ear = np.array([landmarks[RIGHT_EAR_LM].x * w, landmarks[RIGHT_EAR_LM].y * h])

    face_cx    = (l_ear[0] + r_ear[0]) / 2.0
    face_width = dist.euclidean(l_ear, r_ear)
    yaw_deg    = ((nose[0] - face_cx) / (face_width + 1e-9)) * 90.0

    face_height = dist.euclidean(
        np.array([landmarks[10].x * w,  landmarks[10].y * h]),
        np.array([landmarks[152].x * w, landmarks[152].y * h])
    )
    nose_y_norm = (nose[1] - chin[1]) / (face_height + 1e-9)
    pitch_deg   = nose_y_norm * 60.0

    # FIX 1: wider thresholds — normal head movement won't fire
    if   abs(yaw_deg) > YAW_THRESH:   gaze = 'RIGHT' if yaw_deg > 0 else 'LEFT'
    elif pitch_deg < -PITCH_THRESH:   gaze = 'UP'
    elif pitch_deg > PITCH_THRESH:    gaze = 'DOWN'
    else:                             gaze = 'CENTER'

    return round(yaw_deg, 1), round(pitch_deg, 1), gaze


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION (for GBM /predict)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features(landmarks, w, h):
    l_ear = compute_ear(get_pts(landmarks, LEFT_EYE_IDX,  w, h))
    r_ear = compute_ear(get_pts(landmarks, RIGHT_EYE_IDX, w, h))
    ear   = round((l_ear + r_ear) / 2.0, 4)
    mar   = compute_mar(landmarks, w, h)
    yaw, pitch, gaze = compute_head_pose(landmarks, w, h)
    fw    = get_face_width(landmarks, w, h)
    return {
        'ear': ear, 'left_ear': l_ear, 'right_ear': r_ear, 'mar': mar,
        'head_yaw': yaw, 'head_pitch': pitch, 'face_width': fw,
        'ear_diff': abs(l_ear - r_ear),
        'ear_mar_ratio': ear / (mar + 1e-6),
        'yaw_abs': abs(yaw), 'pitch_abs': abs(pitch),
    }, ear, l_ear, r_ear, mar, yaw, pitch, gaze, fw


def predict_with_gbm(landmarks, w, h):
    features, ear, l_ear, r_ear, mar, yaw, pitch, gaze, fw = extract_features(landmarks, w, h)
    X         = np.array([[features[f] for f in ALL_FEATURES]])
    gbm_label = gbm_model.predict(X)[0]
    gbm_proba = gbm_model.predict_proba(X)[0]
    p_drowsy  = float(gbm_proba[1])
    score     = round(p_drowsy * 100, 1)
    too_far   = fw < MIN_FACE_WIDTH_PX

    if   too_far:           status = 'TOO_FAR'
    elif gbm_label == 1:    status = 'DROWSY'
    elif mar > MAR_THRESHOLD:   status = 'YAWNING'
    elif gaze != 'CENTER':  status = 'DISTRACTED'
    elif score >= 30:       status = 'WARNING'
    else:                   status = 'ALERT'

    return (status, score, features,
            ear, l_ear, r_ear, mar, yaw, pitch, gaze, fw,
            too_far, mar > MAR_THRESHOLD, gaze != 'CENTER', p_drowsy)


# ═══════════════════════════════════════════════════════════════════════════════
#  DRAWING
# ═══════════════════════════════════════════════════════════════════════════════

def draw_eye(frame, pts, color):
    hull = cv2.convexHull(pts.astype(np.int32))
    cv2.drawContours(frame, [hull], -1, color, 1)
    for (x, y) in pts.astype(int):
        cv2.circle(frame, (x, y), 2, color, -1)

def draw_mouth(frame, landmarks, w, h, color, mar_val, is_yawning):
    OUTER_LIP = [61,185,40,39,37,0,267,269,270,409,
                 291,375,321,405,314,17,84,181,91,146]
    pts = np.array(
        [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in OUTER_LIP],
        dtype=np.int32
    )
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
    cx = (int(landmarks[MOUTH_LEFT].x*w) + int(landmarks[MOUTH_RIGHT].x*w)) // 2
    cy = int(landmarks[MOUTH_TOP].y*h) - 10
    lbl = f'MAR:{mar_val:.2f}' + (' YAWN!' if is_yawning else '')
    cv2.putText(frame, lbl, (cx-45, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def draw_gaze_arrow(frame, landmarks, w, h, gaze, yaw, pitch):
    nx = int(landmarks[NOSE_TIP].x * w)
    ny = int(landmarks[NOSE_TIP].y * h)
    dx = int(yaw / 90.0 * 30)
    dy = int(pitch / 60.0 * 30)
    color = (0, 165, 255) if gaze != 'CENTER' else (0, 255, 80)
    cv2.arrowedLine(frame, (nx, ny), (nx+dx, ny+dy), color, 2, tipLength=0.4)
    cv2.putText(frame, gaze, (nx+dx+4, ny+dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

def annotate_frame(frame, lm, l_ear, r_ear, avg_ear, mar_val,
                   score, status, is_yawning, is_distracted,
                   face_width, gaze, yaw, pitch, ear_threshold,
                   calibrating=False, model_label='GBM'):
    h, w    = frame.shape[:2]
    is_drow = status == 'DROWSY'
    is_warn = status == 'WARNING'
    is_dist = status == 'DISTRACTED'
    too_far = face_width < MIN_FACE_WIDTH_PX

    eye_c   = (0,0,255) if is_drow else ((0,165,255) if (is_warn or is_dist) else (0,255,80))
    mouth_c = (0,0,255) if is_yawning else ((0,165,255) if mar_val > 0.45 else (80,200,120))

    draw_eye(frame, get_pts(lm, LEFT_EYE_IDX,  w, h), eye_c)
    draw_eye(frame, get_pts(lm, RIGHT_EYE_IDX, w, h), eye_c)
    draw_mouth(frame, lm, w, h, mouth_c, mar_val, is_yawning)
    draw_gaze_arrow(frame, lm, w, h, gaze, yaw, pitch)

    # HUD bar
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,90), (0,0,0), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    sc = ((0,0,255) if is_drow else
          (255,100,0) if is_dist else
          (0,165,255) if is_warn else
          (0,220,80))

    cv2.putText(frame, f'EAR:{avg_ear:.3f}', (8,26),   cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255,255,255), 2)
    cv2.putText(frame, f'MAR:{mar_val:.3f}', (8,52),   cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200,200,200), 1)
    cv2.putText(frame, f'Thr:{ear_threshold:.3f}', (8,74), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150,220,150), 1)
    cv2.putText(frame, f'Scr:{score}%',      (200,26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255,255,255), 2)
    cv2.putText(frame, f'Yaw:{yaw:+.0f}deg', (200,52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170,170,170), 1)
    cv2.putText(frame, f'{model_label}',     (200,74), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120,200,120), 1)
    fw_lbl = f'Face:{int(face_width)}px {"OK" if not too_far else "FAR"}'
    cv2.putText(frame, fw_lbl, (380,52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150,150,150), 1)
    cv2.putText(frame, status, (w-200,55), cv2.FONT_HERSHEY_SIMPLEX, 1.3, sc, 3)

    # Calibration banner
    if calibrating:
        cv2.rectangle(frame, (0, h-40), (w, h), (0,100,200), -1)
        cv2.putText(frame, 'CALIBRATING — keep eyes open and face forward',
                    (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    # Drowsy overlay
    if is_drow and not too_far:
        for t, a in [(12,80),(8,140),(4,200)]:
            ov2 = frame.copy()
            cv2.rectangle(ov2, (0,0), (w-1,h-1), (0,0,255), t)
            cv2.addWeighted(ov2, a/255, frame, 1-a/255, 0, frame)
        cv2.rectangle(frame, (0,0), (w-1,h-1), (0,0,255), 4)
        txt = '! WAKE UP !'
        ts  = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
        cv2.putText(frame, txt, ((w-ts[0])//2, h-50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)

    # Distracted overlay
    if is_dist and not too_far and not is_drow:
        for t, a in [(10,60),(5,120)]:
            ov2 = frame.copy()
            cv2.rectangle(ov2, (0,0), (w-1,h-1), (0,120,255), t)
            cv2.addWeighted(ov2, a/255, frame, 1-a/255, 0, frame)
        cv2.rectangle(frame, (0,0), (w-1,h-1), (0,120,255), 3)
        txt = f'EYES ON ROAD! ({gaze})'
        ts  = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        cv2.putText(frame, txt, ((w-ts[0])//2, h-50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,120,255), 2)

    if too_far:
        cv2.putText(frame, 'Move closer to camera',
                    (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
    if is_yawning and not is_drow and not too_far:
        cv2.putText(frame, 'YAWNING DETECTED',
                    (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE STREAM DETECTION  (rule-based + calibration + beep)
# ═══════════════════════════════════════════════════════════════════════════════

def drowsiness_score(state):
    if len(state['ear_history']) < 10:
        return 0
    avg_ear  = np.mean(state['ear_history'])
    ear_sc   = max(0.0, (0.30 - avg_ear) / 0.30 * 100)
    elapsed  = state['frame_count'] / (30 * 60) + 1e-9
    blink_sc = min(100, max(0, (state['blink_count'] / elapsed - 15) * 3))
    yawn_sc  = min(100, state['yawns'] * 20)
    return round(min(100, 0.6*ear_sc + 0.25*blink_sc + 0.15*yawn_sc), 1)


def run_detection(landmarks, w, h, state):
    """
    Rule-based detection for live webcam stream.
    Includes per-person EAR calibration (first CALIB_FRAMES frames).
    """
    face_width = get_face_width(landmarks, w, h)
    too_far    = face_width < MIN_FACE_WIDTH_PX

    l_ear   = compute_ear(get_pts(landmarks, LEFT_EYE_IDX,  w, h))
    r_ear   = compute_ear(get_pts(landmarks, RIGHT_EYE_IDX, w, h))
    avg_ear = round((l_ear + r_ear) / 2.0, 4)
    mar_val = compute_mar(landmarks, w, h)
    yaw, pitch, gaze = compute_head_pose(landmarks, w, h)

    state['face_width']  = face_width
    state['too_far']     = too_far
    state['frame_count'] += 1
    state['gaze']        = gaze
    state['head_yaw']    = yaw
    state['head_pitch']  = pitch

    # ── FIX 2: calibration phase ─────────────────────────────────────────────
    calibrating = state['calibrating']
    if calibrating and not too_far:
        # Only collect EAR when eyes appear open (mar not huge = not yawning)
        if avg_ear > 0.15 and mar_val < MAR_THRESHOLD:
            state['calib_ears'].append(avg_ear)
        if len(state['calib_ears']) >= CALIB_FRAMES:
            baseline = float(np.median(state['calib_ears']))
            state['ear_threshold'] = round(max(0.15, baseline * EAR_CALIB_RATIO), 4)
            state['calibrated']    = True
            state['calibrating']   = False
            print(f'[EAR CAL] baseline={baseline:.4f}  threshold={state["ear_threshold"]:.4f}')

    ear_threshold = state['ear_threshold']

    if too_far:
        state.update({'consec_eye': 0, 'is_drowsy_eye': False,
                      'consec_distract': 0, 'is_distracted': False})
        state['ear_history'].append(0.30)
        stop_beep()
        return (avg_ear, l_ear, r_ear, mar_val, 0, 'TOO_FAR',
                False, False, face_width, gaze, yaw, pitch, True, calibrating)

    state['ear_history'].append(avg_ear)

    # ── FIX 4 & 5: eye closure and yawn evaluated independently ──────────────
    # Eye drowsiness — uses calibrated threshold
    if avg_ear < ear_threshold:
        state['consec_eye'] += 1
        if state['consec_eye'] == CONSEC_FRAMES:
            state['alerts'] += 1
            state['is_drowsy_eye'] = True
    else:
        if 2 <= state['consec_eye'] <= 8:
            state['blink_count'] += 1
        state['consec_eye']    = 0
        state['is_drowsy_eye'] = False

    # Yawn — mouth only, independent of eye state
    if mar_val > MAR_THRESHOLD:
        state['consec_yawn'] += 1
        state['is_yawning']   = True
        if state['consec_yawn'] == YAWN_CONSEC:
            state['yawns'] += 1
    else:
        state['consec_yawn'] = 0
        state['is_yawning']  = False

    # Distraction — FIX 1: wider threshold, must be sustained
    if gaze != 'CENTER':
        state['consec_distract'] += 1
        if state['consec_distract'] >= DISTRACTION_FRAMES:
            state['is_distracted'] = True
            if state['consec_distract'] == DISTRACTION_FRAMES:
                state['distractions'] += 1
    else:
        state['consec_distract'] = 0
        state['is_distracted']   = False

    score = drowsiness_score(state)

    # Status priority: DROWSY > DISTRACTED > WARNING > ALERT
    # FIX 4: yawn alone does NOT cause DROWSY
    if state['consec_eye'] >= CONSEC_FRAMES or score >= 60:
        status = 'DROWSY'
    elif state['is_distracted']:
        status = 'DISTRACTED'
    elif score >= 30:
        status = 'WARNING'
    else:
        status = 'ALERT'

    # ── FIX 3: beep control ───────────────────────────────────────────────────
    if status == 'DROWSY' and not calibrating:
        start_beep()
    else:
        stop_beep()

    return (avg_ear, l_ear, r_ear, mar_val, score, status,
            state['is_yawning'], state['is_distracted'],
            face_width, gaze, yaw, pitch, False, calibrating)


def frame_to_b64(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Image upload — GBM model.pkl used for single-frame prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    f = request.files['file']
    if not f.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({'error': 'Use JPG or PNG'}), 400

    from werkzeug.utils import secure_filename
    fname = secure_filename(f.filename)
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    f.save(fpath)

    frame = cv2.imread(fpath)
    if frame is None:
        return jsonify({'error': 'Could not read image'}), 400

    h, w     = frame.shape[:2]
    detector = make_detector(mp_vision.RunningMode.IMAGE)
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result   = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    if not result.face_landmarks:
        return jsonify({
            'status': 'NO_FACE', 'model_used': 'GBM',
            'message': 'No face detected — try a clearer, well-lit photo.',
            'ear': 0.0, 'mar': 0.0, 'score': 0, 'gaze': 'CENTER',
            'too_far': False, 'face_width': 0,
            'alerts': 0, 'yawns': 0, 'blinks': 0, 'distractions': 0,
            'is_yawning': False, 'is_distracted': False,
            'left_ear': 0.0, 'right_ear': 0.0,
            'consec_eye': 0, 'consec_distract': 0,
            'consec_needed': CONSEC_FRAMES, 'distract_needed': DISTRACTION_FRAMES,
        })

    lm = result.face_landmarks[0]
    (status, score, features,
     avg_ear, l_ear, r_ear, mar_val, yaw, pitch, gaze, fw,
     too_far, is_yawning, is_distracted, p_drowsy) = predict_with_gbm(lm, w, h)

    out = frame.copy()
    annotate_frame(out, lm, l_ear, r_ear, avg_ear, mar_val,
                   score, status, is_yawning, is_distracted,
                   fw, gaze, yaw, pitch,
                   ear_threshold=EAR_THRESHOLD_DEFAULT,
                   calibrating=False, model_label='GBM')

    return jsonify({
        'status':          status,
        'model_used':      'GBM (model.pkl)',
        'score':           score,
        'gbm_confidence':  round(p_drowsy * 100, 1),
        'ear':             avg_ear,
        'left_ear':        l_ear,
        'right_ear':       r_ear,
        'mar':             mar_val,
        'head_yaw':        yaw,
        'head_pitch':      pitch,
        'gaze':            gaze,
        'too_far':         too_far,
        'face_width':      round(fw, 1),
        'is_yawning':      is_yawning,
        'is_distracted':   is_distracted,
        'alerts':          1 if status == 'DROWSY' else 0,
        'yawns':           1 if is_yawning else 0,
        'blinks':          0,
        'distractions':    1 if is_distracted else 0,
        'consec_eye':      0,
        'consec_distract': 0,
        'consec_needed':   CONSEC_FRAMES,
        'distract_needed': DISTRACTION_FRAMES,
        'image_b64':       frame_to_b64(out),
        'original_url':    f'/static/uploads/{fname}',
    })


def gen_webcam_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    detector  = make_detector(mp_vision.RunningMode.VIDEO)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out    = frame.copy()
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_img, frame_idx * 33)
        frame_idx += 1

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            ret_vals = run_detection(lm, w, h, stream_state)
            (avg_ear, l_ear, r_ear, mar_val, score, status,
             is_yawning, is_distracted, face_width, gaze, yaw, pitch,
             too_far, calibrating) = ret_vals

            stream_state.update({
                'ear':       round(avg_ear, 3),
                'mar':       round(mar_val, 3),
                'score':     score,
                'status':    status,
                'left_ear':  round(l_ear, 3),
                'right_ear': round(r_ear, 3),
                'too_far':   too_far,
                'face_width': round(face_width, 1),
                'gaze':      gaze,
                'head_yaw':  yaw,
                'head_pitch': pitch,
            })

            annotate_frame(out, lm, l_ear, r_ear, avg_ear, mar_val,
                           score, status, is_yawning, is_distracted,
                           face_width, gaze, yaw, pitch,
                           ear_threshold=stream_state['ear_threshold'],
                           calibrating=calibrating,
                           model_label='Live/Rules')
        else:
            stream_state['status'] = 'NO_FACE'
            stop_beep()
            cv2.putText(out, 'No face detected', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        _, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')

    cap.release()
    stop_beep()


@app.route('/video_feed')
def video_feed():
    return Response(gen_webcam_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream_stats')
def stream_stats():
    return jsonify({
        'ear':             stream_state['ear'],
        'left_ear':        stream_state.get('left_ear', 0),
        'right_ear':       stream_state.get('right_ear', 0),
        'mar':             stream_state['mar'],
        'score':           stream_state['score'],
        'status':          stream_state['status'],
        'alerts':          stream_state['alerts'],
        'yawns':           stream_state['yawns'],
        'blinks':          stream_state['blink_count'],
        'distractions':    stream_state['distractions'],
        'is_yawning':      stream_state['is_yawning'],
        'is_distracted':   stream_state['is_distracted'],
        'too_far':         stream_state['too_far'],
        'face_width':      stream_state.get('face_width', 0),
        'gaze':            stream_state['gaze'],
        'head_yaw':        stream_state['head_yaw'],
        'head_pitch':      stream_state['head_pitch'],
        'consec_eye':      stream_state['consec_eye'],
        'consec_distract': stream_state['consec_distract'],
        'consec_needed':   CONSEC_FRAMES,
        'distract_needed': DISTRACTION_FRAMES,
        'calibrating':     stream_state['calibrating'],
        'ear_threshold':   round(stream_state['ear_threshold'], 4),
    })


@app.route('/recalibrate', methods=['POST'])
def recalibrate():
    """Reset EAR calibration — call when a new driver sits down."""
    stream_state['calib_ears']   = []
    stream_state['calibrated']   = False
    stream_state['calibrating']  = True
    stream_state['ear_threshold'] = EAR_THRESHOLD_DEFAULT
    return jsonify({'ok': True, 'message': 'Calibration reset — look at camera with eyes open.'})


if __name__ == '__main__':
    app.run(debug=True, threaded=True)