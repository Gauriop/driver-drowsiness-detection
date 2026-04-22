"""
cnn_inference.py
================
Lightweight helper that app.py imports to run CNN inference on uploaded images.

Usage in app.py:
    from cnn_inference import CNNPredictor
    cnn = CNNPredictor('cnn_output/best_cnn_model.keras',
                       'cnn_output/cnn_metadata.json')
    result = cnn.predict(frame_bgr)   # returns dict compatible with /predict response

Why a separate module?
    Keeps app.py clean. All TensorFlow / MTCNN imports are isolated here so
    the webcam path (MediaPipe) is unaffected if the CNN model isn't present.
"""

import json
import os
import numpy as np
import cv2

# ── Lazy imports so startup time is not hurt if CNN is unused ────────────────
_tf = None
_mp_preprocess = None
_face_detector = None   # MTCNN instance, loaded once


def _ensure_tf():
    global _tf, _mp_preprocess
    if _tf is None:
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        _tf = tf
        _mp_preprocess = preprocess_input


def _ensure_mtcnn():
    global _face_detector
    if _face_detector is None:
        try:
            from mtcnn import MTCNN
            _face_detector = MTCNN()
        except ImportError:
            _face_detector = False   # sentinel: MTCNN not installed


# ── Face crop (same logic as notebook) ──────────────────────────────────────

def _crop_face(img_bgr, img_size=(224, 224), margin=0.25):
    """Return a face-cropped, resized RGB float32 [0,1] array."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    _ensure_mtcnn()
    if not _face_detector:
        return (cv2.resize(rgb, img_size).astype(np.float32) / 255.0)

    results = _face_detector.detect_faces(rgb)
    if not results:
        return (cv2.resize(rgb, img_size).astype(np.float32) / 255.0)

    best = max(results, key=lambda r: r['confidence'])
    x, y, w, h = best['box']
    H, W = rgb.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    x1 = max(0, x - mx);  y1 = max(0, y - my)
    x2 = min(W, x + w + mx); y2 = min(H, y + h + my)
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        crop = rgb
    return (cv2.resize(crop, img_size).astype(np.float32) / 255.0)


# ── Main predictor class ─────────────────────────────────────────────────────

class CNNPredictor:
    """
    Wraps a trained Keras CNN model and exposes a predict() method
    that returns the same dict structure as app.py's MediaPipe path
    so the /predict endpoint can swap between them transparently.
    """

    def __init__(self, model_path: str, metadata_path: str):
        """
        Parameters
        ----------
        model_path    : path to best_cnn_model.keras (or .h5)
        metadata_path : path to cnn_metadata.json produced by the notebook
        """
        _ensure_tf()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f'CNN model not found: {model_path}')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f'CNN metadata not found: {metadata_path}')

        self.model = _tf.keras.models.load_model(model_path)
        with open(metadata_path) as f:
            self.meta = json.load(f)

        self.class_names      = self.meta['class_names']   # ['alert','drowsy','yawning']
        self.img_size         = tuple(self.meta['img_size'])
        self.use_face_crop    = self.meta.get('use_face_crop', True)
        self.drowsy_threshold = self.meta.get('drowsy_threshold', 0.45)
        self.num_classes      = self.meta.get('num_classes', 3)

        print(f'[CNNPredictor] model loaded: {model_path}')
        print(f'[CNNPredictor] classes: {self.class_names}  img_size: {self.img_size}')

    def predict(self, frame_bgr: np.ndarray) -> dict:
        """
        Run inference on a BGR OpenCV frame.

        Returns
        -------
        dict with keys matching what app.py's /predict endpoint sends to the
        frontend, plus 'cnn_probs' for debugging.
        """
        _ensure_tf()

        # ── 1. Preprocessing ────────────────────────────────────────────────
        if self.use_face_crop:
            face_rgb_01 = _crop_face(frame_bgr, img_size=self.img_size)
        else:
            rgb = cv2.cvtColor(cv2.resize(frame_bgr, self.img_size), cv2.COLOR_BGR2RGB)
            face_rgb_01 = rgb.astype(np.float32) / 255.0

        # MobileNetV2 preprocess_input expects [0,255] → maps to [-1,1]
        x = _mp_preprocess(np.expand_dims(face_rgb_01 * 255.0, axis=0))

        # ── 2. Inference ────────────────────────────────────────────────────
        probs     = self.model.predict(x, verbose=0)[0]
        pred_idx  = int(np.argmax(probs))
        pred_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])

        # ── 3. Map to app.py status strings ─────────────────────────────────
        #   0 = alert   → STATUS 'ALERT'
        #   1 = drowsy  → STATUS 'DROWSY'
        #   2 = yawning → STATUS 'YAWNING' (no alarm, just banner)
        status_map = {
            'alert':   'ALERT',
            'drowsy':  'DROWSY',
            'yawning': 'ALERT',    # yawning: no drowsy alarm; is_yawning=True handles UI
        }
        status     = status_map.get(pred_class, 'ALERT')
        is_yawning = (pred_class == 'yawning')
        is_drowsy  = (pred_class == 'drowsy')

        # Drowsiness score (0-100) derived from the drowsy probability
        drowsy_prob = float(probs[1]) if self.num_classes > 1 else 0.0
        yawn_prob   = float(probs[2]) if self.num_classes > 2 else 0.0
        score = round(min(100, (drowsy_prob * 0.8 + yawn_prob * 0.3) * 100), 1)

        return {
            # Fields read by the frontend JS (updateUI)
            'status':       status,
            'score':        score,
            'is_yawning':   is_yawning,
            'is_distracted': False,    # CNN doesn't do distraction — MediaPipe handles live
            'face_found':   True,
            'ear':          0.0,       # not computed by CNN
            'left_ear':     0.0,
            'right_ear':    0.0,
            'mar':          0.0,
            'too_far':      False,
            'face_width':   0,
            'gaze':         'CENTER',
            'head_yaw':     0,
            'head_pitch':   0,
            'consec_eye':   0,
            'consec_distract': 0,
            'consec_needed': 48,
            'distract_needed': 50,
            'alerts':       1 if is_drowsy else 0,
            'yawns':        1 if is_yawning else 0,
            'blinks':       0,
            'distractions': 0,
            # Extra CNN-specific fields (for debugging / future UI)
            'model_used':   'CNN',
            'cnn_class':    pred_class,
            'cnn_confidence': round(confidence, 4),
            'cnn_probs': {cn: round(float(p), 4)
                          for cn, p in zip(self.class_names, probs)},
        }


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print('Usage: python cnn_inference.py <model.keras> <image.jpg>')
        sys.exit(1)

    predictor = CNNPredictor(sys.argv[1], sys.argv[1].replace('.keras', '').replace('.h5','')
                             .replace('best_cnn_model','') + 'cnn_metadata.json')
    frame     = cv2.imread(sys.argv[2])
    if frame is None:
        print('Could not read image')
        sys.exit(1)

    result = predictor.predict(frame)
    print(json.dumps(result, indent=2))