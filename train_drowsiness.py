# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════
import argparse
import csv
import os
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.spatial import distance as dist

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_sample_weight


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

LABEL_MAP = {
    'eyeclose': 1,
    'yawn': 1,
    'neutral': 0,
    'happy': 0
}

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

MEDIAPIPE_MODEL_PATH = 'face_landmarker.task'
MEDIAPIPE_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'face_landmarker/face_landmarker/float16/1/face_landmarker.task'
)

LEFT_EYE_IDX  = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE_IDX = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT = 13, 14, 61, 291
NOSE_TIP = 1
LEFT_EAR_LM = 234
RIGHT_EAR_LM = 454

BASE_FEATURES = ['ear','left_ear','right_ear','mar','head_yaw','head_pitch','face_width']
DERIVED_FEATURES = ['ear_diff','ear_mar_ratio','yaw_abs','pitch_abs']
ALL_FEATURES = BASE_FEATURES + DERIVED_FEATURES

CSV_COLUMNS = ALL_FEATURES + ['label_raw','label','subject']


# ═══════════════════════════════════════════════════════════════════════════
# MEDIAPIPE
# ═══════════════════════════════════════════════════════════════════════════

def ensure_model():
    if not os.path.exists(MEDIAPIPE_MODEL_PATH):
        print("Downloading MediaPipe model...")
        urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, MEDIAPIPE_MODEL_PATH)

def make_detector():
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_pts(lm, idxs, w, h):
    return np.array([(lm[i].x*w, lm[i].y*h) for i in idxs])

def compute_ear(pts):
    v1 = dist.euclidean(pts[1], pts[5])
    v2 = dist.euclidean(pts[2], pts[4])
    h  = dist.euclidean(pts[0], pts[3])
    return (v1+v2)/(2*h) if h>0 else 0

def compute_mar(lm, w, h):
    def pt(i): return np.array([lm[i].x*w, lm[i].y*h])
    v = dist.euclidean(pt(MOUTH_TOP), pt(MOUTH_BOTTOM))
    h = dist.euclidean(pt(MOUTH_LEFT), pt(MOUTH_RIGHT))
    return v/h if h>0 else 0

def compute_head(lm, w, h):
    nose = np.array([lm[NOSE_TIP].x*w, lm[NOSE_TIP].y*h])
    l = np.array([lm[LEFT_EAR_LM].x*w, lm[LEFT_EAR_LM].y*h])
    r = np.array([lm[RIGHT_EAR_LM].x*w, lm[RIGHT_EAR_LM].y*h])

    cx = (l[0]+r[0])/2
    width = dist.euclidean(l,r)
    yaw = ((nose[0]-cx)/(width+1e-9))*90

    return yaw, 0, width


def landmarks_to_row(lm, w, h, label_raw, label, subject):
    l_pts = get_pts(lm, LEFT_EYE_IDX, w, h)
    r_pts = get_pts(lm, RIGHT_EYE_IDX, w, h)

    l_ear = compute_ear(l_pts)
    r_ear = compute_ear(r_pts)
    ear = (l_ear+r_ear)/2

    mar = compute_mar(lm,w,h)
    yaw, pitch, fw = compute_head(lm,w,h)

    return {
        'ear':ear,'left_ear':l_ear,'right_ear':r_ear,
        'mar':mar,'head_yaw':yaw,'head_pitch':pitch,'face_width':fw,
        'ear_diff':abs(l_ear-r_ear),
        'ear_mar_ratio':ear/(mar+1e-6),
        'yaw_abs':abs(yaw),'pitch_abs':abs(pitch),
        'label_raw':label_raw,
        'label':label,
        'subject':subject
    }


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract(dataset, out_csv):
    detector = make_detector()
    rows = []

    for subj in Path(dataset).iterdir():
        if not subj.is_dir(): continue

        for label_dir in subj.iterdir():
            key = label_dir.name.lower()
            if key not in LABEL_MAP: continue

            label = LABEL_MAP[key]

            for img_path in label_dir.glob("*"):
                if img_path.suffix.lower() not in IMAGE_EXTS: continue

                img = cv2.imread(str(img_path))
                if img is None: continue

                # 🔥 augmentation
                if np.random.rand() < 0.3:
                    img = cv2.flip(img, 1)

                h,w = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if not res.face_landmarks: continue

                row = landmarks_to_row(
                    res.face_landmarks[0],
                    w,h,
                    label_dir.name,
                    label,
                    subj.name
                )
                rows.append(row)

    with open(out_csv,"w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows → {out_csv}")


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train(csv_path, model_path):
    df = pd.read_csv(csv_path)

    subjects = df['subject'].unique()
    test_sub = subjects[-1]

    train_df = df[df['subject']!=test_sub]
    test_df  = df[df['subject']==test_sub]

    X_train = train_df[ALL_FEATURES].values
    y_train = train_df['label'].values

    X_test  = test_df[ALL_FEATURES].values
    y_test  = test_df['label'].values

    # 🔥 alert importance boost
    sample_weights = np.where(y_train == 0, 1.5, 1.0)

    model = Pipeline([
        ('scaler',StandardScaler()),
        ('clf',GradientBoostingClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=15,
            subsample=0.7,
            random_state=42
        ))
    ])

    model.fit(X_train,y_train,clf__sample_weight=sample_weights)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    print("\n=== REAL TEST RESULTS ===")
    print(classification_report(y_test,y_pred))

    auc = roc_auc_score(y_test,y_prob)
    print("ROC-AUC:",auc)

    cm = confusion_matrix(y_test,y_pred)

    disp = ConfusionMatrixDisplay(cm,display_labels=['alert','drowsy'])
    disp.plot()

    plt.savefig("confusion_matrix.png")
    print("Saved confusion_matrix.png")

    joblib.dump(model,model_path)
    print("Model saved:",model_path)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--csv', default='features.csv')
    parser.add_argument('--model', default='model.pkl')
    args = parser.parse_args()

    ensure_model()
    extract(args.dataset, args.csv)
    train(args.csv, args.model)

if __name__ == "__main__":
    main()