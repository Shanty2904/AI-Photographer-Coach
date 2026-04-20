"""
sharpness.py
------------
Measures overall and subject-specific sharpness / focus quality:
  - Global sharpness via Laplacian variance
  - Subject (face) sharpness vs full-frame sharpness
  - Detects if subject is out of focus while background is sharp (focus miss)
  - Tenengrad gradient method for more robust sharpness estimation
"""

import cv2
import numpy as np

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _laplacian_sharpness(region: np.ndarray) -> float:
    if region.size == 0:
        return 0.0
    return float(cv2.Laplacian(region, cv2.CV_64F).var())


def _tenengrad(region: np.ndarray) -> float:
    """Sobel-based sharpness — more robust than Laplacian for real photos."""
    if region.size == 0:
        return 0.0
    gx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx**2 + gy**2))


def analyze_sharpness(frame: np.ndarray) -> dict:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w  = gray.shape

    global_lap  = _laplacian_sharpness(gray)
    global_ten  = _tenengrad(gray)

    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    focus_miss = False
    subject_sharpness = None
    subject_score_mod = 0.0

    if len(faces) > 0:
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        pad  = 15
        x1   = max(0, fx - pad);  y1 = max(0, fy - pad)
        x2   = min(w, fx + fw + pad); y2 = min(h, fy + fh + pad)
        face_region = gray[y1:y2, x1:x2]

        subject_sharpness = _tenengrad(face_region)

        # Focus miss: background sharper than subject
        bg_mask   = np.ones_like(gray, dtype=bool)
        bg_mask[y1:y2, x1:x2] = False
        bg_region = gray[bg_mask]
        if bg_region.size > 0:
            bg_sharp = _tenengrad(bg_region.reshape(1, -1))
            if subject_sharpness < bg_sharp * 0.5:
                focus_miss      = True
                subject_score_mod = -3.0

    # Score based on Laplacian variance thresholds (typical for 640x480)
    if global_lap > 800:
        label = "tack_sharp";  base_score = 10.0; suggestion = ""
    elif global_lap > 400:
        label = "sharp";       base_score = 8.5;  suggestion = ""
    elif global_lap > 150:
        label = "soft";        base_score = 6.0;  suggestion = "Image appears slightly soft. Hold the camera steadier or tap to focus on your subject."
    elif global_lap > 50:
        label = "blurry";      base_score = 3.5;  suggestion = "Image is blurry. Ensure subject is in focus and hold camera still."
    else:
        label = "very_blurry"; base_score = 1.0;  suggestion = "Very blurry image. Tap your subject on screen to force focus, and hold very still."

    if focus_miss:
        suggestion = "Focus appears to be on the background, not the subject. Tap your subject on screen to refocus."

    final_score = max(0.0, round(base_score + subject_score_mod, 2))

    return {
        "sharpness_label":      label,
        "sharpness_score":      final_score,
        "global_laplacian":     round(global_lap, 2),
        "global_tenengrad":     round(global_ten, 2),
        "subject_sharpness":    round(subject_sharpness, 2) if subject_sharpness is not None else None,
        "focus_miss":           focus_miss,
        "sharpness_suggestion": suggestion,
    }
