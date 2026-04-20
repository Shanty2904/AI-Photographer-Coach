"""
bokeh.py
--------
Detects background blur (bokeh) quality:
  - Compares sharpness between subject region and background
  - High foreground sharpness + low background sharpness = good bokeh
  - Uses Laplacian variance as sharpness proxy
"""

import cv2
import numpy as np

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def analyze_bokeh(frame: np.ndarray) -> dict:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w  = gray.shape
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    def sharpness(region: np.ndarray) -> float:
        if region.size == 0:
            return 0.0
        return float(cv2.Laplacian(region, cv2.CV_64F).var())

    if len(faces) == 0:
        # No face — compare top third (bg) vs bottom third (fg)
        fg_sharp = sharpness(gray[h*2//3:, :])
        bg_sharp = sharpness(gray[:h//3, :])
        method   = "thirds"
    else:
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        # Subject region: face bounding box (expanded slightly)
        pad   = 20
        x1    = max(0, fx - pad)
        y1    = max(0, fy - pad)
        x2    = min(w, fx + fw + pad)
        y2    = min(h, fy + fh + pad)
        fg_sharp = sharpness(gray[y1:y2, x1:x2])

        # Background: everything outside face region
        mask = np.ones_like(gray)
        mask[y1:y2, x1:x2] = 0
        bg_region = gray[mask == 1]
        bg_sharp  = sharpness(bg_region.reshape(-1, 1)) if bg_region.size > 0 else 0.0
        method    = "face"

    # Bokeh ratio: how much sharper the foreground is vs background
    ratio = fg_sharp / (bg_sharp + 1e-5)

    if ratio > 3.0:
        label      = "excellent"
        score      = 10.0
        suggestion = ""
    elif ratio > 1.5:
        label      = "good"
        score      = 7.5
        suggestion = ""
    elif ratio > 0.8:
        label      = "flat"
        score      = 5.0
        suggestion = "Background and subject appear equally sharp. Use portrait mode or move closer to your subject for better background separation."
    else:
        label      = "reverse_bokeh"
        score      = 3.0
        suggestion = "Background appears sharper than subject. Ensure your subject is in focus."

    return {
        "bokeh_label":       label,
        "bokeh_score":       round(score, 2),
        "fg_sharpness":      round(fg_sharp, 2),
        "bg_sharpness":      round(bg_sharp, 2),
        "bokeh_ratio":       round(ratio, 3),
        "bokeh_method":      method,
        "bokeh_suggestion":  suggestion,
    }
