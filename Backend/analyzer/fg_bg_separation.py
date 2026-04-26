from __future__ import annotations
"""
fg_bg_separation.py
-------------------
Analyses foreground-background separation quality:
  - Uses GrabCut (when face available) or edge-based segmentation
  - Measures contrast between subject and background
  - Detects subject-background color similarity (camouflage problem)
  - Checks background distraction level
"""

import cv2
import numpy as np

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _grabcut_separation(frame: np.ndarray, fx: int, fy: int, fw: int, fh: int) -> float:
    """
    Use GrabCut with face bounding box as hint.
    Returns ratio of foreground pixels.
    """
    try:
        mask   = np.zeros(frame.shape[:2], np.uint8)
        bgd    = np.zeros((1, 65), np.float64)
        fgd    = np.zeros((1, 65), np.float64)
        # Expand rect slightly
        pad    = int(min(fw, fh) * 0.3)
        h, w   = frame.shape[:2]
        rect   = (
            max(0, fx - pad),
            max(0, fy - pad),
            min(fw + pad * 2, w - fx),
            min(fh + pad * 2, h - fy)
        )
        cv2.grabCut(frame, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        return float(np.mean(fg_mask))
    except Exception:
        return 0.3   # fallback


def _color_contrast(frame: np.ndarray, faces) -> dict:
    """
    Compare average color of subject region vs background.
    Low contrast = subject blends into background.
    """
    h, w  = frame.shape[:2]
    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

    if len(faces) > 0:
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        subj_region = lab[fy:fy+fh, fx:fx+fw]
        # Background: frame excluding face region
        mask = np.ones((h, w), dtype=bool)
        mask[fy:fy+fh, fx:fx+fw] = False
        bg_pixels = lab[mask]
    else:
        # No face: compare center vs edges
        margin      = h // 5
        subj_region = lab[margin:-margin, margin:-margin]
        bg_pixels   = lab[:margin, :].reshape(-1, 3)

    subj_mean = np.mean(subj_region.reshape(-1, 3), axis=0)
    bg_mean   = np.mean(bg_pixels.reshape(-1, 3), axis=0) if bg_pixels.size > 0 else subj_mean

    # Euclidean distance in Lab space (perceptual color difference)
    delta_e = float(np.linalg.norm(subj_mean - bg_mean))

    if delta_e > 40:
        sep_label      = "strong"
        sep_score      = 10.0
        sep_suggestion = ""
    elif delta_e > 20:
        sep_label      = "moderate"
        sep_score      = 7.5
        sep_suggestion = ""
    elif delta_e > 10:
        sep_label      = "weak"
        sep_score      = 5.0
        sep_suggestion = "Subject and background have similar colors. Try a background with more contrast to your subject."
    else:
        sep_label      = "camouflaged"
        sep_score      = 2.0
        sep_suggestion = "Subject blends into background. Move subject against a contrasting background."

    return {
        "color_separation_label":   sep_label,
        "color_delta_e":            round(delta_e, 2),
        "color_separation_score":   sep_score,
        "color_separation_suggestion": sep_suggestion,
    }


def _background_distraction(frame: np.ndarray, faces) -> dict:
    """
    Check if background has distracting bright spots or busy patterns.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    if len(faces) > 0:
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        mask = np.ones((h, w), dtype=bool)
        mask[fy:fy+fh, fx:fx+fw] = False
        bg = gray[mask]
    else:
        bg = gray.flatten()

    # Bright spots in background
    bright_ratio = float(np.sum(bg > 230)) / (bg.size + 1e-5)
    is_distracting = bright_ratio > 0.05

    suggestion = ""
    if is_distracting:
        suggestion = "Bright spots detected in background. Reframe to avoid highlights behind your subject."

    return {
        "bg_bright_ratio":           round(bright_ratio, 4),
        "bg_distracting":            is_distracting,
        "bg_distraction_suggestion": suggestion,
    }


def analyze_fg_bg_separation(frame: np.ndarray) -> dict:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    color_data = _color_contrast(frame, faces)
    bg_data    = _background_distraction(frame, faces)

    # Composite score
    base_score = color_data["color_separation_score"]
    if bg_data["bg_distracting"]:
        base_score = max(0.0, base_score - 1.5)

    suggestion = color_data["color_separation_suggestion"] or bg_data["bg_distraction_suggestion"]

    return {
        "fg_bg_score":       round(base_score, 2),
        "fg_bg_suggestion":  suggestion,
        **color_data,
        **bg_data,
    }
