"""
noise.py
--------
Detects image noise and grain:
  - Uses Laplacian residual method to estimate noise level
  - Classifies: clean / acceptable / noisy / very noisy
  - Common cause: low light + high ISO on phone cameras
"""

import cv2
import numpy as np


def _estimate_noise(gray: np.ndarray) -> float:
    """
    Estimate noise sigma using the method from:
    J. Immerkær, "Fast Noise Variance Estimation", 1996.
    Convolves with a high-frequency kernel and measures the result.
    """
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0

    kernel = np.array([
        [ 1, -2,  1],
        [-2,  4, -2],
        [ 1, -2,  1]
    ], dtype=np.float32)

    filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    sigma    = np.sqrt(np.pi / 2) * np.mean(np.abs(filtered)) / 6.0
    return float(sigma)


def _banding_detection(gray: np.ndarray) -> bool:
    """
    Detect horizontal banding (common in high-ISO phone shots).
    Looks for periodic horizontal patterns in row-average signal.
    """
    row_means = np.mean(gray.astype(np.float32), axis=1)
    diff      = np.diff(row_means)
    # High variance in row-to-row differences = banding
    return float(np.std(diff)) > 8.0


def analyze_noise(frame: np.ndarray) -> dict:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sigma = _estimate_noise(gray)
    banding = _banding_detection(gray)

    if sigma < 3.0:
        label      = "clean"
        score      = 10.0
        suggestion = ""
    elif sigma < 7.0:
        label      = "acceptable"
        score      = 8.0
        suggestion = ""
    elif sigma < 15.0:
        label      = "noisy"
        score      = 5.0
        suggestion = "Noticeable noise/grain detected. Move to a brighter environment or turn on more lights."
    else:
        label      = "very_noisy"
        score      = 2.0
        suggestion = "Heavy noise detected — likely caused by low light. Increase lighting significantly for a cleaner shot."

    if banding and score > 3.0:
        score      = max(2.0, score - 2.0)
        suggestion += " Horizontal banding detected — try a different angle or lighting source."

    return {
        "noise_label":      label,
        "noise_sigma":      round(sigma, 3),
        "noise_score":      round(score, 2),
        "banding_detected": banding,
        "noise_suggestion": suggestion.strip(),
    }
