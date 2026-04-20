"""
horizon.py
----------
Detects camera tilt and horizon alignment using:
  - Canny edge detection
  - Probabilistic Hough Line Transform
  - Angle statistics of dominant horizontal lines
"""

import cv2
import numpy as np


def _detect_dominant_angle(gray: np.ndarray) -> float | None:
    """
    Returns the median angle (in degrees) of the most dominant
    near-horizontal lines in the frame, or None if no lines found.
    Angle convention: 0° = perfectly horizontal.
    """
    # Blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=50,
        maxLineGap=15
    )

    if lines is None:
        return None

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue   # skip vertical lines
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Keep only near-horizontal lines (within ±45°)
        if abs(angle) <= 45:
            angles.append(angle)

    if not angles:
        return None

    return float(np.median(angles))


def analyze_horizon(frame: np.ndarray) -> dict:
    """
    Analyse camera tilt.
    Returns tilt angle, whether the frame is level, and a suggestion.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    angle = _detect_dominant_angle(gray)

    if angle is None:
        return {
            "tilt_angle": 0.0,
            "is_level": True,
            "horizon_score": 7.0,        # neutral — can't determine
            "horizon_suggestion": "Could not detect horizon lines. Ensure the scene has clear horizontal elements.",
        }

    abs_angle = abs(angle)

    if abs_angle <= 2.0:
        is_level = True
        score = 10.0
        suggestion = ""
    elif abs_angle <= 5.0:
        is_level = False
        score = 7.0
        direction = "counter-clockwise" if angle > 0 else "clockwise"
        suggestion = f"Slight tilt of {abs_angle:.1f}°. Rotate camera {direction} a little."
    else:
        is_level = False
        score = max(0.0, 10.0 - abs_angle * 0.8)
        direction = "counter-clockwise" if angle > 0 else "clockwise"
        suggestion = f"Significant tilt of {abs_angle:.1f}°. Rotate camera {direction} to level the horizon."

    return {
        "tilt_angle": round(angle, 2),
        "tilt_abs": round(abs_angle, 2),
        "is_level": is_level,
        "horizon_score": round(score, 2),
        "horizon_suggestion": suggestion,
    }
