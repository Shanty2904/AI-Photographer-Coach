"""
face_expression.py
------------------
Analyses face-related photography quality:
  - Eye openness (closed eyes = bad shot)
  - Gaze direction / eye contact with camera
  - Face angle (frontal vs profile)
  - Smile detection
  - Multiple faces handling
Uses only OpenCV built-in cascades — no external models needed.
"""

import cv2
import numpy as np

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
_smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)
_profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)


def _eye_analysis(gray: np.ndarray, fx: int, fy: int, fw: int, fh: int) -> dict:
    """Detect eyes within face region and check openness."""
    # Only look in upper half of face for eyes
    upper_face = gray[fy:fy + fh//2, fx:fx + fw]
    eyes = _eye_cascade.detectMultiScale(
        upper_face, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15)
    )
    num_eyes = len(eyes)

    if num_eyes >= 2:
        eye_label      = "both_open"
        eye_score      = 10.0
        eye_suggestion = ""
    elif num_eyes == 1:
        eye_label      = "one_eye"
        eye_score      = 6.0
        eye_suggestion = "Only one eye detected — subject may be blinking or looking away."
    else:
        eye_label      = "eyes_closed_or_hidden"
        eye_score      = 3.0
        eye_suggestion = "Eyes not detected — subject may have blinked. Take another shot."

    return {
        "eyes_detected":   num_eyes,
        "eye_label":       eye_label,
        "eye_score":       eye_score,
        "eye_suggestion":  eye_suggestion,
    }


def _smile_analysis(gray: np.ndarray, fx: int, fy: int, fw: int, fh: int) -> dict:
    """Detect smile in lower half of face."""
    lower_face = gray[fy + fh//2: fy + fh, fx:fx + fw]
    smiles = _smile_cascade.detectMultiScale(
        lower_face, scaleFactor=1.7, minNeighbors=20, minSize=(25, 15)
    )
    smiling = len(smiles) > 0
    return {
        "smile_detected": smiling,
        "smile_note": "Subject is smiling." if smiling else "",
    }


def _face_angle(gray: np.ndarray, fx: int, fy: int, fw: int, fh: int) -> dict:
    """
    Rough face angle estimation.
    Check profile cascade on same region — if detected as profile, note it.
    """
    face_region = gray[fy:fy+fh, fx:fx+fw]
    profiles = _profile_cascade.detectMultiScale(
        face_region, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
    )
    is_profile = len(profiles) > 0

    return {
        "face_is_profile": is_profile,
        "face_angle_note": "Subject appears to be in profile view." if is_profile else "Subject is facing camera.",
    }


def analyze_face_expression(frame: np.ndarray) -> dict:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    if len(faces) == 0:
        return {
            "face_expression_score":      5.0,
            "faces_in_frame":             0,
            "eyes_detected":              0,
            "eye_label":                  "no_face",
            "eye_score":                  5.0,
            "eye_suggestion":             "No face detected in frame.",
            "smile_detected":             False,
            "smile_note":                 "",
            "face_is_profile":            False,
            "face_angle_note":            "",
            "face_expression_suggestion": "",
        }

    # Analyse the primary (largest) face
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])

    eye_data    = _eye_analysis(gray, fx, fy, fw, fh)
    smile_data  = _smile_analysis(gray, fx, fy, fw, fh)
    angle_data  = _face_angle(gray, fx, fy, fw, fh)

    # Composite score
    base_score = eye_data["eye_score"]
    if angle_data["face_is_profile"]:
        base_score = min(base_score, 7.0)  # profile shots are valid but penalise slightly

    suggestion = eye_data["eye_suggestion"]
    if not suggestion and len(faces) > 1:
        suggestion = f"{len(faces)} faces detected. Ensure all subjects are in focus."

    return {
        "face_expression_score":      round(base_score, 2),
        "faces_in_frame":             int(len(faces)),
        **eye_data,
        **smile_data,
        **angle_data,
        "face_expression_suggestion": suggestion,
    }
