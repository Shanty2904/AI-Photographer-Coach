"""
aspect_ratio.py
---------------
Analyses framing and suggests optimal crop ratios:
  - Detects current aspect ratio
  - Suggests best crop for detected content (portrait, landscape, square)
  - Checks headroom (space above head) for portrait shots
  - Checks if subject is being cut off at edges
"""

import cv2
import numpy as np

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Common aspect ratios and their names
ASPECT_RATIOS = {
    "1:1":   1.000,   # Square (Instagram)
    "4:3":   1.333,   # Standard photo
    "3:2":   1.500,   # DSLR standard
    "16:9":  1.778,   # Widescreen
    "9:16":  0.563,   # Portrait/Stories
    "4:5":   0.800,   # Instagram portrait
}


def _nearest_ratio(w: int, h: int) -> str:
    actual = w / h
    return min(ASPECT_RATIOS, key=lambda k: abs(ASPECT_RATIOS[k] - actual))


def _headroom_analysis(frame: np.ndarray, faces) -> dict:
    """
    Check headroom — space between top of head and frame edge.
    Too much headroom: subject looks small/lost.
    Too little: claustrophobic, head cut off.
    """
    h, w = frame.shape[:2]

    if len(faces) == 0:
        return {
            "headroom_ratio":      None,
            "headroom_label":      "no_face",
            "headroom_suggestion": "",
        }

    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    top_of_head  = fy
    headroom     = top_of_head / h   # ratio of frame above head

    if headroom < 0.03:
        label      = "too_tight"
        suggestion = "Head is too close to the top edge. Leave a bit more space above the subject."
    elif headroom < 0.12:
        label      = "good"
        suggestion = ""
    elif headroom < 0.25:
        label      = "generous"
        suggestion = ""
    else:
        label      = "too_much"
        suggestion = "Too much empty space above subject. Reframe downward or zoom in slightly."

    return {
        "headroom_ratio":      round(headroom, 3),
        "headroom_label":      label,
        "headroom_suggestion": suggestion,
    }


def _edge_cutoff(frame: np.ndarray, faces) -> dict:
    """Check if subject face is being cut off at any edge."""
    if len(faces) == 0:
        return {"subject_cutoff": False, "cutoff_suggestion": ""}

    h, w  = frame.shape[:2]
    margin = 0.03   # 3% from edge counts as cutoff

    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    cutoff = (
        fx < w * margin or
        fy < h * margin or
        (fx + fw) > w * (1 - margin) or
        (fy + fh) > h * (1 - margin)
    )

    return {
        "subject_cutoff": cutoff,
        "cutoff_suggestion": "Subject is being cut off at the frame edge. Reframe to include the full subject." if cutoff else "",
    }


def _suggest_crop(frame: np.ndarray, faces) -> str:
    """Suggest best crop ratio based on content."""
    h, w   = frame.shape[:2]
    is_portrait_orientation = h > w

    if len(faces) > 0:
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_fills_height = fh / h > 0.4

        if face_fills_height:
            return "4:5 or 9:16 for close portrait"
        elif is_portrait_orientation:
            return "4:5 for Instagram portrait or 9:16 for Stories"
        else:
            return "3:2 for standard portrait or 1:1 for square"
    else:
        if is_portrait_orientation:
            return "9:16 for Stories/Reels or 4:5 for Instagram"
        else:
            return "16:9 for landscape/cinematic or 3:2 for standard"


def analyze_aspect_ratio(frame: np.ndarray) -> dict:
    h, w   = frame.shape[:2]
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces  = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    current_ratio = _nearest_ratio(w, h)
    headroom_data = _headroom_analysis(frame, faces)
    cutoff_data   = _edge_cutoff(frame, faces)
    crop_suggest  = _suggest_crop(frame, faces)

    # Score based on headroom and cutoff
    score = 8.0   # default neutral
    if cutoff_data["subject_cutoff"]:
        score -= 3.0
    if headroom_data["headroom_label"] in ("too_tight", "too_much"):
        score -= 1.5

    suggestion = (
        cutoff_data["cutoff_suggestion"] or
        headroom_data["headroom_suggestion"] or
        ""
    )

    return {
        "aspect_ratio_label":   current_ratio,
        "frame_width":          w,
        "frame_height":         h,
        "aspect_ratio_score":   round(max(0.0, score), 2),
        "crop_suggestion":      crop_suggest,
        "aspect_ratio_suggestion": suggestion,
        **headroom_data,
        **cutoff_data,
    }
