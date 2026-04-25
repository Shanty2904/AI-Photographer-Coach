"""
face_expression.py
------------------
Analyses face-related photography quality:
  - Eye openness (closed eyes = bad shot)
  - Gaze direction / eye contact with camera
  - Face angle (frontal vs profile)
  - Smile detection
  - Multiple faces handling

Face detection strategy (most → least strict, stops on first hit):
  1. haarcascade_frontalface_default.xml  — high precision
  2. haarcascade_frontalface_alt.xml       — better for partial occlusion
  3. haarcascade_frontalface_alt2.xml      — different training set
  4. haarcascade_profileface.xml           — side-on faces
     + horizontal flip of frame            — other profile direction

Eye detection strategy (glasses-aware):
  1. haarcascade_eye_tree_eyeglasses.xml  — handles spectacles
  2. haarcascade_eye.xml                  — bare eyes fallback
  Three passes with progressively relaxed parameters.

CLAHE is applied to the full greyscale frame before any detection.
"""

import cv2
import numpy as np

# ── Face cascades ────────────────────────────────────────────
_face_cascades = [
    cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
    cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"),
    cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"),
]
_profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

# ── Eye cascades ─────────────────────────────────────────────
_eye_glasses_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ── Smile cascade ────────────────────────────────────────────
_smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

# ── CLAHE — applied to full frame before detection ───────────
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ─────────────────────────────────────────────────────────────
# FACE DETECTION
# ─────────────────────────────────────────────────────────────

def _detect_faces(enhanced_gray: np.ndarray):
    """
    Try frontal cascades first (progressively relaxed), then profile
    (normal + horizontally flipped frame) as a last resort.
    Returns list of (x, y, w, h) tuples in frame coordinates.
    """
    h, w = enhanced_gray.shape

    # ── Pass 1-3: frontal cascades, two sensitivity levels each ──
    for cascade in _face_cascades:
        for (min_n, min_sz) in [(4, 30), (2, 20)]:
            faces = cascade.detectMultiScale(
                enhanced_gray,
                scaleFactor=1.05,
                minNeighbors=min_n,
                minSize=(min_sz, min_sz),
            )
            if len(faces) > 0:
                return [tuple(f) for f in faces]

    # ── Pass 4: profile cascade (normal + flipped) ────────────
    for flipped in (False, True):
        src = cv2.flip(enhanced_gray, 1) if flipped else enhanced_gray
        faces = _profile_cascade.detectMultiScale(
            src, scaleFactor=1.05, minNeighbors=2, minSize=(20, 20)
        )
        if len(faces) > 0:
            result = []
            for (fx, fy, fw, fh) in faces:
                # Mirror x-coordinate back if we used the flipped image
                ox = (w - fx - fw) if flipped else fx
                result.append((ox, fy, fw, fh))
            return result

    return []


# ─────────────────────────────────────────────────────────────
# EYE DETECTION
# ─────────────────────────────────────────────────────────────

def _preprocess_roi(gray_roi: np.ndarray) -> np.ndarray:
    """CLAHE + bilateral filter on a small ROI."""
    enhanced = _clahe.apply(gray_roi)
    return cv2.bilateralFilter(enhanced, d=5, sigmaColor=30, sigmaSpace=30)


def _run_eye_cascades(roi: np.ndarray, scale: float, neighbors: int,
                      min_size: int, dedup_px: int) -> list:
    """
    Run glasses cascade then bare-eye cascade.
    Deduplicates by centre proximity (dedup_px radius).
    """
    detected: list[tuple[int, int]] = []
    results = []

    for cascade in (_eye_glasses_cascade, _eye_cascade):
        eyes = cascade.detectMultiScale(
            roi,
            scaleFactor=scale,
            minNeighbors=neighbors,
            minSize=(min_size, min_size),
        )
        for (ex, ey, ew, eh) in (eyes if len(eyes) > 0 else []):
            cx, cy = ex + ew // 2, ey + eh // 2
            if not any(abs(cx - ox) < dedup_px and abs(cy - oy) < dedup_px
                       for ox, oy in detected):
                detected.append((cx, cy))
                results.append((ex, ey, ew, eh))

    return results


def _eye_analysis(gray: np.ndarray, fx: int, fy: int, fw: int, fh: int) -> dict:
    """
    Detect eyes in the upper 65 % of the face bounding box.
    Three passes with progressively relaxed parameters.
    Dedup radius scales with face width (12 % of fw).
    """
    h_frame, w_frame = gray.shape
    # Clamp ROI to frame bounds
    fy_safe = max(0, fy)
    fy_end  = min(h_frame, fy + int(fh * 0.65))
    fx_safe = max(0, fx)
    fx_end  = min(w_frame, fx + fw)

    upper_gray = gray[fy_safe:fy_end, fx_safe:fx_end]
    if upper_gray.size == 0:
        return {"eyes_detected": 0, "eye_label": "eyes_closed_or_hidden",
                "eye_score": 3.0, "eye_suggestion": "Could not read eye region."}

    upper = _preprocess_roi(upper_gray)
    dedup = max(15, int(fw * 0.12))   # proportional to face size

    # Pass 1 — normal
    eyes = _run_eye_cascades(upper, scale=1.1, neighbors=4, min_size=15, dedup_px=dedup)

    # Pass 2 — relaxed
    if len(eyes) < 2:
        relaxed = _run_eye_cascades(upper, scale=1.05, neighbors=2,
                                    min_size=12, dedup_px=dedup)
        existing = [(ex + ew // 2, ey + eh // 2) for ex, ey, ew, eh in eyes]
        for (ex, ey, ew, eh) in relaxed:
            cx, cy = ex + ew // 2, ey + eh // 2
            if not any(abs(cx - ox) < dedup and abs(cy - oy) < dedup
                       for ox, oy in existing):
                eyes.append((ex, ey, ew, eh))
                existing.append((cx, cy))

    # Pass 3 — last resort (glasses cascade only, very relaxed)
    if len(eyes) < 1:
        last = _eye_glasses_cascade.detectMultiScale(
            upper, scaleFactor=1.03, minNeighbors=1, minSize=(8, 8)
        )
        eyes = list(last) if len(last) > 0 else eyes

    num_eyes = min(len(eyes), 2)

    if num_eyes >= 2:
        return {"eyes_detected": 2, "eye_label": "both_open",
                "eye_score": 10.0, "eye_suggestion": ""}
    elif num_eyes == 1:
        return {"eyes_detected": 1, "eye_label": "one_eye",
                "eye_score": 6.0,
                "eye_suggestion": "Only one eye clearly visible — subject may be looking to the side."}
    else:
        return {"eyes_detected": 0, "eye_label": "eyes_closed_or_hidden",
                "eye_score": 3.0,
                "eye_suggestion": "Eyes not detected — subject may have blinked. Try another shot."}


# ─────────────────────────────────────────────────────────────
# SMILE + FACE ANGLE
# ─────────────────────────────────────────────────────────────

def _smile_analysis(gray: np.ndarray, fx: int, fy: int, fw: int, fh: int) -> dict:
    """Detect smile in lower half of face."""
    h_frame, w_frame = gray.shape
    y0 = min(h_frame, fy + fh // 2)
    y1 = min(h_frame, fy + fh)
    lower_face = gray[y0:y1, max(0, fx):min(w_frame, fx + fw)]
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
    Detect whether the face is in profile by running the profile
    cascade on the face region. Also try the flipped version.
    """
    h_frame, w_frame = gray.shape
    face_region = gray[max(0, fy):min(h_frame, fy + fh),
                       max(0, fx):min(w_frame, fx + fw)]

    is_profile = False
    for src in (face_region, cv2.flip(face_region, 1)):
        profiles = _profile_cascade.detectMultiScale(
            src, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15)
        )
        if len(profiles) > 0:
            is_profile = True
            break

    return {
        "face_is_profile": is_profile,
        "face_angle_note": ("Subject appears to be in profile view."
                            if is_profile else "Subject is facing camera."),
    }


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def analyze_face_expression(frame: np.ndarray) -> dict:
    # Apply CLAHE to full frame — dramatically improves detection
    # in uneven lighting (backlit, side-lit, indoor artificial light)
    raw_gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced    = _clahe.apply(raw_gray)

    faces = _detect_faces(enhanced)

    if not faces:
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

    # Analyse the primary (largest) face using the RAW gray image
    # for eye/smile/angle — the enhanced image was only used to find faces
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])

    eye_data    = _eye_analysis(raw_gray, fx, fy, fw, fh)
    smile_data  = _smile_analysis(raw_gray, fx, fy, fw, fh)
    angle_data  = _face_angle(raw_gray, fx, fy, fw, fh)

    base_score = eye_data["eye_score"]
    if angle_data["face_is_profile"]:
        base_score = min(base_score, 7.0)

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
