"""
color_harmony.py
----------------
Analyses color harmony and dominant palette:
  - Extracts dominant colors using K-Means clustering
  - Classifies harmony type: complementary, analogous, monochromatic, triadic
  - Scores based on how well colors work together
  - Detects color cast issues
"""

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def _rgb_to_hsv_single(r: float, g: float, b: float):
    """Convert 0-255 RGB to HSV (H: 0-360, S: 0-1, V: 0-1)."""
    arr = np.array([[[int(b), int(g), int(r)]]], dtype=np.uint8)
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[0][0]
    return float(h) * 2, float(s) / 255.0, float(v) / 255.0  # OpenCV H is 0-180


def _dominant_colors(frame: np.ndarray, k: int = 5):
    """Extract k dominant colors using MiniBatchKMeans."""
    small  = cv2.resize(frame, (100, 100))
    pixels = small.reshape(-1, 3).astype(np.float32)
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
    kmeans.fit(pixels)

    centers = kmeans.cluster_centers_.astype(int)  # BGR
    labels  = kmeans.labels_
    counts  = np.bincount(labels, minlength=k)
    order   = np.argsort(-counts)

    colors = []
    for i in order:
        b, g, r = centers[i]
        h, s, v = _rgb_to_hsv_single(r, g, b)
        colors.append({
            "rgb":    [int(r), int(g), int(b)],
            "hex":    f"#{int(r):02x}{int(g):02x}{int(b):02x}",
            "hue":    round(h, 1),
            "sat":    round(s, 3),
            "val":    round(v, 3),
            "weight": round(float(counts[i]) / len(labels), 3),
        })
    return colors


def _harmony_type(hues: list[float]) -> tuple[str, float]:
    """
    Classify harmony from list of dominant hues.
    Returns (harmony_label, score).
    """
    if len(hues) < 2:
        return "monochromatic", 7.0

    # Compute pairwise angular differences on the color wheel
    diffs = []
    for i in range(len(hues)):
        for j in range(i + 1, len(hues)):
            d = abs(hues[i] - hues[j])
            d = min(d, 360 - d)
            diffs.append(d)

    avg_diff = float(np.mean(diffs))
    max_diff = float(np.max(diffs))

    if avg_diff < 30:
        return "monochromatic", 8.0     # Very similar hues
    elif avg_diff < 60:
        return "analogous", 9.0         # Adjacent hues — very pleasing
    elif 150 < max_diff < 210:
        return "complementary", 9.5     # Opposite hues — high impact
    elif max_diff > 110 and len(hues) >= 3:
        return "triadic", 8.5           # Three spread hues
    elif avg_diff < 90:
        return "split-complementary", 8.0
    else:
        return "discordant", 4.0        # Random hues — visually jarring


def _color_cast(frame: np.ndarray) -> dict:
    """Detect strong color cast (too much of one channel)."""
    b = float(np.mean(frame[:, :, 0]))
    g = float(np.mean(frame[:, :, 1]))
    r = float(np.mean(frame[:, :, 2]))
    total = b + g + r + 1e-5

    rb = r / total
    gb = g / total
    bb = b / total

    cast = "none"
    cast_suggestion = ""
    if rb > 0.40:
        cast = "red/warm"
        cast_suggestion = "Strong warm/red cast detected. Consider adjusting white balance."
    elif bb > 0.40:
        cast = "blue/cool"
        cast_suggestion = "Strong blue/cool cast detected. Consider warming up the white balance."
    elif gb > 0.40:
        cast = "green"
        cast_suggestion = "Green cast detected — common under fluorescent lighting."

    return {"color_cast": cast, "color_cast_suggestion": cast_suggestion}


def analyze_color_harmony(frame: np.ndarray) -> dict:
    try:
        from sklearn.cluster import MiniBatchKMeans as _check
    except ImportError:
        return {
            "harmony_label": "unknown",
            "harmony_score": 5.0,
            "dominant_colors": [],
            "color_cast": "unknown",
            "color_cast_suggestion": "Install scikit-learn for color analysis: pip install scikit-learn",
            "harmony_suggestion": "",
        }

    colors = _dominant_colors(frame, k=5)

    # Only consider colors with meaningful saturation (ignore near-greys)
    saturated = [c for c in colors if c["sat"] > 0.15]
    hues = [c["hue"] for c in saturated[:4]] if saturated else []

    harmony_label, harmony_score = _harmony_type(hues)

    suggestion = ""
    if harmony_label == "discordant":
        suggestion = "Color palette looks clashing. Try simplifying the colors in your scene for a more harmonious look."

    result = {
        "harmony_label":     harmony_label,
        "harmony_score":     round(harmony_score, 2),
        "dominant_colors":   colors[:5],
        "harmony_suggestion": suggestion,
    }
    result.update(_color_cast(frame))
    return result
