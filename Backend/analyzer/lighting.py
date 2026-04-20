"""
lighting.py
-----------
Analyzes photographic lighting quality:
  - Exposure classification (underexposed / well-lit / overexposed)
  - Contrast (flat vs punchy vs blown-out)
  - Shadow harshness
  - Color temperature estimate (warm / neutral / cool)
  - Histogram statistics
"""

import cv2
import numpy as np


def _exposure_analysis(gray: np.ndarray) -> dict:
    """
    Classify exposure from average pixel brightness.
    Scale: 0 (black) → 255 (white)
    """
    avg = float(np.mean(gray))
    std = float(np.std(gray))

    if avg < 60:
        label = "underexposed"
        score = max(0.0, (avg / 60) * 6.0)       # scales 0–6
        suggestion = "Scene is too dark. Move to better light or increase exposure."
    elif avg > 200:
        label = "overexposed"
        score = max(0.0, 6.0 - ((avg - 200) / 55) * 6.0)
        suggestion = "Scene is too bright. Reduce light or add some shade."
    else:
        label = "well-exposed"
        # Peak score at avg ≈ 128 (mid-tone)
        score = 10.0 - abs(avg - 128) / 72 * 3.0
        suggestion = ""

    return {
        "exposure_label": label,
        "avg_brightness": round(avg, 2),
        "brightness_std": round(std, 2),
        "exposure_score": round(score, 2),
        "exposure_suggestion": suggestion,
    }


def _contrast_analysis(gray: np.ndarray) -> dict:
    """
    Measure contrast via standard deviation of pixel values.
    Low std → flat/foggy. High std → punchy or clipped.
    """
    std = float(np.std(gray))

    if std < 30:
        label = "flat"
        score = 4.0
        suggestion = "Image looks flat/hazy. Try adjusting contrast or finding better light direction."
    elif std > 90:
        label = "high-contrast"
        score = 6.5
        suggestion = "Very high contrast. Watch for clipped highlights or crushed shadows."
    else:
        label = "balanced"
        score = 10.0 - abs(std - 60) / 30 * 2.0
        suggestion = ""

    return {
        "contrast_label": label,
        "contrast_std": round(std, 2),
        "contrast_score": round(score, 2),
        "contrast_suggestion": suggestion,
    }


def _shadow_harshness(gray: np.ndarray) -> dict:
    """
    Detect harsh shadows by looking at the proportion of very dark pixels (<40)
    relative to mid-tones. A high ratio suggests deep, hard shadows.
    """
    total_pixels = gray.size
    shadow_pixels = int(np.sum(gray < 40))
    shadow_ratio = shadow_pixels / total_pixels

    is_harsh = shadow_ratio > 0.20
    score = max(0.0, 10.0 - shadow_ratio * 30)

    suggestion = ""
    if is_harsh:
        suggestion = "Harsh shadows detected. Try using a reflector, fill flash, or shoot in open shade."

    return {
        "shadow_ratio": round(shadow_ratio, 3),
        "harsh_shadows": is_harsh,
        "shadow_score": round(score, 2),
        "shadow_suggestion": suggestion,
    }


def _color_temperature_estimate(frame: np.ndarray) -> dict:
    """
    Rough color temperature estimation by comparing average R vs B channel.
    R > B → warm (golden hour), R ≈ B → neutral, B > R → cool (overcast/shade)
    """
    b_channel = float(np.mean(frame[:, :, 0]))  # OpenCV: BGR
    g_channel = float(np.mean(frame[:, :, 1]))
    r_channel = float(np.mean(frame[:, :, 2]))

    diff = r_channel - b_channel

    if diff > 20:
        temp_label = "warm"
        temp_note = "Warm golden tones — great for portraits and sunsets."
    elif diff < -20:
        temp_label = "cool"
        temp_note = "Cool blue tones — works for landscapes, moody scenes."
    else:
        temp_label = "neutral"
        temp_note = "Neutral color balance — versatile lighting."

    return {
        "color_temp_label": temp_label,
        "color_temp_note": temp_note,
        "avg_r": round(r_channel, 2),
        "avg_g": round(g_channel, 2),
        "avg_b": round(b_channel, 2),
    }


def _histogram_data(gray: np.ndarray) -> dict:
    """
    Returns a 16-bin normalised histogram for sparkline display in the app.
    Values sum to 1.0.
    """
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
    hist = hist.flatten()
    hist = hist / hist.sum()
    return {"histogram_bins": [round(float(v), 4) for v in hist]}


def analyze_lighting(frame: np.ndarray) -> dict:
    """
    Master lighting analysis. Returns a flat dict with all lighting metrics.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result = {}
    result.update(_exposure_analysis(gray))
    result.update(_contrast_analysis(gray))
    result.update(_shadow_harshness(gray))
    result.update(_color_temperature_estimate(frame))
    result.update(_histogram_data(gray))

    return result
