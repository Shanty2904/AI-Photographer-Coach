"""
motion_blur.py
--------------
Detects motion blur and camera shake:
  - Uses FFT frequency analysis to detect directional blur patterns
  - Measures blur direction (horizontal / vertical / diagonal)
  - Distinguishes motion blur from general softness
  - Provides severity classification
"""

import cv2
import numpy as np


def _fft_blur_analysis(gray: np.ndarray) -> dict:
    """
    Analyse frequency spectrum via FFT.
    Motion blur concentrates energy along a narrow band in frequency space.
    """
    f       = np.fft.fft2(gray.astype(np.float32))
    fshift  = np.fft.fftshift(f)
    mag     = np.log(np.abs(fshift) + 1)

    h, w    = mag.shape
    cx, cy  = w // 2, h // 2

    # Sample energy in horizontal and vertical bands through center
    band       = 5
    h_energy   = float(np.mean(mag[cy-band:cy+band, :]))
    v_energy   = float(np.mean(mag[:, cx-band:cx+band]))
    total_energy = float(np.mean(mag))

    h_ratio = h_energy / (total_energy + 1e-5)
    v_ratio = v_energy / (total_energy + 1e-5)

    return {
        "h_energy_ratio": round(h_ratio, 4),
        "v_energy_ratio": round(v_ratio, 4),
        "total_energy":   round(total_energy, 4),
    }


def _directional_blur(gray: np.ndarray) -> str:
    """
    Determine primary blur direction using Sobel gradients.
    High horizontal gradient energy = vertical motion blur, and vice versa.
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    ex = float(np.mean(np.abs(gx)))
    ey = float(np.mean(np.abs(gy)))

    ratio = ex / (ey + 1e-5)
    if ratio > 1.5:
        return "vertical"       # strong horizontal edges → vertical motion
    elif ratio < 0.67:
        return "horizontal"     # strong vertical edges → horizontal motion
    else:
        return "diagonal/shake"


def analyze_motion_blur(frame: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Laplacian variance to detect overall blur
    lap_var  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    fft_data = _fft_blur_analysis(gray)

    # Detect motion blur specifically (vs soft focus)
    # Motion blur: very low lap_var AND skewed FFT energy
    energy_skew = abs(fft_data["h_energy_ratio"] - fft_data["v_energy_ratio"])
    is_motion_blur = lap_var < 200 and energy_skew > 0.03

    if is_motion_blur:
        direction = _directional_blur(gray)
    else:
        direction = "none"

    if lap_var > 500:
        label      = "none"
        score      = 10.0
        suggestion = ""
    elif lap_var > 200:
        label      = "slight"
        score      = 7.5
        suggestion = "Slight blur detected. Try tapping to lock focus and holding your breath when shooting."
    elif is_motion_blur:
        label      = "motion_blur"
        score      = 3.0
        suggestion = f"Motion blur detected ({direction} direction). Hold the camera steadier or increase shutter speed."
    else:
        label      = "camera_shake"
        score      = 4.0
        suggestion = "Camera shake detected. Brace your elbows against your body or use a surface to stabilise."

    return {
        "motion_blur_label":      label,
        "motion_blur_score":      round(score, 2),
        "blur_direction":         direction,
        "is_motion_blur":         is_motion_blur,
        "laplacian_variance":     round(lap_var, 2),
        "fft_h_ratio":            fft_data["h_energy_ratio"],
        "fft_v_ratio":            fft_data["v_energy_ratio"],
        "motion_blur_suggestion": suggestion,
    }
