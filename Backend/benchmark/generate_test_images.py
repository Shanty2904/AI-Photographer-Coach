"""
generate_test_images.py
-----------------------
Creates synthetic test images with deterministic, known properties
for all 12 analyzer modules.  Each image is pixel-crafted so its
expected output labels are unambiguous.

Run once before benchmarking:
    python -m benchmark.generate_test_images

Output:
    benchmark/test_images/   — PNG images
    benchmark/ground_truth.json — labels per image
"""

import json
import math
import os
import sys

import cv2
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "test_images")
LABEL_FILE = os.path.join(os.path.dirname(__file__), "ground_truth.json")

W, H = 640, 480  # standard frame size


def _save(name: str, img: np.ndarray) -> str:
    path = os.path.join(OUT_DIR, name)
    cv2.imwrite(path, img)
    return path


def _base_texture(w=W, h=H) -> np.ndarray:
    """Mid-grey textured base — gives cascades something to work with."""
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    for y in range(0, h, 16):
        for x in range(0, w, 16):
            val = 128 + ((x // 16 + y // 16) % 2) * 30
            img[y:y+16, x:x+16] = val
    return img


# ─────────────────────────────────────────────
# 1. SHARPNESS
# ─────────────────────────────────────────────
def gen_sharpness():
    records = []

    # tack_sharp  — high-freq checkerboard
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            img[y, x] = 255 if (x // 4 + y // 4) % 2 == 0 else 0
    _save("sharp_tack.png", img)
    records.append({"file": "sharp_tack.png", "module": "sharpness",
                    "expected_label": "tack_sharp",
                    "expected_score_min": 9.0})

    # sharp  — fine grid
    img = _base_texture()
    for y in range(0, H, 8):
        img[y, :] = 200
    for x in range(0, W, 8):
        img[:, x] = 200
    _save("sharp_fine.png", img)
    records.append({"file": "sharp_fine.png", "module": "sharpness",
                    "expected_label": "sharp",
                    "expected_score_min": 7.0})

    # blurry  — apply heavy Gaussian
    img = _base_texture()
    for y in range(0, H, 8):
        img[y, :] = 200
    img = cv2.GaussianBlur(img, (31, 31), 0)
    _save("blurry.png", img)
    records.append({"file": "blurry.png", "module": "sharpness",
                    "expected_label": "blurry",
                    "expected_score_max": 5.0})

    # very_blurry
    img = _base_texture()
    img = cv2.GaussianBlur(img, (71, 71), 0)
    _save("very_blurry.png", img)
    records.append({"file": "very_blurry.png", "module": "sharpness",
                    "expected_label": "very_blurry",
                    "expected_score_max": 3.0})

    return records


# ─────────────────────────────────────────────
# 2. MOTION BLUR
# ─────────────────────────────────────────────
def gen_motion_blur():
    records = []

    # no blur — sharp checkerboard
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            img[y, x] = 255 if (x // 4 + y // 4) % 2 == 0 else 0
    _save("mb_none.png", img)
    records.append({"file": "mb_none.png", "module": "motion_blur",
                    "expected_label": "none",
                    "expected_score_min": 9.0})

    # motion blur — horizontal kernel
    img = _base_texture()
    for y in range(0, H, 8):
        img[y, :] = 220
    kernel = np.zeros((1, 61), np.float32)
    kernel[0, :] = 1.0 / 61
    img = cv2.filter2D(img, -1, kernel)
    _save("mb_horizontal.png", img)
    records.append({"file": "mb_horizontal.png", "module": "motion_blur",
                    "expected_label": "motion_blur",
                    "expected_score_max": 5.0})

    return records


# ─────────────────────────────────────────────
# 3. NOISE
# ─────────────────────────────────────────────
def gen_noise():
    records = []

    # clean — uniform grey
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    _save("noise_clean.png", img)
    records.append({"file": "noise_clean.png", "module": "noise",
                    "expected_label": "clean",
                    "expected_score_min": 9.0})

    # noisy — Gaussian noise sigma ~25
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    noise = np.random.normal(0, 25, (H, W, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    _save("noise_noisy.png", img)
    records.append({"file": "noise_noisy.png", "module": "noise",
                    "expected_label": "noisy",
                    "expected_score_max": 7.0})

    # very noisy — sigma ~60
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    noise = np.random.normal(0, 60, (H, W, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    _save("noise_very_noisy.png", img)
    records.append({"file": "noise_very_noisy.png", "module": "noise",
                    "expected_label": "very_noisy",
                    "expected_score_max": 4.0})

    return records


# ─────────────────────────────────────────────
# 4. LIGHTING / EXPOSURE
# ─────────────────────────────────────────────
def gen_lighting():
    records = []

    # underexposed
    img = np.full((H, W, 3), 30, dtype=np.uint8)
    _save("light_under.png", img)
    records.append({"file": "light_under.png", "module": "lighting",
                    "expected_exposure_label": "underexposed",
                    "expected_exposure_score_max": 5.0})

    # well-exposed (avg ~128)
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    _save("light_good.png", img)
    records.append({"file": "light_good.png", "module": "lighting",
                    "expected_exposure_label": "well-exposed",
                    "expected_exposure_score_min": 8.0})

    # overexposed
    img = np.full((H, W, 3), 240, dtype=np.uint8)
    _save("light_over.png", img)
    records.append({"file": "light_over.png", "module": "lighting",
                    "expected_exposure_label": "overexposed",
                    "expected_exposure_score_max": 5.0})

    # warm — R channel dominant
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :, 2] = 200   # R high
    img[:, :, 1] = 120
    img[:, :, 0] = 60    # B low
    _save("light_warm.png", img)
    records.append({"file": "light_warm.png", "module": "lighting",
                    "expected_color_temp": "warm"})

    # cool — B channel dominant
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :, 0] = 200   # B high
    img[:, :, 1] = 120
    img[:, :, 2] = 60    # R low
    _save("light_cool.png", img)
    records.append({"file": "light_cool.png", "module": "lighting",
                    "expected_color_temp": "cool"})

    # harsh shadows — >20% pixels very dark
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    img[:int(H * 0.35), :] = 20   # top third very dark
    _save("light_shadow.png", img)
    records.append({"file": "light_shadow.png", "module": "lighting",
                    "expected_harsh_shadows": True})

    # flat contrast
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    noise = np.random.normal(0, 8, (H, W, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    _save("light_flat.png", img)
    records.append({"file": "light_flat.png", "module": "lighting",
                    "expected_contrast_label": "flat"})

    return records


# ─────────────────────────────────────────────
# 5. HORIZON / TILT
# ─────────────────────────────────────────────
def gen_horizon():
    records = []

    def _horizon_image(angle_deg: float) -> np.ndarray:
        img = np.full((H, W, 3), 180, dtype=np.uint8)
        # Draw a thick horizontal line then rotate the whole image
        cv2.line(img, (0, H // 2), (W, H // 2), (50, 50, 50), 6)
        # Many parallel lines to give Hough transform plenty to detect
        for offset in range(-120, 121, 30):
            cv2.line(img, (0, H // 2 + offset), (W, H // 2 + offset),
                     (100, 100, 100), 2)
        cx, cy = W // 2, H // 2
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        return cv2.warpAffine(img, M, (W, H), borderValue=(180, 180, 180))

    _save("horizon_level.png", _horizon_image(0))
    records.append({"file": "horizon_level.png", "module": "horizon",
                    "expected_is_level": True,
                    "expected_score_min": 9.0})

    _save("horizon_slight.png", _horizon_image(4))
    records.append({"file": "horizon_slight.png", "module": "horizon",
                    "expected_is_level": False,
                    "expected_score_max": 8.0})

    _save("horizon_heavy.png", _horizon_image(12))
    records.append({"file": "horizon_heavy.png", "module": "horizon",
                    "expected_is_level": False,
                    "expected_score_max": 6.0})

    return records


# ─────────────────────────────────────────────
# 6. BOKEH
# ─────────────────────────────────────────────
def gen_bokeh():
    records = []

    # excellent bokeh — sharp centre circle, blurred surround
    img = np.full((H, W, 3), 60, dtype=np.uint8)
    # Blurred background (draw texture then blur)
    bg = _base_texture()
    bg = cv2.GaussianBlur(bg, (51, 51), 0)
    img = bg.copy()
    # Sharp foreground circle
    fg = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            if (x // 4 + y // 4) % 2 == 0:
                fg[y, x] = 240
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask, (W // 2, H // 2), min(W, H) // 4, 255, -1)
    img[mask == 255] = fg[mask == 255]
    _save("bokeh_excellent.png", img)
    records.append({"file": "bokeh_excellent.png", "module": "bokeh",
                    "expected_label": "excellent",
                    "expected_score_min": 9.0})

    # flat — uniform sharpness everywhere
    img = _base_texture()
    for y in range(0, H, 4):
        img[y, :] = 200
    for x in range(0, W, 4):
        img[:, x] = 200
    _save("bokeh_flat.png", img)
    records.append({"file": "bokeh_flat.png", "module": "bokeh",
                    "expected_label": "flat",
                    "expected_score_max": 7.0})

    return records


# ─────────────────────────────────────────────
# 7. COLOR HARMONY
# ─────────────────────────────────────────────
def gen_color_harmony():
    records = []

    # monochromatic — shades of blue
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for i, shade in enumerate([40, 80, 120, 160, 200]):
        x0, x1 = i * W // 5, (i + 1) * W // 5
        img[:, x0:x1] = (shade, 20, 10)   # BGR: mostly blue
    _save("color_mono.png", img)
    records.append({"file": "color_mono.png", "module": "color_harmony",
                    "expected_label": "monochromatic"})

    # complementary — red + cyan (opposite on wheel)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :W // 2] = (20, 20, 200)      # Red half (BGR)
    img[:, W // 2:] = (200, 200, 20)     # Cyan half (BGR)
    _save("color_complementary.png", img)
    records.append({"file": "color_complementary.png", "module": "color_harmony",
                    "expected_label": "complementary"})

    # warm color cast
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :, 2] = 210    # R dominant
    img[:, :, 1] = 100
    img[:, :, 0] = 40
    _save("color_warm_cast.png", img)
    records.append({"file": "color_warm_cast.png", "module": "color_harmony",
                    "expected_color_cast": "red/warm"})

    return records


# ─────────────────────────────────────────────
# 8. COMPOSITION — leading lines & symmetry
# (subject/thirds needs YOLO/face — tested separately)
# ─────────────────────────────────────────────
def gen_composition():
    records = []

    # symmetric — left == right
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    for x in range(0, W // 2, 30):
        img[:, x:x+15] = 180
        img[:, W - x - 15:W - x] = 180
    _save("comp_symmetric.png", img)
    records.append({"file": "comp_symmetric.png", "module": "composition",
                    "expected_is_symmetric": True,
                    "expected_symmetry_score_min": 7.0})

    # asymmetric
    img = np.full((H, W, 3), 50, dtype=np.uint8)
    img[:, :W // 3] = 220
    _save("comp_asymmetric.png", img)
    records.append({"file": "comp_asymmetric.png", "module": "composition",
                    "expected_is_symmetric": False})

    # leading lines — many diagonal lines
    img = np.full((H, W, 3), 180, dtype=np.uint8)
    for i in range(0, W, 40):
        cv2.line(img, (i, 0), (i + 60, H), (40, 40, 40), 3)
    _save("comp_leading_lines.png", img)
    records.append({"file": "comp_leading_lines.png", "module": "composition",
                    "expected_has_leading_lines": True})

    return records


# ─────────────────────────────────────────────
# 9. FOREGROUND-BACKGROUND SEPARATION
# ─────────────────────────────────────────────
def gen_fg_bg():
    records = []

    # strong separation — dark bg, bright fg centre
    img = np.full((H, W, 3), 30, dtype=np.uint8)   # dark bg (BGR dark)
    pad = 60
    img[pad:H-pad, pad:W-pad] = (200, 50, 50)      # bright blue fg
    _save("fgbg_strong.png", img)
    records.append({"file": "fgbg_strong.png", "module": "fg_bg",
                    "expected_label": "strong",
                    "expected_score_min": 9.0})

    # camouflaged — same color fg and bg
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    img[80:H-80, 80:W-80] = 132    # nearly identical
    _save("fgbg_camouflaged.png", img)
    records.append({"file": "fgbg_camouflaged.png", "module": "fg_bg",
                    "expected_label": "camouflaged",
                    "expected_score_max": 4.0})

    # distracting background — bright spots in bg
    img = np.full((H, W, 3), 80, dtype=np.uint8)
    for bx in range(20, W, 60):
        for by in range(20, H, 60):
            cv2.circle(img, (bx, by), 8, (245, 245, 245), -1)
    _save("fgbg_distracting.png", img)
    records.append({"file": "fgbg_distracting.png", "module": "fg_bg",
                    "expected_bg_distracting": True})

    return records


# ─────────────────────────────────────────────
# 10. ASPECT RATIO
# ─────────────────────────────────────────────
def gen_aspect_ratio():
    records = []

    sizes = [
        ("ar_landscape_16x9.png", 1280, 720, "16:9"),
        ("ar_portrait_9x16.png",  720, 1280, "9:16"),
        ("ar_square.png",         640,  640, "1:1"),
        ("ar_standard_4x3.png",   640,  480, "4:3"),
    ]
    for fname, w, h, expected in sizes:
        img = np.full((h, w, 3), 128, dtype=np.uint8)
        cv2.line(img, (0, h//2), (w, h//2), (80, 80, 80), 3)
        _save(fname, img)
        records.append({"file": fname, "module": "aspect_ratio",
                        "expected_ratio_label": expected})

    return records


# ─────────────────────────────────────────────
# 11 & 12. FACE EXPRESSION + SUBJECT
# Pure-synthetic Haar triggers are unreliable.
# We generate placeholder records; benchmark
# skips accuracy check but still runs and
# reports "no_face" as expected for these.
# ─────────────────────────────────────────────
def gen_face_placeholders():
    records = []

    # No-face image — uniform grey
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    _save("face_none.png", img)
    records.append({"file": "face_none.png", "module": "face_expression",
                    "expected_faces": 0,
                    "expected_eye_label": "no_face"})

    records.append({"file": "face_none.png", "module": "subject",
                    "expected_subject_detected": False})

    return records


# ─────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────
def main():
    np.random.seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)

    all_records = []
    generators = [
        ("sharpness",      gen_sharpness),
        ("motion_blur",    gen_motion_blur),
        ("noise",          gen_noise),
        ("lighting",       gen_lighting),
        ("horizon",        gen_horizon),
        ("bokeh",          gen_bokeh),
        ("color_harmony",  gen_color_harmony),
        ("composition",    gen_composition),
        ("fg_bg",          gen_fg_bg),
        ("aspect_ratio",   gen_aspect_ratio),
        ("face/subject",   gen_face_placeholders),
    ]

    for name, fn in generators:
        recs = fn()
        all_records.extend(recs)
        print(f"  OK  {name:20s}  ({len(recs)} images)")

    with open(LABEL_FILE, "w") as f:
        json.dump(all_records, f, indent=2)

    print(f"\n  {len(all_records)} test cases written to benchmark/ground_truth.json")
    print(f"  Images saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
