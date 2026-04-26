from __future__ import annotations
"""
test_pipeline.py
----------------
Smoke test for the full 12-module CV + LLM pipeline.
Run this BEFORE testing with the phone.

Usage:
    python test_pipeline.py                    # uses webcam
    python test_pipeline.py path/to/image.jpg  # uses a static image
"""

import sys
import cv2
import numpy as np

from analyzer.composition      import analyze_composition
from analyzer.lighting         import analyze_lighting
from analyzer.horizon          import analyze_horizon
from analyzer.subject          import analyze_subject
from analyzer.sharpness        import analyze_sharpness
from analyzer.motion_blur      import analyze_motion_blur
from analyzer.noise            import analyze_noise
from analyzer.bokeh            import analyze_bokeh
from analyzer.color_harmony    import analyze_color_harmony
from analyzer.face_expression  import analyze_face_expression
from analyzer.fg_bg_separation import analyze_fg_bg_separation
from analyzer.aspect_ratio     import analyze_aspect_ratio
from analyzer.scorer           import compute_score
from llm.tip_generator         import generate_tip


def print_section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def print_row(label, value, suggestion=""):
    print(f"  {label:<30} {value}")
    if suggestion:
        print(f"  {'':>30} ⚠  {suggestion}")


def run_test(frame):
    print("\n" + "="*55)
    print("   AI PHOTOGRAPHER COACH — Full Pipeline Test")
    print("="*55)

    print_section("1 / 12 · Composition")
    comp = analyze_composition(frame)
    print_row("Subject detected",    comp["subject_detected"])
    print_row("On rule of thirds",   comp.get("on_rule_of_thirds"), comp.get("thirds_suggestion",""))
    print_row("Thirds score",        comp.get("thirds_score"))
    print_row("Symmetry score",      comp.get("symmetry_score"))
    print_row("Leading lines",       comp.get("leading_lines_detected"))

    print_section("2 / 12 · Lighting")
    light = analyze_lighting(frame)
    print_row("Exposure",            light["exposure_label"],  light.get("exposure_suggestion",""))
    print_row("Avg brightness",      light["avg_brightness"])
    print_row("Contrast",            light["contrast_label"],  light.get("contrast_suggestion",""))
    print_row("Harsh shadows",       light["harsh_shadows"],   light.get("shadow_suggestion",""))
    print_row("Color temperature",   light["color_temp_label"])

    print_section("3 / 12 · Horizon / Tilt")
    horiz = analyze_horizon(frame)
    print_row("Tilt angle",          f"{horiz['tilt_angle']}°", horiz.get("horizon_suggestion",""))
    print_row("Is level",            horiz["is_level"])

    print_section("4 / 12 · Subject & Depth")
    subj = analyze_subject(frame)
    print_row("Face detected",       subj["face_detected"])
    print_row("Subject size",        subj.get("subject_size_label",""), subj.get("subject_size_suggestion",""))
    print_row("Clutter",             subj["clutter_label"],             subj.get("clutter_suggestion",""))
    print_row("Depth layers",        subj["depth_layers"],              subj.get("depth_suggestion",""))

    print_section("5 / 12 · Sharpness / Focus")
    sharp = analyze_sharpness(frame)
    print_row("Sharpness label",     sharp["sharpness_label"],  sharp.get("sharpness_suggestion",""))
    print_row("Laplacian variance",  sharp["global_laplacian"])
    print_row("Focus miss",          sharp["focus_miss"])

    print_section("6 / 12 · Motion Blur")
    mb = analyze_motion_blur(frame)
    print_row("Blur label",          mb["motion_blur_label"],   mb.get("motion_blur_suggestion",""))
    print_row("Blur direction",      mb["blur_direction"])

    print_section("7 / 12 · Noise & Grain")
    noise = analyze_noise(frame)
    print_row("Noise label",         noise["noise_label"],      noise.get("noise_suggestion",""))
    print_row("Noise sigma",         noise["noise_sigma"])
    print_row("Banding detected",    noise["banding_detected"])

    print_section("8 / 12 · Background Blur (Bokeh)")
    bokeh = analyze_bokeh(frame)
    print_row("Bokeh label",         bokeh["bokeh_label"],      bokeh.get("bokeh_suggestion",""))
    print_row("Bokeh ratio",         bokeh["bokeh_ratio"])

    print_section("9 / 12 · Color Harmony & Palette")
    ch = analyze_color_harmony(frame)
    print_row("Harmony type",        ch["harmony_label"],       ch.get("harmony_suggestion",""))
    print_row("Color cast",          ch["color_cast"],          ch.get("color_cast_suggestion",""))
    for i, c in enumerate(ch.get("dominant_colors", [])[:3]):
        print_row(f"  Color {i+1}",  f"{c['hex']}  weight={c['weight']}")

    print_section("10 / 12 · Face Expression & Eye Contact")
    fe = analyze_face_expression(frame)
    print_row("Faces in frame",      fe["faces_in_frame"])
    print_row("Eye label",           fe["eye_label"],           fe.get("eye_suggestion",""))
    print_row("Smile detected",      fe["smile_detected"])
    print_row("Face is profile",     fe["face_is_profile"])

    print_section("11 / 12 · Foreground-Background Separation")
    fg = analyze_fg_bg_separation(frame)
    print_row("Separation label",    fg["color_separation_label"], fg.get("color_separation_suggestion",""))
    print_row("Color delta-E",       fg["color_delta_e"])
    print_row("BG distracting",      fg["bg_distracting"],         fg.get("bg_distraction_suggestion",""))

    print_section("12 / 12 · Aspect Ratio & Crop")
    ar = analyze_aspect_ratio(frame)
    print_row("Current ratio",       ar["aspect_ratio_label"])
    print_row("Headroom",            ar.get("headroom_label",""),  ar.get("headroom_suggestion",""))
    print_row("Subject cutoff",      ar["subject_cutoff"],         ar.get("cutoff_suggestion",""))
    print_row("Crop suggestion",     ar["crop_suggestion"])

    # Final Score
    score = compute_score(
        comp, light, horiz, subj,
        sharp, mb, noise, bokeh,
        ch, fe, fg, ar
    )

    print("\n" + "="*55)
    print(f"  TOTAL SCORE : {score['total']} / 10  ({score['grade']})")
    print("="*55)
    for cat, info in score["breakdown"].items():
        filled = int(info["score"])
        bar    = "█" * filled + "░" * (10 - filled)
        print(f"  {cat:<18} [{bar}]  {info['score']:>4}  {info['grade']}")
    print(f"\n  Weakest area : {score['weakest_category']}")

    if score["total"] < 7.0:
        print("\n  Generating AI tip (calling Ollama)...")
        all_data = {
            **comp, **light, **horiz, **subj,
            **sharp, **mb, **noise, **bokeh,
            **ch, **fe, **fg, **ar,
            "weakest_category": score["weakest_category"],
        }
        tip = generate_tip(all_data)
        print(f"\n  Tip: {tip}")
    else:
        print("\n  Score >= 7 — no LLM tip needed.")

    print("\n" + "="*55 + "\n")


def main():
    if len(sys.argv) > 1:
        path  = sys.argv[1]
        frame = cv2.imread(path)
        if frame is None:
            print(f"ERROR: Could not load image at {path}")
            sys.exit(1)
        print(f"Using image: {path}")
    else:
        print("No image path given — opening webcam. Press SPACE to capture, Q to quit.")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open webcam.")
            sys.exit(1)
        frame = None
        while True:
            ret, f = cap.read()
            if not ret:
                break
            cv2.imshow("Press SPACE to analyse this frame", f)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                frame = f
                break
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)
        cap.release()
        cv2.destroyAllWindows()
        if frame is None:
            print("No frame captured.")
            sys.exit(1)

    frame = cv2.resize(frame, (640, 480))
    run_test(frame)


if __name__ == "__main__":
    main()
