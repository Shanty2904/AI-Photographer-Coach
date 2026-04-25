"""
benchmark.py
------------
Runs all 12 analyzer modules against the synthetic test set and
reports per-module accuracy.

Usage:
    cd Backend
    python -m benchmark.benchmark

Prerequisite:
    python -m benchmark.generate_test_images
"""

import json
import os
import sys
import time

import cv2
import numpy as np

# Allow running from Backend/ root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analyzer.sharpness        import analyze_sharpness
from analyzer.motion_blur      import analyze_motion_blur
from analyzer.noise            import analyze_noise
from analyzer.lighting         import analyze_lighting
from analyzer.horizon          import analyze_horizon
from analyzer.bokeh            import analyze_bokeh
from analyzer.color_harmony    import analyze_color_harmony
from analyzer.composition      import analyze_composition
from analyzer.fg_bg_separation import analyze_fg_bg_separation
from analyzer.aspect_ratio     import analyze_aspect_ratio
from analyzer.face_expression  import analyze_face_expression
from analyzer.subject          import analyze_subject

IMG_DIR    = os.path.join(os.path.dirname(__file__), "test_images")
LABEL_FILE = os.path.join(os.path.dirname(__file__), "ground_truth.json")

# -------------------------------------------------------------
# DISPATCH: maps module name → (analyzer_fn, check_fn)
# check_fn(result, record) → (passed: bool, detail: str)
# -------------------------------------------------------------

def _check_sharpness(r, rec):
    label  = r["sharpness_label"]
    score  = r["sharpness_score"]
    exp_l  = rec.get("expected_label")
    exp_sm = rec.get("expected_score_min")
    exp_sx = rec.get("expected_score_max")
    notes  = []

    label_ok = (exp_l is None) or (label == exp_l)
    score_ok = True
    if exp_sm is not None and score < exp_sm:
        score_ok = False
        notes.append(f"score {score:.1f} < min {exp_sm}")
    if exp_sx is not None and score > exp_sx:
        score_ok = False
        notes.append(f"score {score:.1f} > max {exp_sx}")
    if not label_ok:
        notes.append(f"label '{label}' != expected '{exp_l}'")

    return label_ok and score_ok, f"label={label} score={score:.1f}" + (f" | FAIL: {'; '.join(notes)}" if notes else "")


def _check_motion_blur(r, rec):
    label = r["motion_blur_label"]
    score = r["motion_blur_score"]
    exp_l = rec.get("expected_label")
    exp_sm = rec.get("expected_score_min")
    exp_sx = rec.get("expected_score_max")
    notes = []

    label_ok = (exp_l is None) or (label == exp_l)
    score_ok = True
    if exp_sm is not None and score < exp_sm:
        score_ok = False
        notes.append(f"score {score:.1f} < min {exp_sm}")
    if exp_sx is not None and score > exp_sx:
        score_ok = False
        notes.append(f"score {score:.1f} > max {exp_sx}")
    if not label_ok:
        notes.append(f"label '{label}' != '{exp_l}'")

    return label_ok and score_ok, f"label={label} score={score:.1f}" + (f" | FAIL: {'; '.join(notes)}" if notes else "")


def _check_noise(r, rec):
    label  = r["noise_label"]
    score  = r["noise_score"]
    exp_l  = rec.get("expected_label")
    exp_sm = rec.get("expected_score_min")
    exp_sx = rec.get("expected_score_max")
    notes  = []

    label_ok = (exp_l is None) or (label == exp_l)
    score_ok = True
    if exp_sm is not None and score < exp_sm:
        score_ok = False
        notes.append(f"score {score:.1f} < min {exp_sm}")
    if exp_sx is not None and score > exp_sx:
        score_ok = False
        notes.append(f"score {score:.1f} > max {exp_sx}")
    if not label_ok:
        notes.append(f"label '{label}' != '{exp_l}'")

    return label_ok and score_ok, f"label={label} score={score:.1f}" + (f" | FAIL: {'; '.join(notes)}" if notes else "")


def _check_lighting(r, rec):
    notes = []
    ok = True

    if "expected_exposure_label" in rec:
        if r["exposure_label"] != rec["expected_exposure_label"]:
            ok = False
            notes.append(f"exposure '{r['exposure_label']}' != '{rec['expected_exposure_label']}'")

    if "expected_exposure_score_min" in rec and r["exposure_score"] < rec["expected_exposure_score_min"]:
        ok = False
        notes.append(f"exp_score {r['exposure_score']:.1f} < min {rec['expected_exposure_score_min']}")

    if "expected_exposure_score_max" in rec and r["exposure_score"] > rec["expected_exposure_score_max"]:
        ok = False
        notes.append(f"exp_score {r['exposure_score']:.1f} > max {rec['expected_exposure_score_max']}")

    if "expected_color_temp" in rec and r["color_temp_label"] != rec["expected_color_temp"]:
        ok = False
        notes.append(f"temp '{r['color_temp_label']}' != '{rec['expected_color_temp']}'")

    if "expected_harsh_shadows" in rec and r["harsh_shadows"] != rec["expected_harsh_shadows"]:
        ok = False
        notes.append(f"harsh_shadows {r['harsh_shadows']} != {rec['expected_harsh_shadows']}")

    if "expected_contrast_label" in rec and r["contrast_label"] != rec["expected_contrast_label"]:
        ok = False
        notes.append(f"contrast '{r['contrast_label']}' != '{rec['expected_contrast_label']}'")

    detail = f"exp={r['exposure_label']} temp={r['color_temp_label']} contrast={r['contrast_label']}"
    return ok, detail + (f" | FAIL: {'; '.join(notes)}" if notes else "")


def _check_horizon(r, rec):
    notes = []
    ok = True

    if "expected_is_level" in rec and r["is_level"] != rec["expected_is_level"]:
        ok = False
        notes.append(f"is_level {r['is_level']} != {rec['expected_is_level']}")

    if "expected_score_min" in rec and r["horizon_score"] < rec["expected_score_min"]:
        ok = False
        notes.append(f"score {r['horizon_score']:.1f} < min {rec['expected_score_min']}")

    if "expected_score_max" in rec and r["horizon_score"] > rec["expected_score_max"]:
        ok = False
        notes.append(f"score {r['horizon_score']:.1f} > max {rec['expected_score_max']}")

    return ok, f"is_level={r['is_level']} angle={r.get('tilt_angle',0):.1f}° score={r['horizon_score']:.1f}" + (f" | FAIL: {'; '.join(notes)}" if notes else "")


def _check_bokeh(r, rec):
    label = r["bokeh_label"]
    score = r["bokeh_score"]
    notes = []
    ok = True

    if "expected_label" in rec and label != rec["expected_label"]:
        ok = False
        notes.append(f"label '{label}' != '{rec['expected_label']}'")

    if "expected_score_min" in rec and score < rec["expected_score_min"]:
        ok = False
        notes.append(f"score {score:.1f} < min {rec['expected_score_min']}")

    if "expected_score_max" in rec and score > rec["expected_score_max"]:
        ok = False
        notes.append(f"score {score:.1f} > max {rec['expected_score_max']}")

    return ok, f"label={label} score={score:.1f}" + (f" | FAIL: {'; '.join(notes)}" if notes else "")


def _check_color_harmony(r, rec):
    notes = []
    ok = True

    if "expected_label" in rec and r["harmony_label"] != rec["expected_label"]:
        ok = False
        notes.append(f"harmony '{r['harmony_label']}' != '{rec['expected_label']}'")

    if "expected_color_cast" in rec and r.get("color_cast") != rec["expected_color_cast"]:
        ok = False
        notes.append(f"cast '{r.get('color_cast')}' != '{rec['expected_color_cast']}'")

    return ok, f"harmony={r['harmony_label']} cast={r.get('color_cast','?')}" + (f" | FAIL: {'; '.join(notes)}" if notes else "")


def _check_composition(r, rec):
    notes = []
    ok = True

    if "expected_is_symmetric" in rec and r.get("is_symmetric") != rec["expected_is_symmetric"]:
        ok = False
        notes.append(f"is_symmetric {r.get('is_symmetric')} != {rec['expected_is_symmetric']}")

    if "expected_symmetry_score_min" in rec and r.get("symmetry_score", 0) < rec["expected_symmetry_score_min"]:
        ok = False
        notes.append(f"sym_score {r.get('symmetry_score'):.1f} < min {rec['expected_symmetry_score_min']}")

    if "expected_has_leading_lines" in rec and r.get("has_leading_lines") != rec["expected_has_leading_lines"]:
        ok = False
        notes.append(f"leading_lines {r.get('has_leading_lines')} != {rec['expected_has_leading_lines']}")

    return ok, f"sym={r.get('is_symmetric')} lines={r.get('leading_lines_detected',0)} sym_score={r.get('symmetry_score',0):.1f}" + (f" | FAIL: {'; '.join(notes)}" if notes else "")


def _check_fg_bg(r, rec):
    notes = []
    ok = True

    if "expected_label" in rec and r.get("color_separation_label") != rec["expected_label"]:
        ok = False
        notes.append(f"sep '{r.get('color_separation_label')}' != '{rec['expected_label']}'")

    if "expected_score_min" in rec and r.get("fg_bg_score", 0) < rec["expected_score_min"]:
        ok = False
        notes.append(f"score {r.get('fg_bg_score'):.1f} < min {rec['expected_score_min']}")

    if "expected_score_max" in rec and r.get("fg_bg_score", 0) > rec["expected_score_max"]:
        ok = False
        notes.append(f"score {r.get('fg_bg_score'):.1f} > max {rec['expected_score_max']}")

    if "expected_bg_distracting" in rec and r.get("bg_distracting") != rec["expected_bg_distracting"]:
        ok = False
        notes.append(f"bg_distracting {r.get('bg_distracting')} != {rec['expected_bg_distracting']}")

    return ok, f"sep={r.get('color_separation_label')} dE={r.get('color_delta_e',0):.1f}" + (f" | FAIL: {'; '.join(notes)}" if notes else "")


def _check_aspect_ratio(r, rec):
    notes = []
    ok = True

    if "expected_ratio_label" in rec and r.get("aspect_ratio_label") != rec["expected_ratio_label"]:
        ok = False
        notes.append(f"ratio '{r.get('aspect_ratio_label')}' != '{rec['expected_ratio_label']}'")

    return ok, f"ratio={r.get('aspect_ratio_label')}" + (f" | FAIL: {'; '.join(notes)}" if notes else "")


def _check_face_expression(r, rec):
    notes = []
    ok = True

    if "expected_faces" in rec and r.get("faces_in_frame") != rec["expected_faces"]:
        ok = False
        notes.append(f"faces {r.get('faces_in_frame')} != {rec['expected_faces']}")

    if "expected_eye_label" in rec and r.get("eye_label") != rec["expected_eye_label"]:
        ok = False
        notes.append(f"eye_label '{r.get('eye_label')}' != '{rec['expected_eye_label']}'")

    return ok, f"faces={r.get('faces_in_frame')} eyes={r.get('eye_label')}" + (f" | FAIL: {'; '.join(notes)}" if notes else "")


def _check_subject(r, rec):
    notes = []
    ok = True

    if "expected_subject_detected" in rec and r.get("subject_detected") != rec["expected_subject_detected"]:
        ok = False
        notes.append(f"subject_detected {r.get('subject_detected')} != {rec['expected_subject_detected']}")

    return ok, f"detected={r.get('subject_detected')} label={r.get('subject_label','--')}" + (f" | FAIL: {'; '.join(notes)}" if notes else "")


MODULE_MAP = {
    "sharpness":      (analyze_sharpness,        _check_sharpness),
    "motion_blur":    (analyze_motion_blur,       _check_motion_blur),
    "noise":          (analyze_noise,             _check_noise),
    "lighting":       (analyze_lighting,          _check_lighting),
    "horizon":        (analyze_horizon,           _check_horizon),
    "bokeh":          (analyze_bokeh,             _check_bokeh),
    "color_harmony":  (analyze_color_harmony,     _check_color_harmony),
    "composition":    (analyze_composition,       _check_composition),
    "fg_bg":          (analyze_fg_bg_separation,  _check_fg_bg),
    "aspect_ratio":   (analyze_aspect_ratio,      _check_aspect_ratio),
    "face_expression":(analyze_face_expression,   _check_face_expression),
    "subject":        (analyze_subject,           _check_subject),
}

# -------------------------------------------------------------
# MAIN RUNNER
# -------------------------------------------------------------

def _load_image(fname):
    path = os.path.join(IMG_DIR, fname)
    img  = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def run_benchmark():
    if not os.path.exists(LABEL_FILE):
        print("ERROR: ground_truth.json not found.")
        print("Run:  python -m benchmark.generate_test_images  first.\n")
        sys.exit(1)

    with open(LABEL_FILE) as f:
        records = json.load(f)

    # Group by module for the summary table
    from collections import defaultdict
    by_module: dict[str, list] = defaultdict(list)
    for rec in records:
        by_module[rec["module"]].append(rec)

    print("\n" + "=" * 72)
    print("   AI PHOTOGRAPHER COACH -- Synthetic Accuracy Benchmark")
    print("=" * 72)

    total_pass = 0
    total_fail = 0
    module_results = {}

    for module_name in [
        "sharpness", "motion_blur", "noise", "lighting",
        "horizon", "bokeh", "color_harmony", "composition",
        "fg_bg", "aspect_ratio", "face_expression", "subject",
    ]:
        recs = by_module.get(module_name, [])
        if not recs:
            continue

        fn, check = MODULE_MAP[module_name]
        passed = 0
        failed = 0
        rows   = []
        t0 = time.perf_counter()

        for rec in recs:
            try:
                img    = _load_image(rec["file"])
                result = fn(img)
                ok, detail = check(result, rec)
            except Exception as e:
                ok     = False
                detail = f"EXCEPTION: {e}"

            status = "PASS" if ok else "FAIL"
            rows.append((rec["file"], status, detail))
            if ok:
                passed += 1
            else:
                failed += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000
        accuracy   = passed / len(recs) * 100
        total_pass += passed
        total_fail += failed
        module_results[module_name] = {"pass": passed, "fail": failed, "acc": accuracy}

        pct_bar = "#" * int(accuracy / 10) + "-" * (10 - int(accuracy / 10))
        print(f"\n  -- {module_name.upper():20s}  [{pct_bar}]  {accuracy:5.1f}%  ({passed}/{len(recs)})  {elapsed_ms:.0f}ms")
        for fname, status, detail in rows:
            flag = "    " if status == "PASS" else "  >> "
            print(f"  {flag}{status}  {fname:<35}  {detail}")

    # -- Summary table --
    grand_total = total_pass + total_fail
    overall_acc = total_pass / grand_total * 100 if grand_total else 0

    print("\n" + "=" * 72)
    print(f"  OVERALL ACCURACY : {overall_acc:.1f}%  ({total_pass}/{grand_total} tests passed)")
    print("=" * 72)
    print(f"\n  {'Module':<22} {'Accuracy':>10}  {'Pass':>5}  {'Fail':>5}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*5}  {'-'*5}")
    for mod, res in module_results.items():
        flag = "  " if res["fail"] == 0 else "! "
        print(f"  {flag}{mod:<20} {res['acc']:>9.1f}%  {res['pass']:>5}  {res['fail']:>5}")
    print()


if __name__ == "__main__":
    run_benchmark()
