from __future__ import annotations
"""
scorer.py — updated with all 12 modules
"""

WEIGHTS = {
    "composition":      0.15,
    "lighting":         0.12,
    "horizon":          0.08,
    "subject":          0.08,
    "sharpness":        0.12,
    "motion_blur":      0.10,
    "noise":            0.08,
    "bokeh":            0.07,
    "color_harmony":    0.07,
    "face_expression":  0.05,
    "fg_bg":            0.05,
    "aspect_ratio":     0.03,
}

def _grade(s):
    if s >= 9.0: return "Excellent"
    if s >= 7.5: return "Good"
    if s >= 6.0: return "Fair"
    if s >= 4.0: return "Needs Work"
    return "Poor"

def compute_score(composition, lighting, horizon, subject,
                  sharpness=None, motion_blur=None, noise=None,
                  bokeh=None, color_harmony=None, face_expression=None,
                  fg_bg=None, aspect_ratio=None):

    sharpness       = sharpness       or {}
    motion_blur     = motion_blur     or {}
    noise           = noise           or {}
    bokeh           = bokeh           or {}
    color_harmony   = color_harmony   or {}
    face_expression = face_expression or {}
    fg_bg           = fg_bg           or {}
    aspect_ratio    = aspect_ratio    or {}

    def comp_score():
        t = composition.get("thirds_score", 5.0)
        g = composition.get("golden_score", 5.0)
        s = composition.get("symmetry_score", 5.0)
        l = composition.get("lines_score", 5.0)
        if not composition.get("subject_detected", False):
            return s * 0.5 + l * 0.5
        return t*0.45 + g*0.25 + s*0.15 + l*0.15

    def light_score():
        return (lighting.get("exposure_score", 5.0) * 0.50 +
                lighting.get("contrast_score", 5.0) * 0.30 +
                lighting.get("shadow_score",   5.0) * 0.20)

    def subj_score():
        return (subject.get("subject_size_score",   5.0) * 0.35 +
                subject.get("clutter_score",         5.0) * 0.25 +
                subject.get("depth_score",           5.0) * 0.25 +
                subject.get("negative_space_score",  5.0) * 0.15)

    def ch_score():
        base = color_harmony.get("harmony_score", 7.0)
        if color_harmony.get("color_cast", "none") != "none":
            base = max(0.0, base - 1.5)
        return base

    scores = {
        "composition":     round(comp_score(), 2),
        "lighting":        round(light_score(), 2),
        "horizon":         round(horizon.get("horizon_score", 7.0), 2),
        "subject":         round(subj_score(), 2),
        "sharpness":       round(sharpness.get("sharpness_score", 5.0), 2),
        "motion_blur":     round(motion_blur.get("motion_blur_score", 7.0), 2),
        "noise":           round(noise.get("noise_score", 7.0), 2),
        "bokeh":           round(bokeh.get("bokeh_score", 5.0), 2),
        "color_harmony":   round(ch_score(), 2),
        "face_expression": round(face_expression.get("face_expression_score", 5.0), 2),
        "fg_bg":           round(fg_bg.get("fg_bg_score", 5.0), 2),
        "aspect_ratio":    round(aspect_ratio.get("aspect_ratio_score", 8.0), 2),
    }

    total     = round(sum(scores[k] * WEIGHTS[k] for k in WEIGHTS), 2)
    breakdown = {k: {"score": scores[k], "grade": _grade(scores[k]), "weight": WEIGHTS[k]} for k in WEIGHTS}
    weakest   = min(breakdown, key=lambda k: breakdown[k]["score"])

    return {"total": total, "grade": _grade(total), "weakest_category": weakest, "breakdown": breakdown, **scores}
