from __future__ import annotations
"""
Local Ollama coaching layer.

The computer-vision pipeline analyzes the frame locally with OpenCV/YOLO.
This module asks a local Ollama text model to turn those measurements into
one short, practical photography instruction.
"""

import json
import os

import requests

OLLAMA_URL = os.getenv("PHOTO_COACH_OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv(
    "PHOTO_COACH_OLLAMA_MODEL",
    "coney_/gpt-oss_claude-sonnet4.6:latest",
)
TIMEOUT_SECS = int(os.getenv("PHOTO_COACH_OLLAMA_TIMEOUT", "90"))


def _build_prompt(data: dict) -> str:
    lines = [
        "You are a professional photography coach helping while someone frames a shot.",
        "You cannot see the image directly; use the computer-vision measurements below.",
        "Give ONE short, practical instruction the photographer can do right now.",
        "Max 2 sentences. Do not mention scores or numbers.",
        "",
    ]

    if not data.get("subject_detected", True):
        lines.append("- No subject detected in frame.")
    else:
        _append_if_present(lines, "Composition", data.get("thirds_suggestion"))
        if data.get("is_symmetric"):
            lines.append("- Composition: frame is symmetrical.")

    if data.get("exposure_label") in ("underexposed", "overexposed"):
        _append_if_present(lines, "Lighting", data.get("exposure_suggestion"))

    if data.get("contrast_label") == "flat":
        _append_if_present(lines, "Contrast", data.get("contrast_suggestion"))

    if data.get("harsh_shadows"):
        _append_if_present(lines, "Shadows", data.get("shadow_suggestion"))

    if not data.get("is_level", True):
        _append_if_present(lines, "Tilt", data.get("horizon_suggestion"))

    if data.get("subject_size_label") in ("too_small", "too_close"):
        _append_if_present(lines, "Subject size", data.get("subject_size_suggestion"))

    if data.get("clutter_label") == "cluttered":
        _append_if_present(lines, "Background", data.get("clutter_suggestion"))

    _append_if_present(lines, "Depth", data.get("depth_suggestion"))
    _append_if_present(lines, "Sharpness", data.get("sharpness_suggestion"))
    _append_if_present(lines, "Motion blur", data.get("motion_blur_suggestion"))
    _append_if_present(lines, "Noise", data.get("noise_suggestion"))
    _append_if_present(lines, "Bokeh", data.get("bokeh_suggestion"))
    _append_if_present(lines, "Color harmony", data.get("harmony_suggestion"))
    _append_if_present(lines, "Color cast", data.get("color_cast_suggestion"))
    _append_if_present(lines, "Eye contact", data.get("eye_suggestion"))
    _append_if_present(lines, "Separation", data.get("color_separation_suggestion"))
    _append_if_present(lines, "Background distraction", data.get("bg_distraction_suggestion"))
    _append_if_present(lines, "Headroom", data.get("headroom_suggestion"))
    _append_if_present(lines, "Crop", data.get("crop_suggestion"))

    weakest = data.get("weakest_category", "")
    if weakest:
        lines.append(f"- Focus your tip on improving: {weakest}")

    compact_fields = {
        "total_score": data.get("total_score"),
        "grade": data.get("grade"),
        "weakest_category": data.get("weakest_category"),
        "sharpness_label": data.get("sharpness_label"),
        "motion_blur_label": data.get("motion_blur_label"),
        "noise_label": data.get("noise_label"),
        "bokeh_label": data.get("bokeh_label"),
        "harmony_label": data.get("harmony_label"),
        "color_cast": data.get("color_cast"),
        "eye_label": data.get("eye_label"),
        "faces_in_frame": data.get("faces_in_frame"),
        "color_separation_label": data.get("color_separation_label"),
        "aspect_ratio_label": data.get("aspect_ratio_label"),
        "headroom_label": data.get("headroom_label"),
    }
    lines.append("")
    lines.append("Other measurements:")
    lines.append(json.dumps({k: v for k, v in compact_fields.items() if v not in (None, "")}, ensure_ascii=True))
    lines.append("")
    lines.append("Tip:")
    return "\n".join(lines)


def _append_if_present(lines: list[str], label: str, value: str | None) -> None:
    if value:
        lines.append(f"- {label}: {value}")


def generate_tip(analysis_data: dict) -> str:
    prompt = _build_prompt(analysis_data)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.4,
                    "num_predict": 400,
                    "top_p": 0.9,
                },
            },
            timeout=TIMEOUT_SECS,
        )
        response.raise_for_status()
        data = response.json()
        tip = data.get("response", "").strip()
        if tip.lower().startswith("tip:"):
            tip = tip[4:].strip()
        if _looks_cut_off(tip):
            return _fallback_tip(analysis_data)
        return tip or _fallback_tip(analysis_data)

    except requests.exceptions.ConnectionError:
        return _fallback_tip(analysis_data)
    except requests.exceptions.Timeout:
        return _fallback_tip(analysis_data)
    except Exception as exc:
        print(f"[tip_generator] Ollama error: {exc}")
        return _fallback_tip(analysis_data)


def _fallback_tip(data: dict) -> str:
    checks = [
        ("exposure_suggestion", data.get("exposure_label") in ("underexposed", "overexposed")),
        ("horizon_suggestion", not data.get("is_level", True)),
        ("thirds_suggestion", not data.get("on_rule_of_thirds", True)),
        ("shadow_suggestion", data.get("harsh_shadows", False)),
        ("clutter_suggestion", data.get("clutter_label") == "cluttered"),
        ("subject_size_suggestion", data.get("subject_size_label") in ("too_small", "too_close")),
        ("depth_suggestion", bool(data.get("depth_suggestion"))),
        ("sharpness_suggestion", bool(data.get("sharpness_suggestion"))),
        ("motion_blur_suggestion", bool(data.get("motion_blur_suggestion"))),
        ("contrast_suggestion", data.get("contrast_label") == "flat"),
    ]

    for key, condition in checks:
        if condition and data.get(key):
            return data[key]

    return "The frame looks ready; hold steady and take the shot."


def _looks_cut_off(text: str) -> bool:
    if not text:
        return True
    tail = text.strip().lower().rstrip(".!?,")
    dangling_words = {
        "a",
        "an",
        "the",
        "to",
        "or",
        "and",
        "with",
        "for",
        "of",
        "in",
        "on",
        "at",
        "by",
    }
    return tail.split()[-1] in dangling_words
