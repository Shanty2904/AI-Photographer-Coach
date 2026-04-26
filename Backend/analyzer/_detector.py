from __future__ import annotations
"""
_detector.py
------------
Shared subject detector used by all analyzer modules.
Imports YOLOv8 once and reuses across the pipeline.
"""
from analyzer.subject import detect_subjects, _primary_subject, YOLO_AVAILABLE

def get_primary_subject(frame):
    """Returns (detections, primary_subject_dict_or_None)"""
    dets = detect_subjects(frame)
    return dets, _primary_subject(dets)
