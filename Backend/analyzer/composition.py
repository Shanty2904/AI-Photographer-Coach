from __future__ import annotations
"""
composition.py — Rule of thirds, golden ratio, symmetry, leading lines
Uses shared YOLOv8 detector for subject center.
"""
import cv2
import numpy as np

PHI = 1.618033988749895


def _get_subject_center(frame):
    """Get subject center using YOLO (or Haar fallback)."""
    try:
        from analyzer._detector import get_primary_subject
        _, subject = get_primary_subject(frame)
        if subject:
            return subject["cx_norm"], subject["cy_norm"]
    except Exception:
        pass
    # Haar fallback
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fc   = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    h, w = gray.shape
    faces = fc.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    if len(faces) > 0:
        x,y,fw,fh = max(faces, key=lambda f: f[2]*f[3])
        return (x+fw/2)/w, (y+fh/2)/h
    return None


def _rule_of_thirds_score(cx, cy):
    thirds_x = [1/3, 2/3]; thirds_y = [1/3, 2/3]; tol = 0.10
    best_dist = float("inf"); best_point = None
    for tx in thirds_x:
        for ty in thirds_y:
            d = ((cx-tx)**2+(cy-ty)**2)**0.5
            if d < best_dist: best_dist=d; best_point=(tx,ty)
    on_thirds = best_dist <= tol
    score = max(0.0, 10.0-(best_dist/tol)*5.0)
    suggestion = ""
    if not on_thirds:
        tx,ty = best_point
        mh = "left" if cx>tx else "right"
        mv = "up"   if cy>ty else "down"
        suggestion = f"Move subject slightly {mh} and {mv} to hit a rule-of-thirds point."
    return {
        "on_rule_of_thirds": on_thirds,
        "subject_position": {"x":round(cx,3),"y":round(cy,3)},
        "nearest_thirds_point": {"x":round(best_point[0],3),"y":round(best_point[1],3)},
        "thirds_score": round(score,2),
        "thirds_suggestion": suggestion,
    }


def _golden_ratio_score(cx, cy):
    gl = [1/PHI, 1-1/PHI]; tol = 0.08
    dx = abs(cx-min(gl, key=lambda g: abs(cx-g)))
    dy = abs(cy-min(gl, key=lambda g: abs(cy-g)))
    aligned = dx<=tol and dy<=tol
    score = max(0.0, 10.0-((dx+dy)/(2*tol))*5.0)
    return {"on_golden_ratio":aligned,"golden_score":round(score,2)}


def _symmetry_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h,w  = gray.shape
    left = gray[:,:w//2]; right = cv2.flip(gray[:,w//2:],1)
    mw   = min(left.shape[1],right.shape[1])
    diff = np.mean(np.abs(left[:,:mw].astype(np.float32)-right[:,:mw].astype(np.float32)))
    score = max(0.0, 10.0-(diff/25.5))
    return {"is_symmetric":diff<25.0,"symmetry_diff":round(float(diff),2),"symmetry_score":round(score,2)}


def _leading_lines_score(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize=3)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=80,minLineLength=60,maxLineGap=10)
    n     = len(lines) if lines is not None else 0
    score = 7.0 if 3<=n<=15 else (5.0 if n>15 else 4.0)
    return {"leading_lines_detected":n,"has_leading_lines":3<=n<=15,"lines_score":score}


def analyze_composition(frame):
    result = {
        "subject_detected":False,"on_rule_of_thirds":False,
        "thirds_score":5.0,"thirds_suggestion":"No subject detected in frame.",
        "on_golden_ratio":False,"golden_score":5.0,
        "subject_position":{"x":0.5,"y":0.5},
        "nearest_thirds_point":{"x":0.33,"y":0.33},
    }
    center = _get_subject_center(frame)
    if center:
        cx,cy = center
        result["subject_detected"] = True
        result.update(_rule_of_thirds_score(cx,cy))
        result.update(_golden_ratio_score(cx,cy))
    result.update(_symmetry_score(frame))
    result.update(_leading_lines_score(frame))
    return result
