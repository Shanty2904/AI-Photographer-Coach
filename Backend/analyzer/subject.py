from __future__ import annotations
"""
subject.py — YOLOv8n subject detection
Detects 80 COCO classes. Falls back to Haar if ultralytics not installed.
"""

from pathlib import Path

import cv2
import numpy as np

_yolo_model = None
_yolo_error = None
YOLO_AVAILABLE = True

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

PRIMARY_CLASSES = {
    "person","cat","dog","bird","horse","cow","elephant","bear","zebra",
    "giraffe","bottle","cup","book","laptop","cell phone","keyboard",
    "mouse","backpack","handbag","vase","teddy bear","potted plant",
    "chair","bicycle","motorcycle","clock","tv","remote","umbrella"
}


def _detect_yolo(frame):
    model = _get_yolo_model()
    if model is None:
        return _detect_haar(frame)

    h, w    = frame.shape[:2]
    results = model(frame, verbose=False, conf=0.30)[0]
    dets    = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label  = results.names[cls_id]
        conf   = float(box.conf[0])
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        bw,bh = x2-x1, y2-y1
        dets.append({
            "label": label, "confidence": round(conf,3),
            "cx_norm": round((x1+bw/2)/w,3), "cy_norm": round((y1+bh/2)/h,3),
            "area_norm": round((bw*bh)/(w*h),4),
            "x1_px":int(x1),"y1_px":int(y1),"x2_px":int(x2),"y2_px":int(y2),
        })
    return dets


def _get_yolo_model():
    global _yolo_model, _yolo_error, YOLO_AVAILABLE

    if _yolo_model is not None:
        return _yolo_model
    if _yolo_error is not None:
        return None

    try:
        from ultralytics import YOLO

        model_path = Path(__file__).resolve().parents[1] / "yolov8n.pt"
        _yolo_model = YOLO(str(model_path))
        YOLO_AVAILABLE = True
        return _yolo_model
    except Exception as exc:
        _yolo_error = exc
        YOLO_AVAILABLE = False
        return None


def is_yolo_available():
    return _get_yolo_model() is not None


def _detect_haar(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w  = gray.shape
    faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    dets  = []
    for (fx,fy,fw,fh) in (faces if len(faces)>0 else []):
        dets.append({
            "label":"person","confidence":0.6,
            "cx_norm":round((fx+fw/2)/w,3),"cy_norm":round((fy+fh/2)/h,3),
            "area_norm":round((fw*fh)/(w*h),4),
            "x1_px":fx,"y1_px":fy,"x2_px":fx+fw,"y2_px":fy+fh,
        })
    return dets


def detect_subjects(frame):
    return _detect_yolo(frame) if YOLO_AVAILABLE else _detect_haar(frame)


def _primary_subject(detections):
    primary = [d for d in detections if d["label"] in PRIMARY_CLASSES]
    pool    = primary if primary else detections
    return max(pool, key=lambda d: d["area_norm"]) if pool else None


def _subject_size_score(subject, frame_shape):
    if subject is None:
        return {
            "face_detected":False,"subject_detected":False,"subject_label":"none",
            "face_area_ratio":0.0,"subject_size_label":"no_subject",
            "subject_size_score":3.0,
            "subject_size_suggestion":"No subject detected. Point camera at your subject.",
        }
    area  = subject["area_norm"]
    label = subject["label"]
    if area < 0.03:
        sl,sc,sg = "too_small",4.0,f"Subject ({label}) is too small. Move closer or zoom in."
    elif area > 0.60:
        sl,sc,sg = "too_close",6.0,"Subject fills too much of the frame. Step back slightly."
    elif 0.08 <= area <= 0.40:
        sl,sc,sg = "ideal",10.0,""
    else:
        sl,sc,sg = "acceptable",7.5,""
    return {
        "face_detected":  label=="person",
        "subject_detected":True,
        "subject_label":  label,
        "subject_confidence": subject["confidence"],
        "face_area_ratio": round(area,3),
        "subject_size_label": sl,
        "subject_size_score": sc,
        "subject_size_suggestion": sg,
    }


def _clutter_score(frame, detections):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(5,5),0)
    edges   = cv2.Canny(blurred,50,150)
    mask    = np.ones_like(edges, dtype=bool)
    for d in detections:
        mask[d["y1_px"]:d["y2_px"], d["x1_px"]:d["x2_px"]] = False
    bg      = edges[mask]
    density = float(np.sum(bg>0))/(bg.size+1e-5)
    if density < 0.05:   l,s,sg = "clean",9.0,""
    elif density < 0.15: l,s,sg = "moderate",7.0,""
    else:
        l  = "cluttered"
        s  = max(0.0, 10.0-density*40)
        sg = "Background looks cluttered. Try a simpler background or use portrait/bokeh mode."
    return {"clutter_label":l,"clutter_density":round(density,4),"clutter_score":round(s,2),"clutter_suggestion":sg}


def _depth_layers(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h     = gray.shape[0]; t = h//3
    def sh(r): return float(cv2.Laplacian(r,cv2.CV_64F).var())
    sc    = [sh(gray[:t,:]),sh(gray[t:2*t,:]),sh(gray[2*t:,:])]
    act   = sum(1 for s in sc if s>50)
    return {
        "depth_layers":act,
        "sharpness_top":round(sc[0],2),"sharpness_mid":round(sc[1],2),"sharpness_bottom":round(sc[2],2),
        "depth_score":round(min(10.0,act*3.5),2),
        "depth_suggestion":"" if act>=2 else "Frame lacks depth. Try adding a foreground element.",
    }


def _negative_space(detections, frame_shape):
    h,w = frame_shape[:2]
    fa  = h*w
    if detections:
        sa = sum((d["x2_px"]-d["x1_px"])*(d["y2_px"]-d["y1_px"]) for d in detections)
        sr = min(1.0, sa/fa)
    else:
        sr = 0.1
    ns = 1.0 - sr
    if 0.40<=ns<=0.75: nsc,nsn = 9.0,"Good use of negative space."
    elif ns<0.25:       nsc,nsn = 5.0,"Subject fills too much of the frame. Allow more breathing room."
    else:               nsc,nsn = 7.0,""
    return {"negative_space_ratio":round(ns,3),"subject_fill_ratio":round(sr,3),"negative_space_score":nsc,"negative_space_note":nsn}


def analyze_subject(frame):
    detections = detect_subjects(frame)
    subject    = _primary_subject(detections)
    result = {
        "all_detections":  [{"label":d["label"],"confidence":d["confidence"]} for d in detections],
        "detection_count": len(detections),
        "detector_used":   "yolov8n" if is_yolo_available() else "haar_cascade",
    }
    result.update(_subject_size_score(subject, frame.shape))
    result.update(_clutter_score(frame, detections))
    result.update(_depth_layers(frame))
    result.update(_negative_space(detections, frame.shape))
    return result
