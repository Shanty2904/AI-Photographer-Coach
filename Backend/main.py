from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import base64
from collections import deque
import os
import time
import numpy as np
import cv2
import traceback

from analyzer.composition    import analyze_composition
from analyzer.lighting       import analyze_lighting
from analyzer.horizon        import analyze_horizon
from analyzer.subject        import analyze_subject
from analyzer.sharpness      import analyze_sharpness
from analyzer.motion_blur    import analyze_motion_blur
from analyzer.noise          import analyze_noise
from analyzer.bokeh          import analyze_bokeh
from analyzer.color_harmony  import analyze_color_harmony
from analyzer.face_expression import analyze_face_expression
from analyzer.fg_bg_separation import analyze_fg_bg_separation
from analyzer.aspect_ratio   import analyze_aspect_ratio
from analyzer.scorer         import compute_score
from llm.tip_generator       import generate_tip

app = FastAPI(title="AI Photographer Coach", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class FrameRequest(BaseModel):
    image: str
    width: int = 640
    height: int = 480

class LiveFrameRequest(FrameRequest):
    include_details: bool = False
    include_tip: bool = False
    client_capture_fps: float | None = None

def decode_image(b64_string: str) -> np.ndarray:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    np_arr    = np.frombuffer(img_bytes, np.uint8)
    frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image.")
    return frame

def resize_frame(frame: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    width = max(160, min(int(width or 640), 1280))
    height = max(120, min(int(height or 480), 720))
    return cv2.resize(frame, (width, height))

def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value

def analyze_photo_frame(frame: np.ndarray) -> dict:
    composition     = analyze_composition(frame)
    lighting        = analyze_lighting(frame)
    horizon         = analyze_horizon(frame)
    subject         = analyze_subject(frame)
    sharpness       = analyze_sharpness(frame)
    motion_blur     = analyze_motion_blur(frame)
    noise           = analyze_noise(frame)
    bokeh           = analyze_bokeh(frame)
    color_harmony   = analyze_color_harmony(frame)
    face_expression = analyze_face_expression(frame)
    fg_bg           = analyze_fg_bg_separation(frame)
    aspect_ratio    = analyze_aspect_ratio(frame)

    score = compute_score(
        composition, lighting, horizon, subject,
        sharpness, motion_blur, noise, bokeh,
        color_harmony, face_expression, fg_bg, aspect_ratio
    )

    return to_jsonable({
        "score":          score,
        "composition":    composition,
        "lighting":       lighting,
        "horizon":        horizon,
        "subject":        subject,
        "sharpness":      sharpness,
        "motion_blur":    motion_blur,
        "noise":          noise,
        "bokeh":          bokeh,
        "color_harmony":  color_harmony,
        "face_expression": face_expression,
        "fg_bg":          fg_bg,
        "aspect_ratio":   aspect_ratio,
    })

def live_coaching_message(result: dict) -> dict:
    score = result["score"]
    weakest = score["weakest_category"]
    modules = {
        "composition": result["composition"],
        "lighting": result["lighting"],
        "horizon": result["horizon"],
        "subject": result["subject"],
        "sharpness": result["sharpness"],
        "motion_blur": result["motion_blur"],
        "noise": result["noise"],
        "bokeh": result["bokeh"],
        "color_harmony": result["color_harmony"],
        "face_expression": result["face_expression"],
        "fg_bg": result["fg_bg"],
        "aspect_ratio": result["aspect_ratio"],
    }
    suggestion_keys = {
        "composition": ["thirds_suggestion"],
        "lighting": ["exposure_suggestion", "contrast_suggestion", "shadow_suggestion"],
        "horizon": ["horizon_suggestion"],
        "subject": ["subject_size_suggestion", "clutter_suggestion", "depth_suggestion", "negative_space_note"],
        "sharpness": ["sharpness_suggestion"],
        "motion_blur": ["motion_blur_suggestion"],
        "noise": ["noise_suggestion"],
        "bokeh": ["bokeh_suggestion"],
        "color_harmony": ["harmony_suggestion", "color_cast_suggestion"],
        "face_expression": ["eye_suggestion", "face_angle_note"],
        "fg_bg": ["color_separation_suggestion", "bg_distraction_suggestion"],
        "aspect_ratio": ["headroom_suggestion", "cutoff_suggestion", "crop_suggestion"],
    }

    message = ""
    for key in suggestion_keys.get(weakest, []):
        value = modules[weakest].get(key, "")
        if value:
            message = value
            break

    if not message:
        if score["total"] >= 7.0:
            message = "Looks good. Hold steady and take the shot."
        else:
            message = f"Adjust {weakest.replace('_', ' ')} before taking the shot."

    sharpness_label = result["sharpness"].get("sharpness_label", "")
    subject_detected = result["composition"].get("subject_detected", False)
    ready_to_capture = score["total"] >= 7.0 and subject_detected and sharpness_label not in {"blurry", "very_blurry"}

    return {
        "ready_to_capture": bool(ready_to_capture),
        "score": round(float(score["total"]), 2),
        "grade": score["grade"],
        "weakest_category": weakest,
        "message": message,
    }

def build_live_response(
    result: dict,
    started_at: float,
    client_capture_fps: float | None = None,
    server_analysis_fps: float | None = None,
) -> dict:
    processing_ms = round((time.perf_counter() - started_at) * 1000, 1)
    return {
        "live": live_coaching_message(result),
        "tip": "",
        "error": "",
        "capture": {
            "target_fps": 24,
            "client_capture_fps": round(float(client_capture_fps), 1) if client_capture_fps else None,
            "server_analysis_fps": round(float(server_analysis_fps), 1) if server_analysis_fps else None,
            "processing_ms": processing_ms,
            "meets_24fps_capture_target": bool(client_capture_fps and client_capture_fps >= 24),
        },
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "AI Photographer Coach v2 running.",
        "live_capture_target_fps": 24,
    }

@app.post("/analyze")
async def analyze_frame(req: FrameRequest):
    try:
        frame = decode_image(req.image)
        frame = resize_frame(frame, req.width, req.height)
        result = analyze_photo_frame(frame)
        score = result["score"]

        # --- LLM tip only when score < 7 ---
        tip = ""
        if score["total"] < 7.0:
            summary = {
                **result["composition"], **result["lighting"], **result["horizon"], **result["subject"],
                **result["sharpness"], **result["motion_blur"], **result["noise"], **result["bokeh"],
                **result["color_harmony"], **result["face_expression"], **result["fg_bg"], **result["aspect_ratio"],
                "total_score": score["total"],
                "weakest_category": score["weakest_category"],
            }
            tip = generate_tip(summary)

        return to_jsonable({**result, "tip": tip, "error": ""})

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/live")
async def analyze_live_frame(req: LiveFrameRequest):
    """
    Low-latency preview analysis for the camera screen.
    Call this repeatedly while the user is framing the shot, not after capture.
    """
    try:
        started_at = time.perf_counter()
        frame = decode_image(req.image)
        frame = resize_frame(frame, req.width, req.height)
        result = analyze_photo_frame(frame)
        response = build_live_response(result, started_at, req.client_capture_fps)

        if req.include_details:
            response["analysis"] = result

        if req.include_tip and result["score"]["total"] < 7.0:
            summary = {
                **result["composition"], **result["lighting"], **result["horizon"], **result["subject"],
                **result["sharpness"], **result["motion_blur"], **result["noise"], **result["bokeh"],
                **result["color_harmony"], **result["face_expression"], **result["fg_bg"], **result["aspect_ratio"],
                "total_score": result["score"]["total"],
                "weakest_category": result["score"]["weakest_category"],
            }
            response["tip"] = generate_tip(summary)

        return to_jsonable(response)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/live")
async def live_camera_socket(websocket: WebSocket):
    """
    WebSocket version of /analyze/live for continuous camera previews.
    Send JSON frames: {"image": "...base64 jpeg...", "width": 640, "height": 480}
    """
    await websocket.accept()
    frame_times = deque(maxlen=48)
    try:
        while True:
            started_at = time.perf_counter()
            payload = await websocket.receive_json()
            req = LiveFrameRequest(**payload)
            frame = decode_image(req.image)
            frame = resize_frame(frame, req.width, req.height)
            result = analyze_photo_frame(frame)
            now = time.perf_counter()
            frame_times.append(now)
            server_fps = None
            if len(frame_times) > 1:
                elapsed = frame_times[-1] - frame_times[0]
                if elapsed > 0:
                    server_fps = (len(frame_times) - 1) / elapsed

            response = build_live_response(
                result,
                started_at,
                req.client_capture_fps,
                server_fps,
            )
            if req.include_details:
                response["analysis"] = result
            await websocket.send_json(to_jsonable(response))
    except WebSocketDisconnect:
        return
    except Exception as e:
        traceback.print_exc()
        await websocket.send_json({"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Frontend"))
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
