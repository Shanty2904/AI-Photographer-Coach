# AI Photographer Coach — Backend

Real-time photography composition and lighting feedback system.
Runs on your laptop, receives frames from the browser frontend or Android app over local WiFi.

---

## Folder Structure

```
backend/
├── main.py                  # FastAPI server (entry point)
├── test_pipeline.py         # Smoke test — run before mobile testing
├── requirements.txt
├── analyzer/
│   ├── composition.py       # Rule of thirds, golden ratio, symmetry
│   ├── lighting.py          # Exposure, contrast, shadows, color temp
│   ├── horizon.py           # Tilt / horizon alignment
│   ├── subject.py           # Subject size, clutter, depth
│   └── scorer.py            # Weighted score aggregator
└── llm/
    └── tip_generator.py     # Ollama local LLM integration
```

---

## Setup

### 1. Python Environment
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Install & Run Ollama (Local LLM)
```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download from https://ollama.com/download

# Pull the model (one-time, ~2GB download)
ollama pull llama3.2

# Start Ollama server (keep this running in a separate terminal)
ollama serve
```

> If your laptop has < 8GB RAM, use `ollama pull mistral` instead and
> update `MODEL_NAME = "mistral"` in `llm/tip_generator.py`.

### 3. Test the Pipeline (No Phone Needed)
```bash
# Test with webcam
python test_pipeline.py

# Test with a static image
python test_pipeline.py path/to/photo.jpg
```

---

## Running the Server

```bash
python main.py
```

Server starts at: `http://0.0.0.0:8000`

Open the browser frontend at: `http://localhost:8000`

The web frontend requests a camera stream with a 24fps minimum target and sends
fresh preview frames through `/ws/live`. The UI displays both camera capture FPS
and backend analysis FPS. The full analyzer is heavier than 24fps on most CPUs,
so the frontend applies backpressure and analyzes the newest available frame
instead of queueing stale frames.

### Find Your Laptop's Local IP (for the Android app)
```bash
# Windows
ipconfig
# Look for "IPv4 Address" under your WiFi adapter e.g. 192.168.1.105

# macOS
ifconfig | grep "inet " | grep -v 127
# e.g. inet 192.168.1.105

# Linux
hostname -I
```

Give this IP to your Android app (e.g. `http://192.168.1.105:8000`).

> ⚠️ Both laptop and phone must be on the SAME WiFi network.

### Windows Firewall
If the phone can't reach the server:
```
Control Panel → Windows Defender Firewall
→ Advanced Settings → Inbound Rules → New Rule
→ Port → TCP → 8000 → Allow
```

---

## API Reference

### GET /health
Verify server is running.
```json
{ "status": "ok", "message": "AI Photographer Coach v2 running.", "live_capture_target_fps": 24 }
```

### POST /analyze
Send a frame for analysis.

**Request body:**
```json
{
  "image": "<base64-encoded JPEG string>",
  "width": 640,
  "height": 480
}
```

**Response:**
```json
{
  "score": {
    "total": 7.4,
    "grade": "Good",
    "composition": 8.1,
    "lighting": 6.5,
    "horizon": 9.0,
    "subject": 7.2,
    "breakdown": { ... },
    "weakest_category": "lighting"
  },
  "composition": { "on_rule_of_thirds": true, "thirds_score": 8.5, ... },
  "lighting":    { "exposure_label": "well-exposed", "avg_brightness": 118.4, ... },
  "horizon":     { "tilt_angle": 1.2, "is_level": true, ... },
  "subject":     { "face_detected": true, "subject_size_label": "ideal", ... },
  "tip": "Try moving slightly to your left to use the natural window light more effectively.",
  "error": ""
}
```

> The `tip` field is only non-empty when `score.total < 7.0`.

### POST /analyze/live
Send preview frames while the camera is open. This is the endpoint to use before
the user clicks the shutter.

Recommended use:
- Start sending frames when the camera screen opens.
- Send one compressed 640x480 JPEG frame every 1-3 seconds.
- Show `live.message` directly on top of the camera preview.
- Enable the shutter or show a ready state when `live.ready_to_capture` is true.
- Keep using `/analyze` only for final, saved-photo review if you still want that.

**Request body:**
```json
{
  "image": "<base64-encoded preview frame>",
  "width": 640,
  "height": 480,
  "include_details": false,
  "include_tip": false,
  "client_capture_fps": 24
}
```

**Response:**
```json
{
  "live": {
    "ready_to_capture": false,
    "score": 6.4,
    "grade": "Needs work",
    "weakest_category": "lighting",
    "message": "Move toward brighter, softer light before taking the shot."
  },
  "tip": "",
  "capture": {
    "target_fps": 24,
    "client_capture_fps": 24.0,
    "server_analysis_fps": 1.2,
    "processing_ms": 820.4,
    "meets_24fps_capture_target": true
  },
  "error": ""
}
```

Set `include_details` to `true` only when you need the full analyzer payload during
debugging. Set `include_tip` to `true` sparingly because local LLM generation can
slow down live camera feedback.

### WebSocket /ws/live
For a smoother live camera loop, open:

```text
ws://<laptop-ip>:8000/ws/live
```

Send the same JSON payload as `/analyze/live` for each preview frame. The server
responds with the same `live` coaching object plus `capture` timing metadata.

---

## Notes for the Android App (React Native)

- Use `/analyze/live` while the camera preview is open, before the user clicks the shutter
- Send base64 preview frames from the camera preview loop every **1-3 seconds**
- Show `live.message` as the on-camera coaching overlay
- Use `live.ready_to_capture` to show when the frame is ready for the photo
- Keep `/analyze` for optional review after a photo is saved
- Send frames as base64-encoded JPEG strings (no data URI prefix needed, but it's handled if present)
- Recommended send interval: every **2–3 seconds** to avoid overloading the server
- Compress/resize frames to **640×480** before sending for faster transfer
- Always call `/health` on app start to verify connection before showing the camera screen
- The `histogram_bins` array (16 values, sum = 1.0) can be used to draw a mini histogram sparkline

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `mediapipe` install fails | Use Python 3.10 or 3.11. Not supported on 3.12+ yet |
| Ollama tip is slow | Normal on first run (model loading). Subsequent calls are faster |
| Phone can't reach server | Check firewall, confirm same WiFi, use `0.0.0.0` not `localhost` |
| `No module named 'analyzer'` | Run `python main.py` from inside the `backend/` folder |
| Low face detection confidence | Ensure adequate lighting and face is clearly visible |
