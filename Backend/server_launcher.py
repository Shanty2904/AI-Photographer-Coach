from __future__ import annotations
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
(ROOT / ".ultralytics").mkdir(exist_ok=True)
sys.stdout = (ROOT / "server_launcher.out.log").open("a", encoding="utf-8")
sys.stderr = (ROOT / "server_launcher.err.log").open("a", encoding="utf-8")

import uvicorn

try:
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
except Exception:
    (ROOT / "server_launcher.err.log").write_text(traceback.format_exc(), encoding="utf-8")
    raise
