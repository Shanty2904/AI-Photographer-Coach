"""
AI Photographer Coach — Python server launcher
Supports both HTTP and HTTPS.

Usage:
  python server_launcher.py           → HTTP  on port 8000
  python server_launcher.py --https   → HTTPS on port 8443 (requires cert.pem + cert.key)
"""
import os
import sys
import traceback
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
(ROOT / ".ultralytics").mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--https", action="store_true", help="Enable HTTPS (requires cert.pem + cert.key)")
args = parser.parse_args()

if args.https:
    log_out = ROOT / "server.out.log"
    log_err = ROOT / "server.err.log"
else:
    log_out = ROOT / "server.out.log"
    log_err = ROOT / "server.err.log"

sys.stdout = log_out.open("a", encoding="utf-8")
sys.stderr = log_err.open("a", encoding="utf-8")

import uvicorn

try:
    if args.https:
        cert = ROOT / "cert.pem"
        key  = ROOT / "cert.key"
        if not cert.exists() or not key.exists():
            raise FileNotFoundError(
                "cert.pem / cert.key not found. Run generate_cert.cmd first."
            )
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8443,
            ssl_certfile=str(cert),
            ssl_keyfile=str(key),
            log_level="info",
        )
    else:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
except Exception:
    log_err.write_text(traceback.format_exc(), encoding="utf-8")
    raise
