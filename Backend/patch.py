import os
from pathlib import Path

def patch_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    if "from __future__ import annotations" not in content and "def " in content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("from __future__ import annotations\n" + content)

backend_dir = Path("e:/AI PHOTOGRAPHY/AI-Photographer-Coach-main/AI-Photographer-Coach-main/Backend")
for py_file in backend_dir.rglob("*.py"):
    if "venv" not in str(py_file):
        patch_file(py_file)
