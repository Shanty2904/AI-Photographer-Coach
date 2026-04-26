import os
from pathlib import Path

backend_dir = Path("e:/AI PHOTOGRAPHY/AI-Photographer-Coach-main/AI-Photographer-Coach-main/Backend")
for py_file in backend_dir.rglob("*.py"):
    if "venv" not in str(py_file):
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        idx = content.find("from __future__ import annotations\n")
        if idx != -1 and idx > 0:
            # Take everything from from __future__ onwards
            new_content = content[idx:]
            with open(py_file, "w", encoding="utf-8") as f:
                f.write(new_content)
