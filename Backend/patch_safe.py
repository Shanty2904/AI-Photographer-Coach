import os
from pathlib import Path

for py_file in Path("e:/AI PHOTOGRAPHY/AI-Photographer-Coach-main/AI-Photographer-Coach-main/Backend/analyzer").glob("*.py"):
    with open(py_file, "r", encoding="utf-8") as f:
        content = f.read()
    if "from __future__ import annotations" not in content:
        with open(py_file, "w", encoding="utf-8") as f:
            f.write("from __future__ import annotations\n" + content)
