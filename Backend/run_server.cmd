@echo off
cd /d "%~dp0"
if not exist "%~dp0.ultralytics" mkdir "%~dp0.ultralytics"
set YOLO_CONFIG_DIR=%~dp0.ultralytics
"%~dp0venv\Scripts\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000 > "%~dp0server.out.log" 2> "%~dp0server.err.log"
