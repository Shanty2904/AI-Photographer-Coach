@echo off
:: ============================================================
::  AI Photographer Coach — HTTPS Server Launcher
::  Serves the backend + frontend over HTTPS on port 8443.
::  Requires cert.pem + cert.key (run generate_cert.cmd first).
:: ============================================================
cd /d "%~dp0"

if not exist "%~dp0.ultralytics" mkdir "%~dp0.ultralytics"
set YOLO_CONFIG_DIR=%~dp0.ultralytics

if not exist "%~dp0cert.pem" (
  echo.
  echo  ERROR: cert.pem not found.
  echo  Run  generate_cert.cmd  first to create the TLS certificate.
  echo.
  pause
  exit /b 1
)

echo.
echo  Starting AI Photographer Coach (HTTPS)...
echo  Open on this PC :  https://localhost:8443
echo  Open on your phone: https://YOUR-PC-IP:8443
echo.

"%~dp0venv\Scripts\python.exe" -m uvicorn main:app ^
  --host 0.0.0.0 ^
  --port 8443 ^
  --ssl-keyfile  "%~dp0cert.key" ^
  --ssl-certfile "%~dp0cert.pem" ^
  >> "%~dp0server.out.log" 2>> "%~dp0server.err.log"
