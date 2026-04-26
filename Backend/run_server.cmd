@echo off
cd /d "%~dp0"
if not exist "%~dp0.ultralytics" mkdir "%~dp0.ultralytics"
set YOLO_CONFIG_DIR=%~dp0.ultralytics

echo Checking for existing process on port 8000...
"%~dp0venv\Scripts\python.exe" -c "import socket; s=socket.socket(); r=s.connect_ex(('127.0.0.1',8000)); s.close(); exit(0 if r==0 else 1)" >nul 2>&1
if %errorlevel%==0 (
    echo Port 8000 is in use. Killing it...
    for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R "0\.0\.0\.0:8000.*LISTENING"') do taskkill /PID %%P /F >nul 2>&1
    timeout /t 1 /nobreak >nul
)

echo.
echo Starting AI Photographer Coach backend...
echo Server will be available at http://localhost:8000
echo Press CTRL+C to stop.
echo.

"%~dp0venv\Scripts\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000

echo.
echo Server stopped.
pause
