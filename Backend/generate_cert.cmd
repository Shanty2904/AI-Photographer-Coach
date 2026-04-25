@echo off
:: ============================================================
::  AI Photographer Coach — Self-Signed TLS Certificate Generator
::  Run this ONCE before the first HTTPS launch.
::  Requires: OpenSSL (bundled with Git for Windows)
:: ============================================================
cd /d "%~dp0"

echo.
echo  Generating self-signed TLS certificate for HTTPS...
echo.

:: Locate openssl.exe — tries Git's bundled copy first, then PATH
set OPENSSL=openssl
where openssl >nul 2>&1
if errorlevel 1 (
  if exist "C:\Program Files\Git\usr\bin\openssl.exe" (
    set OPENSSL=C:\Program Files\Git\usr\bin\openssl.exe
  ) else (
    echo  ERROR: openssl.exe not found.
    echo  Install Git for Windows (https://git-scm.com) or OpenSSL for Windows.
    pause
    exit /b 1
  )
)

:: Generate private key + self-signed certificate valid for 825 days
"%OPENSSL%" req -x509 -newkey rsa:2048 -nodes ^
  -keyout "%~dp0cert.key" ^
  -out    "%~dp0cert.pem" ^
  -days   825 ^
  -subj   "/CN=AI-Photographer-Coach" ^
  -addext "subjectAltName=IP:127.0.0.1,IP:0.0.0.0"

if errorlevel 1 (
  echo.
  echo  ERROR: Certificate generation failed.
  pause
  exit /b 1
)

echo.
echo  Done! Created:
echo    cert.pem  (certificate — install this on your phone to trust HTTPS)
echo    cert.key  (private key  — keep this private)
echo.
echo  NEXT: Run  run_server_https.cmd  to start the HTTPS server.
echo.
pause
