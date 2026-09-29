@echo off
title AquaSense AI Server
cls
echo ========================================================
echo   AquaSense AI - Water Quality Prediction System
echo ========================================================
echo.

cd /d "%~dp0"

IF EXIST ".venv\Scripts\python.exe" (
    echo [INFO] Starting Flask Server using virtual environment...
    echo [INFO] Access the application at: http://127.0.0.1:5000
    echo.
    ".venv\Scripts\python.exe" app.py
) ELSE (
    echo [WARNING] .venv not found. Attempting to start with system python...
    python app.py
)

pause
