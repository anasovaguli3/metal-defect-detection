@echo off
cd /d "%~dp0"
title Quyma Detal - Streamlit

if not exist ".venv\Scripts\python.exe" (
    echo XATO: .venv topilmadi. Birinchi marta README dagi o'rnatishni bajaring.
    pause
    exit /b 1
)

echo Streamlit ishga tushmoqda...
echo Brauzer: http://127.0.0.1:8501
echo To'xtatish: Ctrl+C
echo.

.venv\Scripts\python.exe scripts\run_app.py
pause
