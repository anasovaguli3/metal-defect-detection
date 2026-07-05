@echo off

cd /d "%~dp0"

echo ========================================

echo  AI assignment - venv qayta o'rnatish

echo ========================================

echo.



set PY311=C:\Users\anaso\AppData\Local\Programs\Python\Python311\python.exe

set PY312=C:\Users\anaso\AppData\Local\Programs\Python\Python312\python.exe

if exist "%PY311%" (

    set PY=%PY311%

    echo Python 3.11 ishlatiladi.

) else if exist "%PY312%" (

    set PY=%PY312%

    echo Python 3.11 topilmadi — 3.12 ishlatiladi.

) else (

    echo XATO: Python topilmadi.

    pause

    exit /b 1

)



if exist ".venv" (

    echo Eski .venv o'chirilmoqda...

    rmdir /s /q ".venv"

)



echo Yangi venv yaratilmoqda...

"%PY%" -m venv .venv

call .venv\Scripts\activate.bat



python -m pip install --upgrade pip

pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt



python scripts\verify_setup.py

echo TAYYOR!

pause

