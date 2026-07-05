#!/usr/bin/env python3
"""Streamlit ilovasini to'g'ri fayl bilan ishga tushirish."""

import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app" / "app.py"

if not APP.exists():
    print(f"XATO: {APP} topilmadi")
    raise SystemExit(1)

if APP.suffix != ".py":
    print(f"XATO: Streamlit faqat .py fayl bilan ishlaydi, topildi: {APP.suffix}")
    raise SystemExit(1)

print(f"Ishga tushirilmoqda: {APP}")
subprocess.run(
    [sys.executable, "-m", "streamlit", "run", str(APP), "--server.address", "127.0.0.1", "--server.port", "8501"],
    check=True,
)
