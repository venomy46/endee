@echo off
echo ============================================================
echo   TravelLens — Setup Script
echo ============================================================

echo.
echo [1/3] Creating Python virtual environment...
python -m venv .venv

echo.
echo [2/3] Installing dependencies...
.venv\Scripts\pip install -r requirements.txt

echo.
echo [3/3] Done! Now run:
echo.
echo   docker compose up -d
echo   .venv\Scripts\python.exe seed_data.py
echo   .venv\Scripts\uvicorn.exe app:app --port 8000
echo.
echo Then open: http://localhost:8000
echo ============================================================
pause
