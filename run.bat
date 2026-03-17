@echo off
echo ============================================================
echo   Starting TravelLens
echo ============================================================

echo.
echo [1/2] Starting Endee Vector Database...
docker compose up -d

echo.
echo [2/2] Starting TravelLens Server...
echo The app will open at: http://localhost:8000
echo (Press Ctrl+C to stop the server)
echo.

.venv\Scripts\uvicorn.exe app:app --port 8000
