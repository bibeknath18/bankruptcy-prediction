@echo off
echo Starting BankruptcyGuard...
echo.

echo Starting FastAPI Backend...
start "BankruptcyGuard API" cmd /k "cd /d C:\Users\HP\bankruptcy-prediction && uvicorn src.api.main:app --host 0.0.0.0 --port 8000"

timeout /t 5 /nobreak > nul

echo Starting React Frontend...
start "BankruptcyGuard UI" cmd /k "cd /d C:\Users\HP\bankruptcy-prediction\frontend && npm start"

echo.
echo BankruptcyGuard is starting...
echo API:      http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo.
pause
