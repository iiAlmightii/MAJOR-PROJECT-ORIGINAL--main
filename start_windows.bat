@echo off
echo ======================================
echo   VolleyVision - Windows Startup
echo ======================================

:: Start Backend
echo.
echo [1/2] Starting Backend...
cd backend
if not exist ".venv" (
  echo   Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate
pip install -q -r requirements.txt
start "VolleyVision Backend" cmd /k "call .venv\Scripts\activate && python run.py"
cd ..

:: Start Frontend
echo.
echo [2/2] Starting Frontend...
cd frontend
if not exist "node_modules" (
  echo   Installing npm dependencies...
  npm install
)
start "VolleyVision Frontend" cmd /k "npm run dev"
cd ..

echo.
echo ======================================
echo   Backend:  http://localhost:8000
echo   API Docs: http://localhost:8000/api/docs
echo   Frontend: http://localhost:5173
echo.
echo   Admin: admin@volleyball.com / Admin@123456
echo ======================================
pause
