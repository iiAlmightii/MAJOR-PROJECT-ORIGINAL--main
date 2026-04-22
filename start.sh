#!/usr/bin/env bash
set -e

echo "======================================"
echo "  VolleyVision - Startup Script"
echo "======================================"

# Check dependencies
command -v python3 &>/dev/null || { echo "ERROR: Python 3 not found"; exit 1; }
command -v node    &>/dev/null || { echo "ERROR: Node.js not found"; exit 1; }

# ─── Backend ──────────────────────────────
echo ""
echo "[1/3] Setting up Backend..."
cd backend

if [ ! -d ".venv" ]; then
  echo "  Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate

echo "  Installing Python dependencies..."
pip install -q -r requirements.txt

echo "  Starting FastAPI server on http://localhost:8000"
python run.py &
BACKEND_PID=$!
cd ..

# ─── Frontend ─────────────────────────────
echo ""
echo "[2/3] Setting up Frontend..."
cd frontend

if [ ! -d "node_modules" ]; then
  echo "  Installing Node.js dependencies..."
  npm install
fi

echo "  Starting React dev server on http://localhost:5173"
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "======================================"
echo "  Services running:"
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/api/docs"
echo "  Frontend: http://localhost:5173"
echo ""
echo "  Admin login:"
echo "  Email:    admin@volleyball.com"
echo "  Password: Admin@123456"
echo "======================================"
echo "Press Ctrl+C to stop all services"
echo ""

# Wait and cleanup
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" INT TERM
wait
