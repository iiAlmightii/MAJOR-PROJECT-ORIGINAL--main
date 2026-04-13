# VolleyVision

AI-based volleyball match analytics platform for coaches, players, and admins.

## Stack

- Frontend: React 18, Vite, Tailwind CSS, Zustand, React Query
- Backend: FastAPI, async SQLAlchemy
- Database: PostgreSQL 16
- CV / AI: YOLOv8, supervision, OpenCV
- Video: HTML5 streaming and FFmpeg-backed rally clips

## Repository Layout

- `backend/` - FastAPI application, API routers, services, and workers
- `frontend/` - React SPA for uploads, analytics, and match review
- `training/` - model training scripts and dataset helpers
- `models/weights/` - local model weights directory
- `Dataset/` - local training datasets (Roboflow exports + annotated videos)

## Local Setup

### Backend

1. Start PostgreSQL on port `5432`.
2. Create and activate a Python virtual environment in `backend/`.
3. Install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

4. Start the backend:

```bash
python run.py
```

The API runs at `http://localhost:8000`.

### Frontend

1. Install dependencies:

```bash
cd frontend
npm install
```

2. Start the dev server:

```bash
npm run dev
```

The app runs at `http://localhost:5173`.

## Version Control Notes

The root `.gitignore` excludes local secrets, virtual environments, build output, runtime upload folders, model weight files, and large dataset assets under `Dataset/`.

If you add new large generated artifacts, keep them out of git unless they are required source assets.
