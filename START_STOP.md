# VolleyVision — Start & Stop Commands

## Start the Project

Open **two terminals** and run one command in each.

### Terminal 1 — Backend
```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main/backend"
.venv/bin/python run.py
```
Runs at: http://localhost:8001

### Terminal 2 — Frontend
```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main/frontend"
npm run dev
```
Runs at: http://localhost:5173

---

## Start in Background (one terminal)

```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main/backend" && nohup .venv/bin/python run.py > /tmp/backend.log 2>&1 &
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main/frontend" && nohup npm run dev > /tmp/frontend.log 2>&1 &
```

Check logs anytime:
```bash
tail -f /tmp/backend.log
tail -f /tmp/frontend.log
```

---

## Stop the Project

```bash
pkill -f "python run.py"
pkill -f "vite"
```

Or both at once:
```bash
pkill -f "python run.py"; pkill -f "vite"
```

---

## URLs

| Service  | URL                    |
|----------|------------------------|
| Frontend | http://localhost:5173  |
| Backend  | http://localhost:8001  |
| API Docs | http://localhost:8001/docs |

## Default Login

| Field    | Value                  |
|----------|------------------------|
| Email    | admin@volleyball.com   |
| Password | Admin@123456           |
