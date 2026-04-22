# backend
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
grep -v "^lap==" requirements.txt > /tmp/req_no_lap.txt
pip install -r /tmp/req_no_lap.txt
cp -n .env.example .env
sudo -u postgres psql -c "ALTER USER postgres WITH PASSWORD 'password';"
python run.py

# frontend (new terminal)
cd frontend
npm install
npm run dev
