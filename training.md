  How to Know Training is Complete                                                                                                                                                      
                                                                                                                                                                                        
  Run this command anytime to check:                                                                                                                                                    
                                                                                                                                                                                        
  # Check if the weight file exists (this means training completed successfully):
  ls -lh models/weights/action_detection.pt                                                                                                                                             
                                                                                                                                                                                        
  # Watch live training progress (epoch by epoch):                                                                                                                                      
  tail -f /tmp/action_train.log                                                                                                                                                         
                                                                                                                                                                                        
  # Check GPU — if utilization is >0%, training is actively computing:                                                                                                                  
  watch -n 5 nvidia-smi
                                                                                                                                                                                        
  You'll know it's done when you see something like:                                                                                                                                    
  50/50   2.45G   1.234   2.567   1.123    ...
  Results saved to runs/action_detection/exp                                                                                                                                            
  Best weights saved → models/weights/action_detection.pt                                                                                                                               
                                                         
  Estimated time: On your 4GB GPU with batch=4 and 17,495 images: ~2-4 hours for 50 epochs. You can leave it running overnight.                                                         
                                                                                                                                                                                        
  ---                                                                                                                                                                                   
  Project Walkthrough                                                                                                                                                                   
                                                                                                                                                                                        
  ┌─────────────────────────────────────────────────────────────────┐
  │                    VOLLEY VISION PIPELINE                        │                                                                                                                  
  └─────────────────────────────────────────────────────────────────┘
                                                                                                                                                                                        
  1. You upload a volleyball match video (MP4)                                                                                                                                          
           │                                                                                                                                                                            
           ▼                                                                                                                                                                            
  2. Click "Analyze" → CV Pipeline runs in background       
     ├── YOLOv8 detects every player (bounding boxes)                                                                                                                                   
     ├── ByteTrack assigns stable IDs (#1, #2, #7 etc.) across frames                                                                                                                   
     ├── Ball detector finds the volleyball each frame                                                                                                                                  
     ├── Homography maps pixel → court coordinates (0–1 scale)                                                                                                                          
     ├── Rally detector segments the match into individual rallies                                                                                                                      
     └── Action detection (after training) detects: attack/serve/block/reception/dig/set                                                                                                
           │                                                                                                                                                                            
           ▼                                                                                                                                                                            
  3. Click "Speech" tab → Speech-to-Knowledge runs (NEW)                                                                                                                                
     ├── Whisper ASR transcribes the video's spoken commentary                                                                                                                          
     ├── NLP extracts events: "Great spike by player 7" → {type:attack, player:7, result:success}                                                                                       
     └── Event Fusion: matches speech events with CV events by timestamp (±5s)                                                                                                          
           │                                                                                                                                                                            
           ▼                                                                                                                                                                            
  4. Scoring Engine computes stats                                                                                                                                                      
     ├── Attack success rate per player                                                                                                                                                 
     ├── Serve efficiency
     ├── Block points                                                                                                                                                                   
     └── Rally winners (Team A vs Team B score)             
           │                                                                                                                                                                            
           ▼
  5. You see results in tabs:                                                                                                                                                           
     Overview  → match score, total rallies detected                                                                                                                                    
     Rallies   → list of each rally, seekable in video                                                                                                                                  
     Actions   → timeline of all detected actions with confidence %                                                                                                                     
     Analytics → per-player stats table                                                                                                                                                 
     Summary   → ball heatmap, zone chart, key moments                                                                                                                                  
     Speech    → NLP-extracted events with fusion status  ← NEW                                                                                                                         
                                                                                                                                                                                        
  ---                                                                                                                                                                                   
  Steps to Run the Project                                                                                                                                                              
                                                            
  Step 1 — Start PostgreSQL
                                                                                                                                                                                        
  # Check it's running:
  sudo systemctl status postgresql                                                                                                                                                      
                                                            
  # If not running:
  sudo systemctl start postgresql
                                                                                                                                                                                        
  # Create DB if first time:
  sudo -u postgres psql -c "CREATE DATABASE volleyball_analytics;"                                                                                                                      
  sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'password';"
                                                                                                                                                                                        
  Step 2 — Start Backend
                                                                                                                                                                                        
  cd /home/chandan/MAJOR-PROJECT-ORIGINAL--main/backend     
                                                                                                                                                                                        
  source .venv/bin/activate
  python run.py                                                                                                                                                                        
  # → Starts on http://localhost:8000                       
  # → API docs at http://localhost:8000/api/docs
                                                                                                                                                                                        
  Step 3 — Start Frontend (new terminal)
                                                                                                                                                                                        
  cd /home/chandan/MAJOR-PROJECT-ORIGINAL--main/frontend    
                                                                                                                                                                                        
  npm install      # first time only
  npm run dev                                                                                                                                                                           
  # → Opens at http://localhost:5173                        
                                                                                                                                                                                        
  Step 4 — Login
                                                                                                                                                                                        
  URL:      http://localhost:5173                           
  Email:    admin@volleyball.com
  Password: Admin@123456                                                                                                                                                                
   
  ---                                                                                                                                                                                   
  Which Video to Upload                                     
                                                                                                                                                                                        
  Best option — real volleyball match video:
  - A match filmed from a single fixed camera (sideline or end-line view works best)                                                                                                    
  - Needs to show the full court (or most of it)                                                                                                                                        
  - Duration: even 5–10 minutes is enough to demo
  - Format: MP4, MKV, AVI, MOV                                                                                                                                                          
                                                                                                                                                                                        
  Where to get one (free):                                                                                                                                                              
  1. YouTube — search "volleyball match full game" → download with yt-dlp:                                                                                                              
  pip install yt-dlp                                                                                                                                                                    
  yt-dlp -f "best[ext=mp4][height<=720]" "YOUTUBE_URL" -o match.mp4                                                                                                                     
  2. Your own college matches — any recorded match video works                                                                                                                          
                                                                                                                                                                                        
  What the system needs to work:                                                                                                                                                        
                                                                                                                                                                                        
  ┌─────────────────────────────┬────────────────────────────────────────────────┐                                                                                                      
  │         Requirement         │                      Why                       │
  ├─────────────────────────────┼────────────────────────────────────────────────┤                                                                                                      
  │ Court visible               │ Homography needs 4 court corners               │
  ├─────────────────────────────┼────────────────────────────────────────────────┤
  │ Players visible             │ YOLO person detection needs ≥ 20% frame height │                                                                                                      
  ├─────────────────────────────┼────────────────────────────────────────────────┤
  │ Ball visible (some frames)  │ Rally detector uses ball to segment rallies    │                                                                                                      
  ├─────────────────────────────┼────────────────────────────────────────────────┤                                                                                                      
  │ Commentary audio (optional) │ For Speech-to-Knowledge tab                    │
  └─────────────────────────────┴────────────────────────────────────────────────┘                                                                                                      
                                                            
  What to avoid:                                                                                                                                                                        
  - Broadcast TV video (overlays/graphics confuse the detector)
  - Zoomed-in action clips (no full court = homography fails)                                                                                                                           
  - Very dark or blurry video                                
                                                                                                                                                                                        
  Demo-Ready Workflow After Upload:                                                                                                                                                     
                                                                                                                                                                                        
  1. Upload video → create match                                                                                                                                                        
  2. Click Analyze → wait for 100% (shows live progress bar + WebSocket updates)                                                                                                        
  3. Go to Rallies tab → click any rally to seek video                                                                                                                                  
  4. Go to Actions tab → filter by spike/serve/block
  5. Go to Speech tab → click "Transcribe Match Audio" → wait ~2 min → see extracted events                                                                                             
  6. Go to Analytics tab → player efficiency stats                                                                                                                                      
  7. Go to Summary tab → ball heatmap showing where ball spent most time                                                                                                                
                                                                                                                                                                                        
  The training (action_detection.pt) must complete before step 4 shows real action data. Until then, if there's spoken commentary, the Speech tab alone will extract events and they'll 
  show up in Analytics too via the fusion engine.