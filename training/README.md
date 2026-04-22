# Training Pipeline

## Prerequisites
```bash
pip install ultralytics supervision torch torchvision opencv-python numpy
```

---

## 1. Player Detection (YOLOv8)

**Quick start (no custom data):**
```bash
python player_detection/train_player.py --use-coco-pretrain --model yolov8n
```
This copies the COCO pretrained weights (person detection already works well) to `models/weights/player_detection.pt`.

**With custom volleyball dataset:**
```bash
# Step 1: Prepare dataset
python datasets/download_datasets.py --task player
# Add your images to training/datasets/player_detection/train/images/
# Add YOLO labels to training/datasets/player_detection/train/labels/

# Step 2: Train
python player_detection/train_player.py \
  --data training/datasets/player_detection/dataset.yaml \
  --model yolov8s \
  --epochs 100 \
  --imgsz 1280
```

**Evaluation:**
```bash
python player_detection/train_player.py --eval \
  --weights models/weights/player_detection.pt \
  --data training/datasets/player_detection/dataset.yaml
```

Expected metrics: mAP50 > 0.85, mAP50-95 > 0.55

---

## 2. Ball Detection (YOLOv8)

**Generate synthetic training data:**
```bash
python datasets/download_datasets.py --task ball
# Creates 200 synthetic volleyball images in training/datasets/ball_detection/train/
```

**Train:**
```bash
python ball_detection/train_ball.py \
  --data training/datasets/ball_detection/dataset.yaml \
  --model yolov8n \
  --epochs 200 \
  --imgsz 1280 \
  --batch 8
```

**Dataset sources (recommended for production):**
- TrackNet v2: https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2
- Roboflow volleyball: https://universe.roboflow.com/

Expected metrics: mAP50 > 0.70 (ball detection is hard due to small size)

---

## 3. Action Recognition (R3D-18 3D CNN)

**Download UCF-101 (contains volleyball spikes):**
```bash
python datasets/action_recognition/download_ucf101_volleyball.py
```

**Train:**
```bash
python action_recognition/train_action.py \
  --data training/datasets/action_recognition \
  --epochs 50 \
  --batch 8
```

**Classes detected:** serve, reception, set, attack/spike, block, dig, free_ball

Expected accuracy: > 80% on balanced dataset

---

## Model Weight Files

All trained models go to `models/weights/`:
```
models/weights/
├── player_detection.pt     # YOLOv8 player detector
├── ball_detection.pt       # YOLOv8 ball detector
└── action_recognition.pt   # R3D-18 action classifier
```

The CV pipeline auto-detects and uses these if present; falls back to COCO pretrain otherwise.

---

## GPU vs CPU

- **CUDA GPU strongly recommended** for training
- CPU inference works for demo/testing (slower — ~2-5 fps at 1080p)
- Set `--device cpu` for CPU, `--device 0` for first GPU
