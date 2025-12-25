# Fish Fingerlings Detection Module 🐟

Module for detecting and counting fish fingerlings using YOLO.

## Dataset
- Location: `modules/fingerlings/dataset/`
- Class: `fingerling`
- Images: 1,444 train + 52 validation

## Usage

### Training with Real-time Monitoring 📊

**Terminal 1 - Start Training:**
```bash
# Activate environment
source .venv311/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Start training (YOLOv8 recommended for stability)
python modules/fingerlings/train.py \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 2 \
  --device cpu \
  --name fingerlings_training
```

**Terminal 2 - Monitor Progress:**
```bash
# Activate environment
source .venv311/bin/activate

# Start monitoring (wait ~30 seconds for training to create results.csv)
python modules/fingerlings/monitor.py \
  --path modules/fingerlings/runs/detect/fingerlings_training
```

**Monitor displays:**
- 📉 Training losses (box, class, DFL)
- 📊 Validation metrics (precision, recall, mAP50, mAP50-95)
- 📈 Learning rate curve
- 📋 Latest epoch summary

Charts auto-refresh every 5 seconds!

### Training (Basic)
```bash
# Basic training (from project root)
python modules/fingerlings/train.py

# Custom parameters
python modules/fingerlings/train.py \
  --model yolov8n.pt \
  --epochs 100 \
  --batch 4 \
  --device cpu \
  --name my_fingerlings_model
```

### Testing
```bash
# Test single image
python modules/fingerlings/test.py \
  --model modules/fingerlings/runs/detect/fingerlings_training/weights/best.pt \
  --image modules/fingerlings/dataset/train/images/sample.jpg

# Test multiple images
python modules/fingerlings/test.py \
  --model modules/fingerlings/runs/detect/fingerlings_training/weights/best.pt \
  --images-dir modules/fingerlings/dataset/valid/images
```

## Results
- Dataset: `modules/fingerlings/dataset/`
- Training results: `modules/fingerlings/runs/detect/fingerlings_training/`
- Test results: `modules/fingerlings/runs/detect/fingerlings_predict_dots/`

**Note:** Training outputs are saved in `modules/fingerlings/runs/` directory.

