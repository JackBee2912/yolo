# Baby Shrimp Detection Module 🦐

Module for detecting and counting baby shrimp using YOLO.

## Dataset
- Location: `modules/baby_shrimp/dataset/`
- Class: `babyshrimp`

## Usage

### Training with Real-time Monitoring 📊

**Terminal 1 - Start Training:**
```bash
# Activate environment
source .venv311/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Start training (YOLOv8 recommended for stability)
python modules/baby_shrimp/train.py \
  --model yolov8n.pt \
  --epochs 30 \
  --imgsz 640 \
  --batch 2 \
  --device cpu \
  --name baby_shrimp_training
```

**Terminal 2 - Monitor Progress:**
```bash
# Activate environment
source .venv311/bin/activate

# Start monitoring (wait ~30 seconds for training to create results.csv)
python modules/baby_shrimp/monitor.py \
  --path modules/baby_shrimp/runs/detect/baby_shrimp_training
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
python modules/baby_shrimp/train.py

# Custom parameters
python modules/baby_shrimp/train.py \
  --model yolov8n.pt \
  --epochs 30 \
  --batch 4 \
  --device cpu \
  --name my_shrimp_model
```

### Testing
```bash
# Test single image
python modules/baby_shrimp/test.py \
  --model modules/baby_shrimp/runs/detect/baby_shrimp_training/weights/best.pt \
  --image modules/baby_shrimp/dataset/test/images/sample.jpg

# Test multiple images
python modules/baby_shrimp/test.py \
  --model modules/baby_shrimp/runs/detect/baby_shrimp_training/weights/best.pt \
  --images-dir modules/baby_shrimp/dataset/test/images
```

## Results
- Dataset: `modules/baby_shrimp/dataset/`
- Old training results: `modules/baby_shrimp/results/` (archived)
- New training results: `modules/baby_shrimp/runs/detect/baby_shrimp_training/`
- Test results: `modules/baby_shrimp/runs/detect/baby_shrimp_predict_dots/`

**Note:** Training outputs are saved in `modules/baby_shrimp/runs/` directory.

