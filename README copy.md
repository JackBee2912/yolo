# YOLO Detection Modules

Object detection and counting system using YOLO for aquaculture applications.

## 📁 Project Structure

```
yolo/
├── modules/                    # Detection modules
│   ├── baby_shrimp/           # Baby shrimp detection
│   │   ├── dataset/           # Dataset for baby shrimp
│   │   ├── runs/              # Training outputs (YOLO creates this)
│   │   │   └── detect/        # Detection/training results
│   │   ├── results/           # Archived training results
│   │   ├── train.py           # Training script
│   │   ├── test.py            # Testing script
│   │   ├── monitor.py         # Real-time training monitor
│   │   └── README.md          # Module documentation
│   │
│   └── fingerlings/           # Fish fingerlings detection
│       ├── dataset/           # Dataset for fingerlings
│       ├── runs/              # Training outputs (YOLO creates this)
│       │   └── detect/        # Detection/training results
│       ├── results/           # Archived training results
│       ├── train.py           # Training script
│       ├── test.py            # Testing script
│       ├── monitor.py         # Real-time training monitor
│       └── README.md          # Module documentation
│
├── .venv311/                  # Python virtual environment
├── requirements.txt           # Python dependencies
├── demo_monitor.sh            # Demo script for monitor
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
source .venv311/bin/activate

# Set environment variable (for Apple Silicon)
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### 2. Train a Model with Real-time Monitoring 📊

**Terminal 1 - Start Training:**

Baby Shrimp:
```bash
python modules/baby_shrimp/train.py \
  --model yolov8n.pt \
  --epochs 30 \
  --batch 2 \
  --device cpu \
  --name baby_shrimp_training
```

Fish Fingerlings:
```bash
python modules/fingerlings/train.py \
  --model yolov8n.pt \
  --epochs 50 \
  --batch 2 \
  --device cpu \
  --name fingerlings_training
```

**Terminal 2 - Monitor Progress:**

Baby Shrimp:
```bash
python modules/baby_shrimp/monitor.py \
  --path modules/baby_shrimp/runs/detect/baby_shrimp_training
```

Fish Fingerlings:
```bash
python modules/fingerlings/monitor.py \
  --path modules/fingerlings/runs/detect/fingerlings_training
```

The monitor displays live charts that update every 5 seconds:
- 📉 Training losses (box, class, DFL)
- 📊 Validation metrics (precision, recall, mAP50, mAP50-95)
- 📈 Learning rate curve
- 📋 Latest epoch summary

### 3. Test the Model

**Baby Shrimp:**
```bash
python modules/baby_shrimp/test.py \
  --model runs/train/baby_shrimp_training/weights/best.pt \
  --image modules/baby_shrimp/dataset/test/images/sample.jpg
```

**Fish Fingerlings:**
```bash
python modules/fingerlings/test.py \
  --model runs/train/fingerlings_training/weights/best.pt \
  --image modules/fingerlings/dataset/train/images/sample.jpg
```

## 📚 Documentation

- [Modules Overview](modules/README.md)
- [Baby Shrimp Module](modules/baby_shrimp/README.md)
- [Fish Fingerlings Module](modules/fingerlings/README.md)

## 🎯 Features

- ✅ Modular architecture - each detection task is self-contained
- ✅ Separate datasets and results for each module
- ✅ Red dot visualization on detected objects
- ✅ Batch testing support
- ✅ Support for MPS (Apple Silicon), CUDA, and CPU
- ✅ Optimized for small dense object detection

## 🛠️ Requirements

- Python 3.11
- PyTorch with MPS/CUDA support
- Ultralytics YOLO
- OpenCV

See `requirements.txt` for full dependencies.

## 📦 Modules

### 🦐 Baby Shrimp
Detect and count baby shrimp in dense populations.
- Dataset: 128 images (114 train, 7 val, 7 test)
- Class: `babyshrimp`

### 🐟 Fish Fingerlings
Detect and count fish fingerlings.
- Dataset: 1,496 images (1,444 train, 52 val)
- Class: `fingerling`

## 📄 License

MIT License
