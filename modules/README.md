# Detection Modules

Organized modules for different detection tasks. Each module is self-contained with its own training and testing code.

## Structure

```
modules/
├── baby_shrimp/          # Baby shrimp detection module
│   ├── __init__.py
│   ├── train.py          # Training script
│   ├── test.py           # Testing script
│   └── README.md         # Module documentation
│
└── fingerlings/          # Fish fingerlings detection module
    ├── __init__.py
    ├── train.py          # Training script
    ├── test.py           # Testing script
    └── README.md         # Module documentation
```

## Modules

### 🦐 Baby Shrimp
- **Purpose**: Detect and count baby shrimp in dense populations
- **Dataset**: `modules/baby_shrimp/dataset/`
- **Documentation**: [baby_shrimp/README.md](baby_shrimp/README.md)

### 🐟 Fish Fingerlings
- **Purpose**: Detect and count fish fingerlings
- **Dataset**: `modules/fingerlings/dataset/`
- **Documentation**: [fingerlings/README.md](fingerlings/README.md)

## Quick Start

### Baby Shrimp
```bash
# Train (from project root)
python modules/baby_shrimp/train.py --epochs 30

# Test
python modules/baby_shrimp/test.py \
  --model runs/train/baby_shrimp_training/weights/best.pt \
  --image modules/baby_shrimp/dataset/test/images/13_jpg.rf.72952ceedcd77f161140cc5432ac295b.jpg
```

### Fish Fingerlings
```bash
# Train (from project root)
python modules/fingerlings/train.py --epochs 100

# Test
python modules/fingerlings/test.py \
  --model runs/train/fingerlings_training/weights/best.pt \
  --image modules/fingerlings/dataset/train/images/20-1_jpg.rf.5cdff0eb49b81ff9357f1d12b1d6be6e.jpg
```

## Environment Setup

```bash
# Activate virtual environment
source .venv311/bin/activate

# Set environment variable for MPS (Apple Silicon)
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Notes

- Each module uses its own dataset and doesn't interfere with others
- Training results are saved in `runs/train/{module_name}/`
- Test results are saved in `runs/detect/{module_name}_predict_dots/`
- All modules support MPS (Apple Silicon), CUDA, and CPU

