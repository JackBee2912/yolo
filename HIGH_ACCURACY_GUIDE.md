# 🎯 High Accuracy Training Guide

Hướng dẫn chi tiết để đạt độ chính xác cao nhất cho YOLO detection.

## 📊 So sánh các mức độ

| Level | Model | Epochs | Image Size | Training Time | Accuracy |
|-------|-------|--------|------------|---------------|----------|
| **Fast** | yolov8n | 50 | 640 | ~2 hours | ⭐⭐⭐ |
| **Balanced** | yolov8s | 100 | 640 | ~4 hours | ⭐⭐⭐⭐ |
| **High** | yolov8m | 150 | 1280 | ~8-10 hours | ⭐⭐⭐⭐⭐ |
| **Ultra** | yolov8l | 200 | 1280 | ~15-20 hours | ⭐⭐⭐⭐⭐⭐ |

## 🚀 Quick Start - High Accuracy

### Option 1: Medium Model (Recommended)
```bash
source .venv311/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Fingerlings
python modules/fingerlings/train_high_accuracy.py \
  --model yolov8m.pt \
  --epochs 150 \
  --imgsz 1280 \
  --batch 2 \
  --device cpu \
  --name fingerlings_high_accuracy

# Baby Shrimp
python modules/baby_shrimp/train_high_accuracy.py \
  --model yolov8m.pt \
  --epochs 100 \
  --imgsz 1280 \
  --batch 2 \
  --device cpu \
  --name baby_shrimp_high_accuracy
```

### Option 2: Large Model (Maximum Accuracy)
```bash
# Fingerlings
python modules/fingerlings/train_high_accuracy.py \
  --model yolov8l.pt \
  --epochs 200 \
  --imgsz 1280 \
  --batch 2 \
  --device cpu \
  --name fingerlings_ultra_accuracy

# Baby Shrimp
python modules/baby_shrimp/train_high_accuracy.py \
  --model yolov8l.pt \
  --epochs 150 \
  --imgsz 1280 \
  --batch 2 \
  --device cpu \
  --name baby_shrimp_ultra_accuracy
```

## 🎯 Key Factors for High Accuracy

### 1. Model Size (Most Important!)
- **yolov8n** (nano): 3.2M params - Fast but less accurate
- **yolov8s** (small): 11.2M params - Good balance
- **yolov8m** (medium): 25.9M params - High accuracy ⭐
- **yolov8l** (large): 43.7M params - Very high accuracy ⭐⭐
- **yolov8x** (xlarge): 68.2M params - Maximum accuracy (very slow)

### 2. Training Epochs
- More epochs = better convergence
- Recommended: 100-200 epochs for high accuracy
- Use early stopping (patience) to avoid overfitting

### 3. Image Size
- Larger images = better detection of small objects
- 640: Standard
- 1280: High accuracy (recommended) ⭐
- 1920: Ultra high (very slow)

### 4. Data Augmentation
High accuracy script includes:
- HSV augmentation (color variation)
- Rotation, translation, scale
- Mosaic & mixup
- Copy-paste augmentation

### 5. Learning Rate
- Lower LR = more stable training
- High accuracy uses: lr0=0.0005 (vs 0.001 standard)

## 📈 Expected Results

### Fingerlings (yolov8m, 150 epochs, 1280px):
- **Precision**: 0.85-0.92
- **Recall**: 0.80-0.88
- **mAP50**: 0.88-0.94
- **mAP50-95**: 0.65-0.75

### Baby Shrimp (yolov8m, 100 epochs, 1280px):
- **Precision**: 0.80-0.88
- **Recall**: 0.75-0.85
- **mAP50**: 0.82-0.90
- **mAP50-95**: 0.60-0.70

## 💡 Tips for Maximum Accuracy

### 1. Data Quality
- ✅ Check all labels are correct
- ✅ Add more training images if possible
- ✅ Balance dataset (similar number of images per class)
- ✅ Include diverse conditions (lighting, angles, backgrounds)

### 2. Training Strategy
- ✅ Start with medium model (yolov8m)
- ✅ Use larger image size (1280)
- ✅ Train longer (150+ epochs)
- ✅ Monitor validation metrics
- ✅ Use early stopping to prevent overfitting

### 3. Hardware Considerations
- **CPU**: Slow but works (8-20 hours)
- **GPU**: Much faster (1-3 hours)
- **Batch size**: Keep at 2-4 for stability

### 4. Post-Training
- ✅ Test on validation set
- ✅ Adjust confidence threshold
- ✅ Use ensemble if needed
- ✅ Fine-tune on specific cases

## 🔍 Monitoring Training

Use the monitor script to watch progress:
```bash
# Terminal 2
python modules/fingerlings/monitor.py \
  --path modules/fingerlings/runs/detect/fingerlings_high_accuracy
```

Watch for:
- **Losses decreasing** steadily
- **mAP50 increasing** over time
- **Validation metrics** improving
- **No overfitting** (train vs val gap)

## ⚠️ Common Issues

### Overfitting
- Symptoms: Train accuracy high, val accuracy low
- Solutions: More augmentation, lower LR, early stopping

### Underfitting
- Symptoms: Both train and val accuracy low
- Solutions: Larger model, more epochs, higher LR

### Slow Training
- Solutions: Smaller batch, smaller image size, smaller model

## 📊 Comparison Table

| Setting | Standard | High Accuracy | Difference |
|---------|----------|---------------|------------|
| Model | yolov8n | yolov8m | 8x more params |
| Epochs | 50 | 150 | 3x longer |
| Image Size | 640 | 1280 | 4x more pixels |
| LR | 0.001 | 0.0005 | 2x lower |
| Training Time | 2h | 10h | 5x longer |
| Expected mAP50 | 0.75-0.85 | 0.88-0.94 | +10-15% |

## 🎬 Next Steps

1. **Start training** with high accuracy script
2. **Monitor progress** with monitor.py
3. **Evaluate results** after training
4. **Fine-tune** if needed
5. **Test** on real data

Good luck! 🐟📊✨

