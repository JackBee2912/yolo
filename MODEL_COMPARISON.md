# 📊 YOLOv8 Model Comparison Guide

So sánh chi tiết các model YOLOv8: Nano, Small, Medium, Large

## 🎯 Quick Comparison Table

| Model | Size | Parameters | FLOPs | Speed (CPU) | Speed (GPU) | mAP50 | mAP50-95 | Best For |
|-------|------|------------|-------|-------------|-------------|-------|----------|----------|
| **YOLOv8n** | 6.2 MB | 3.2M | 8.7G | 80 ms | 1.2 ms | 52.3% | 37.3% | Real-time, Edge devices |
| **YOLOv8s** | 21.5 MB | 11.2M | 28.6G | 128 ms | 1.9 ms | 61.8% | 44.9% | Balanced speed/accuracy |
| **YOLOv8m** | 49.7 MB | 25.9M | 78.9G | 234 ms | 3.2 ms | 67.2% | 50.2% | High accuracy ⭐ |
| **YOLOv8l** | 83.7 MB | 43.7M | 165.2G | 375 ms | 4.3 ms | 69.8% | 52.9% | Maximum accuracy |
| **YOLOv8x** | 130.5 MB | 68.2M | 257.8G | 479 ms | 6.9 ms | 71.1% | 53.9% | Research, Best quality |

*Benchmark on COCO dataset, CPU: Intel i7, GPU: RTX 3080*

## 📈 Detailed Comparison

### YOLOv8n (Nano) - Fast & Light
```
Size: 6.2 MB
Parameters: 3.2M
FLOPs: 8.7G
```

**Pros:**
- ✅ Fastest inference speed
- ✅ Smallest model size
- ✅ Perfect for edge devices (Raspberry Pi, mobile)
- ✅ Low memory usage
- ✅ Quick training (~2 hours for 50 epochs)

**Cons:**
- ❌ Lower accuracy
- ❌ May miss small objects
- ❌ Less robust to variations

**Use Cases:**
- Real-time applications
- Mobile/embedded devices
- Quick prototyping
- When speed > accuracy

**Training Time (50 epochs, 640px, CPU):**
- Fingerlings: ~2 hours
- Baby Shrimp: ~1.5 hours

---

### YOLOv8s (Small) - Balanced
```
Size: 21.5 MB
Parameters: 11.2M
FLOPs: 28.6G
```

**Pros:**
- ✅ Good balance of speed and accuracy
- ✅ Still fast enough for real-time
- ✅ Better than nano for small objects
- ✅ Reasonable training time

**Cons:**
- ❌ Not the fastest
- ❌ Not the most accurate
- ❌ Middle ground compromise

**Use Cases:**
- Production applications
- Good default choice
- When you need both speed and accuracy
- Limited GPU resources

**Training Time (100 epochs, 640px, CPU):**
- Fingerlings: ~4 hours
- Baby Shrimp: ~3 hours

---

### YOLOv8m (Medium) - High Accuracy ⭐ RECOMMENDED
```
Size: 49.7 MB
Parameters: 25.9M
FLOPs: 78.9G
```

**Pros:**
- ✅ High accuracy
- ✅ Good for small object detection
- ✅ Best accuracy/speed trade-off
- ✅ Robust to variations
- ✅ Recommended for production

**Cons:**
- ❌ Slower than s/n
- ❌ Larger model size
- ❌ Longer training time
- ❌ More memory usage

**Use Cases:**
- High accuracy requirements ⭐
- Small object detection
- Production systems with GPU
- When accuracy is priority

**Training Time (150 epochs, 1280px, CPU):**
- Fingerlings: ~10 hours
- Baby Shrimp: ~8 hours

**Training Time (150 epochs, 1280px, GPU):**
- Fingerlings: ~2-3 hours
- Baby Shrimp: ~1.5-2 hours

---

### YOLOv8l (Large) - Maximum Accuracy
```
Size: 83.7 MB
Parameters: 43.7M
FLOPs: 165.2G
```

**Pros:**
- ✅ Very high accuracy
- ✅ Excellent for small objects
- ✅ Best for challenging datasets
- ✅ Most robust model

**Cons:**
- ❌ Slow inference
- ❌ Large model size
- ❌ Long training time
- ❌ High memory usage
- ❌ Requires good GPU

**Use Cases:**
- Maximum accuracy needed
- Research projects
- Offline processing
- When speed is not critical

**Training Time (200 epochs, 1280px, CPU):**
- Fingerlings: ~20 hours
- Baby Shrimp: ~15 hours

**Training Time (200 epochs, 1280px, GPU):**
- Fingerlings: ~4-5 hours
- Baby Shrimp: ~3-4 hours

---

## 🎯 Which Model Should You Choose?

### Choose **YOLOv8n** if:
- ⚡ Speed is critical
- 📱 Deploying to edge devices
- 💾 Limited storage/memory
- 🚀 Quick prototyping
- ✅ Accuracy > 80% is acceptable

### Choose **YOLOv8s** if:
- ⚖️ Need balance of speed and accuracy
- 🖥️ Standard hardware
- 💼 Production with real-time requirements
- ✅ Accuracy > 85% is acceptable

### Choose **YOLOv8m** if: ⭐ RECOMMENDED
- 🎯 High accuracy is important
- 🐟 Detecting small objects (fingerlings, shrimp)
- 💪 Have decent hardware (GPU preferred)
- ⏰ Can wait for training
- ✅ Need accuracy > 90%

### Choose **YOLOv8l** if:
- 🏆 Maximum accuracy required
- 🔬 Research or critical applications
- 💻 Have powerful GPU
- ⏳ Time is not a constraint
- ✅ Need accuracy > 92%

## 📊 Expected Results for Your Dataset

### Fingerlings Detection

| Model | Epochs | Image Size | Precision | Recall | mAP50 | mAP50-95 | Training Time (CPU) |
|-------|--------|------------|-----------|--------|-------|----------|---------------------|
| **n** | 50 | 640 | 0.75-0.82 | 0.70-0.78 | 0.75-0.83 | 0.55-0.63 | ~2h |
| **s** | 100 | 640 | 0.80-0.87 | 0.75-0.83 | 0.82-0.88 | 0.60-0.68 | ~4h |
| **m** | 150 | 1280 | 0.85-0.92 | 0.80-0.88 | 0.88-0.94 | 0.65-0.75 | ~10h |
| **l** | 200 | 1280 | 0.88-0.94 | 0.83-0.90 | 0.90-0.96 | 0.70-0.80 | ~20h |

### Baby Shrimp Detection

| Model | Epochs | Image Size | Precision | Recall | mAP50 | mAP50-95 | Training Time (CPU) |
|-------|--------|------------|-----------|--------|-------|----------|---------------------|
| **n** | 50 | 640 | 0.70-0.78 | 0.65-0.73 | 0.72-0.80 | 0.50-0.58 | ~1.5h |
| **s** | 100 | 640 | 0.75-0.83 | 0.70-0.78 | 0.78-0.85 | 0.55-0.63 | ~3h |
| **m** | 100 | 1280 | 0.80-0.88 | 0.75-0.85 | 0.82-0.90 | 0.60-0.70 | ~8h |
| **l** | 150 | 1280 | 0.83-0.90 | 0.78-0.87 | 0.85-0.92 | 0.65-0.75 | ~15h |

## 💡 Recommendations

### For Development/Testing:
```bash
# Start with YOLOv8n for quick iteration
python modules/fingerlings/train.py \
  --model yolov8n.pt \
  --epochs 50 \
  --batch 4 \
  --device cpu
```

### For Production (Recommended): ⭐
```bash
# Use YOLOv8m for best accuracy/speed balance
python modules/fingerlings/train_high_accuracy.py \
  --model yolov8m.pt \
  --epochs 150 \
  --imgsz 1280 \
  --batch 2 \
  --device cpu
```

### For Maximum Accuracy:
```bash
# Use YOLOv8l when accuracy is critical
python modules/fingerlings/train_high_accuracy.py \
  --model yolov8l.pt \
  --epochs 200 \
  --imgsz 1280 \
  --batch 2 \
  --device cpu
```

## 🔄 Progressive Training Strategy

**Recommended approach:**

1. **Start with YOLOv8n** (2 hours)
   - Quick baseline
   - Verify dataset quality
   - Test pipeline

2. **Move to YOLOv8s** (4 hours)
   - Better accuracy
   - Still reasonable time
   - Good for iteration

3. **Final with YOLOv8m** (10 hours) ⭐
   - Production model
   - High accuracy
   - Best balance

4. **Optional: YOLOv8l** (20 hours)
   - If m is not enough
   - Maximum accuracy
   - Final optimization

## 📈 Performance vs Cost

```
Accuracy Gain:
n → s: +8-10% mAP50 (2x training time)
s → m: +6-8% mAP50 (2.5x training time)
m → l: +2-4% mAP50 (2x training time)

Diminishing returns after YOLOv8m!
```

## 🎯 Final Recommendation

**For Fingerlings & Baby Shrimp Detection:**

✅ **Best Choice: YOLOv8m**
- High accuracy (88-94% mAP50)
- Reasonable training time (~10h CPU)
- Good for small object detection
- Production ready
- Best ROI (Return on Investment)

Start with **YOLOv8m** unless you have specific constraints! 🐟📊✨

