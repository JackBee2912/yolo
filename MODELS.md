# YOLO Models Setup

Model weights are not included in this repository due to GitHub file size limits.

## 📥 Download Pre-trained Models

### YOLOv8 Models
```bash
# Download YOLOv8 models
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt
```

### YOLOv11 Models
```bash
# Download YOLOv11 models
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt
```

Or they will be automatically downloaded when you run training for the first time.

## 📁 Trained Models

Your trained models will be saved in:
```
modules/fingerlings/runs/detect/{training_name}/weights/
├── best.pt    # Best model based on validation metrics
└── last.pt    # Last epoch model
```

**Note:** Add `runs/` to `.gitignore` to avoid pushing large trained models.

## 🔗 Model Sources

- YOLOv8: https://github.com/ultralytics/ultralytics
- YOLOv11: https://github.com/ultralytics/ultralytics
- Ultralytics Hub: https://hub.ultralytics.com/

