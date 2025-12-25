#!/bin/bash

# Demo script to show how to use the training monitor

echo "🎬 YOLO Training Monitor Demo"
echo "=============================="
echo ""
echo "This demo shows how to train with real-time monitoring."
echo ""
echo "📋 Steps:"
echo "1. Terminal 1: Start training"
echo "2. Terminal 2: Start monitor (after ~30 seconds)"
echo ""
echo "Press Enter to see the commands..."
read

echo ""
echo "🖥️  TERMINAL 1 - Start Training:"
echo "================================"
echo ""
cat << 'EOF'
source .venv311/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Train fingerlings (50 epochs, ~2-3 hours on CPU)
python modules/fingerlings/train.py \
  --model yolov8n.pt \
  --epochs 50 \
  --batch 2 \
  --device cpu \
  --name fingerlings_training

# OR train baby shrimp (30 epochs, ~1-2 hours on CPU)
python modules/baby_shrimp/train.py \
  --model yolov8n.pt \
  --epochs 30 \
  --batch 2 \
  --device cpu \
  --name baby_shrimp_training
EOF

echo ""
echo "Press Enter to see Terminal 2 commands..."
read

echo ""
echo "🖥️  TERMINAL 2 - Monitor Progress:"
echo "==================================="
echo ""
cat << 'EOF'
source .venv311/bin/activate

# Wait ~30 seconds for training to create results.csv
# Then start monitoring:

# For fingerlings:
python modules/fingerlings/monitor.py \
  --path runs/detect/fingerlings_training

# OR for baby shrimp:
python modules/baby_shrimp/monitor.py \
  --path runs/detect/baby_shrimp_training
EOF

echo ""
echo "📊 What you'll see in the monitor:"
echo "- Top Left: Training losses (box, class, DFL)"
echo "- Top Right: Validation metrics (precision, recall, mAP)"
echo "- Bottom Left: Learning rate schedule"
echo "- Bottom Right: Latest epoch summary"
echo ""
echo "Charts update every 5 seconds automatically!"
echo "Press Ctrl+C to stop monitoring."
echo ""
echo "✅ Demo complete!"

