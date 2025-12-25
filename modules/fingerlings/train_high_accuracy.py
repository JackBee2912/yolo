#!/usr/bin/env python3
"""
High Accuracy Training Script for Fish Fingerlings Detection
Optimized for maximum accuracy with longer training time
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def train_high_accuracy(
    model_path='yolov8m.pt',  # Medium model for better accuracy
    data_yaml='modules/fingerlings/dataset/data.yaml',
    epochs=150,  # More epochs for better convergence
    imgsz=1280,  # Larger image size for better detection
    batch=2,  # Small batch for stability
    device='cpu',
    name='fingerlings_high_accuracy',
    patience=50  # More patience for convergence
):
    """
    Train YOLO model with high accuracy configuration
    
    High Accuracy Settings:
    - Larger model (yolov8m or yolov8l)
    - More epochs (150-200)
    - Larger image size (1280)
    - Lower learning rate for stability
    - More augmentation
    - Longer patience
    """
    
    print("="*60)
    print("🎯 HIGH ACCURACY TRAINING MODE")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    # Load model
    print(f"📦 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Training with high accuracy parameters
    print("🎯 Starting high accuracy training...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project='modules/fingerlings/runs/detect',
        name=name,
        patience=patience,
        device=device,
        
        # Stability settings
        workers=0,
        deterministic=True,
        seed=42,
        
        # Optimizer - Lower LR for better convergence
        optimizer='AdamW',
        lr0=0.0005,  # Lower initial learning rate
        lrf=0.001,   # Lower final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,  # More warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Enhanced Augmentation for better generalization
        hsv_h=0.02,      # Slightly more hue variation
        hsv_s=0.8,       # More saturation variation
        hsv_v=0.5,       # More value variation
        degrees=5.0,     # Small rotation
        translate=0.15,  # More translation
        scale=0.6,       # More scale variation
        shear=2.0,       # Small shear
        perspective=0.0001,  # Tiny perspective
        flipud=0.0,      # No vertical flip
        fliplr=0.5,      # Horizontal flip
        mosaic=1.0,      # Keep mosaic
        mixup=0.1,       # Add mixup for regularization
        copy_paste=0.1,  # Copy-paste augmentation
        
        # Validation & Saving
        val=True,
        save=True,
        save_period=10,  # Save every 10 epochs
        plots=True,
        amp=True,
        verbose=True,
        
        # Additional settings for accuracy
        close_mosaic=20,  # Disable mosaic in last 20 epochs
        box=7.5,         # Box loss weight
        cls=0.5,         # Class loss weight
        dfl=1.5,         # DFL loss weight
    )
    
    print("\n" + "="*60)
    print("✅ HIGH ACCURACY TRAINING COMPLETE!")
    print("="*60)
    print(f"📁 Results: modules/fingerlings/runs/detect/{name}")
    print(f"🏆 Best model: modules/fingerlings/runs/detect/{name}/weights/best.pt")
    print("💡 Tip: Use monitor.py to view training progress")
    print("="*60 + "\n")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='High Accuracy Training for Fingerlings Detection')
    parser.add_argument('--model', type=str, default='yolov8m.pt',
                       help='Model to use (yolov8m.pt, yolov8l.pt, yolov8x.pt)')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of epochs (default: 150)')
    parser.add_argument('--imgsz', type=int, default=1280,
                       help='Image size (default: 1280)')
    parser.add_argument('--batch', type=int, default=2,
                       help='Batch size (default: 2)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu, mps, cuda)')
    parser.add_argument('--name', type=str, default='fingerlings_high_accuracy',
                       help='Experiment name')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience (default: 50)')
    
    args = parser.parse_args()
    
    train_high_accuracy(
        model_path=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=args.patience
    )

