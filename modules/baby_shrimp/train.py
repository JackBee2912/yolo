#!/usr/bin/env python3
"""
Train YOLO model for Baby Shrimp Detection and Counting
"""
import argparse
import torch
from ultralytics import YOLO
from pathlib import Path
import yaml


def setup_training_config(data_yaml_path):
    """Setup and validate training configuration"""
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print("="*60)
    print("📊 DATASET CONFIGURATION")
    print("="*60)
    print(f"Classes: {data_config.get('nc', 'Unknown')}")
    print(f"Names: {data_config.get('names', 'Unknown')}")
    print(f"Train path: {data_config.get('train', 'Unknown')}")
    print(f"Val path: {data_config.get('val', 'Unknown')}")
    print("="*60 + "\n")
    
    return data_config


def train_baby_shrimp(
    data_yaml,
    model_name="yolo11n.pt",
    epochs=30,
    imgsz=640,
    batch=4,
    name="baby_shrimp_training",
    patience=50,
    device=None
):
    """
    Train YOLO model for baby shrimp detection
    
    Args:
        data_yaml: Path to dataset YAML (e.g., 'baby-shrimp-1/data.yaml')
        model_name: Base model to use
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        name: Training run name
        patience: Early stopping patience
        device: Device to use (None for auto-detect)
    """
    print("="*60)
    print("🦐 BABY SHRIMP TRAINING")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Patience: {patience}")
    print("="*60 + "\n")
    
    # Setup config
    setup_training_config(data_yaml)
    
    # Load model
    print("📦 Loading model...")
    model = YOLO(model_name)
    
    # Auto-detect device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
            print("🚀 Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            device = "cuda"
            print("🚀 Using NVIDIA GPU (CUDA)")
        else:
            device = "cpu"
            print("💻 Using CPU")
    
    print(f"Device: {device}\n")
    
    # Training parameters optimized for small dense objects
    print("🎯 Starting training...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project='modules/baby_shrimp/runs/detect',  # Save in module directory
        name=name,
        patience=patience,
        device=device,
        
        # Stability settings
        workers=0,
        deterministic=True,
        seed=42,
        
        # Hyperparameters
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Augmentation for dense small objects
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        
        # Validation
        val=True,
        save=True,
        save_period=-1,
        plots=True,
        amp=True,
        verbose=True,
    )
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"📁 Results: modules/baby_shrimp/runs/detect/{name}")
    print(f"🏆 Best model: modules/baby_shrimp/runs/detect/{name}/weights/best.pt")
    print("💡 Tip: Use monitor.py to view training progress in real-time")
    print("="*60 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO for Baby Shrimp Detection")
    parser.add_argument("--data", default="modules/baby_shrimp/dataset/data.yaml", help="Path to dataset YAML")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--name", default="baby_shrimp_training", help="Run name")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--device", default=None, help="Device (mps/cuda/cpu)")

    args = parser.parse_args()

    # Call with correct parameter names
    train_baby_shrimp(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=args.patience,
        device=args.device
    )


if __name__ == "__main__":
    main()

