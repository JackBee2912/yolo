#!/usr/bin/env python3
"""
Real-time training monitor for YOLO baby_shrimp detection
Displays live charts of training metrics
"""

import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import sys


class TrainingMonitor:
    def __init__(self, results_path):
        self.results_path = Path(results_path)
        self.csv_path = self.results_path / "results.csv"
        
        # Setup plot
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('🐟 Baby Shrimp Training Monitor', fontsize=16, fontweight='bold')
        
        # Configure axes
        self.ax_loss = self.axes[0, 0]
        self.ax_metrics = self.axes[0, 1]
        self.ax_lr = self.axes[1, 0]
        self.ax_summary = self.axes[1, 1]
        
        self.ax_loss.set_title('Training Loss')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True, alpha=0.3)
        
        self.ax_metrics.set_title('Validation Metrics')
        self.ax_metrics.set_xlabel('Epoch')
        self.ax_metrics.set_ylabel('Score')
        self.ax_metrics.grid(True, alpha=0.3)
        
        self.ax_lr.set_title('Learning Rate')
        self.ax_lr.set_xlabel('Epoch')
        self.ax_lr.set_ylabel('LR')
        self.ax_lr.grid(True, alpha=0.3)
        
        self.ax_summary.axis('off')
        
        plt.tight_layout()
        
    def update(self, frame):
        """Update plots with latest data"""
        if not self.csv_path.exists():
            self.ax_summary.clear()
            self.ax_summary.axis('off')
            self.ax_summary.text(0.5, 0.5, 
                               f'⏳ Waiting for training to start...\n\n'
                               f'Looking for:\n{self.csv_path}',
                               ha='center', va='center', fontsize=12)
            return
        
        try:
            # Read CSV
            df = pd.read_csv(self.csv_path)
            df.columns = df.columns.str.strip()  # Remove whitespace
            
            if len(df) == 0:
                return
            
            # Clear axes
            self.ax_loss.clear()
            self.ax_metrics.clear()
            self.ax_lr.clear()
            self.ax_summary.clear()
            
            # Plot losses
            self.ax_loss.set_title('Training Loss')
            self.ax_loss.set_xlabel('Epoch')
            self.ax_loss.set_ylabel('Loss')
            self.ax_loss.grid(True, alpha=0.3)
            
            if 'train/box_loss' in df.columns:
                self.ax_loss.plot(df['epoch'], df['train/box_loss'], label='Box Loss', marker='o', markersize=3)
            if 'train/cls_loss' in df.columns:
                self.ax_loss.plot(df['epoch'], df['train/cls_loss'], label='Class Loss', marker='s', markersize=3)
            if 'train/dfl_loss' in df.columns:
                self.ax_loss.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', marker='^', markersize=3)
            self.ax_loss.legend()
            
            # Plot metrics
            self.ax_metrics.set_title('Validation Metrics')
            self.ax_metrics.set_xlabel('Epoch')
            self.ax_metrics.set_ylabel('Score')
            self.ax_metrics.grid(True, alpha=0.3)
            
            if 'metrics/precision(B)' in df.columns:
                self.ax_metrics.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', marker='o', markersize=3)
            if 'metrics/recall(B)' in df.columns:
                self.ax_metrics.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', marker='s', markersize=3)
            if 'metrics/mAP50(B)' in df.columns:
                self.ax_metrics.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', marker='^', markersize=3)
            if 'metrics/mAP50-95(B)' in df.columns:
                self.ax_metrics.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', marker='d', markersize=3)
            self.ax_metrics.legend()
            
            # Plot learning rate
            self.ax_lr.set_title('Learning Rate')
            self.ax_lr.set_xlabel('Epoch')
            self.ax_lr.set_ylabel('LR')
            self.ax_lr.grid(True, alpha=0.3)
            
            if 'lr/pg0' in df.columns:
                self.ax_lr.plot(df['epoch'], df['lr/pg0'], label='LR', marker='o', markersize=3, color='green')
            self.ax_lr.legend()
            
            # Summary text
            self.ax_summary.axis('off')
            latest = df.iloc[-1]
            
            summary_text = f"📊 Training Summary\n"
            summary_text += f"{'='*40}\n\n"
            summary_text += f"Epoch: {int(latest['epoch'])}\n\n"
            
            if 'metrics/mAP50(B)' in df.columns:
                summary_text += f"mAP50: {latest['metrics/mAP50(B)']:.4f}\n"
            if 'metrics/mAP50-95(B)' in df.columns:
                summary_text += f"mAP50-95: {latest['metrics/mAP50-95(B)']:.4f}\n"
            if 'metrics/precision(B)' in df.columns:
                summary_text += f"Precision: {latest['metrics/precision(B)']:.4f}\n"
            if 'metrics/recall(B)' in df.columns:
                summary_text += f"Recall: {latest['metrics/recall(B)']:.4f}\n\n"
            
            if 'train/box_loss' in df.columns:
                summary_text += f"Box Loss: {latest['train/box_loss']:.4f}\n"
            if 'train/cls_loss' in df.columns:
                summary_text += f"Class Loss: {latest['train/cls_loss']:.4f}\n"
            if 'train/dfl_loss' in df.columns:
                summary_text += f"DFL Loss: {latest['train/dfl_loss']:.4f}\n"
            
            self.ax_summary.text(0.1, 0.5, summary_text, 
                               ha='left', va='center', fontsize=11, family='monospace')
            
            plt.tight_layout()
            
        except Exception as e:
            print(f"Error reading CSV: {e}")
    
    def start(self):
        """Start monitoring"""
        print(f"🔍 Monitoring training at: {self.results_path}")
        print(f"📊 Charts will update every 5 seconds")
        print(f"Press Ctrl+C to stop\n")
        
        ani = FuncAnimation(self.fig, self.update, interval=5000, cache_frame_data=False)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Monitor YOLO training progress")
    parser.add_argument("--path", required=True, help="Path to training results directory")
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.path)
    monitor.start()


if __name__ == "__main__":
    main()

