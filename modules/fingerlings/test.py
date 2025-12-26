#!/usr/bin/env python3
"""
Test YOLO model for Fish Fingerlings Detection and Counting
"""
import argparse
from ultralytics import YOLO
from pathlib import Path
import cv2


def test_fingerlings(model_path, image_path, conf=0.25, imgsz=640, show_dots=True):
    """
    Test model on single image and return detection count
    
    Args:
        model_path: Path to trained model weights
        image_path: Path to test image
        conf: Confidence threshold
        imgsz: Image size for inference
        show_dots: Draw red dots on detections
    
    Returns:
        Number of detections
    """
    print(f"📦 Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"🔍 Testing image: {image_path}")
    results = model.predict(
        source=image_path,
        imgsz=imgsz,
        conf=conf,
        save=False,
        verbose=False
    )
    
    detections = results[0].boxes
    count = len(detections)
    
    if show_dots:
        img = cv2.imread(image_path)

        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(img, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)

        # Add count text on image
        text = f"Count: {count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3

        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Draw background rectangle
        padding = 10
        cv2.rectangle(img,
                     (10, 10),
                     (10 + text_width + padding * 2, 10 + text_height + padding * 2),
                     (0, 0, 0),
                     -1)

        # Draw text
        cv2.putText(img, text,
                   (10 + padding, 10 + text_height + padding),
                   font, font_scale, (0, 255, 0), thickness)

        output_dir = Path("runs/detect/fingerlings_predict_dots")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / Path(image_path).name
        cv2.imwrite(str(output_path), img)
        save_location = str(output_path)
    else:
        save_location = str(results[0].save_dir)
    
    print("\n" + "="*50)
    print("✅ DETECTION RESULTS")
    print("="*50)
    print(f"📸 Image: {Path(image_path).name}")
    print(f"🐟 Count: {count} fingerlings detected")
    print(f"📊 Confidence: {conf}")
    print(f"💾 Saved to: {save_location}")
    print("="*50 + "\n")
    
    return count


def test_batch(model_path, images_dir, conf=0.25, imgsz=640, show_dots=True):
    """Test model on multiple images"""
    images_path = Path(images_dir)
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    
    print(f"📁 Found {len(image_files)} images\n")
    
    results = {}
    total_count = 0
    
    for img_file in image_files:
        count = test_fingerlings(model_path, str(img_file), conf, imgsz, show_dots)
        results[img_file.name] = count
        total_count += count
    
    print("\n" + "="*50)
    print("📊 BATCH TEST SUMMARY")
    print("="*50)
    print(f"Total images: {len(image_files)}")
    print(f"Total detected: {total_count}")
    print(f"Average: {total_count / len(image_files):.1f}")
    print("="*50 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test Fish Fingerlings Detection")
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--image", help="Path to single test image")
    parser.add_argument("--images-dir", help="Path to directory with images")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--no-dots", action="store_true", help="Don't draw red dots")
    
    args = parser.parse_args()
    
    if not args.image and not args.images_dir:
        print("❌ Error: Must provide --image or --images-dir")
        return
    
    if args.image:
        test_fingerlings(args.model, args.image, args.conf, args.imgsz, not args.no_dots)
    elif args.images_dir:
        test_batch(args.model, args.images_dir, args.conf, args.imgsz, not args.no_dots)


if __name__ == "__main__":
    main()

