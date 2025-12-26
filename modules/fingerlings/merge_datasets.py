#!/usr/bin/env python3
"""
Merge dataset and dataset-v2 into a combined dataset
"""
import shutil
from pathlib import Path

def merge_datasets():
    """Merge two datasets into dataset-merged"""
    
    # Paths
    dataset1 = Path("modules/fingerlings/dataset")
    dataset2 = Path("modules/fingerlings/dataset-v2")
    merged = Path("modules/fingerlings/dataset-merged")
    
    print("🔄 Merging datasets...")
    print(f"📁 Dataset 1: {dataset1}")
    print(f"📁 Dataset 2: {dataset2}")
    print(f"📁 Output: {merged}")
    
    # Create merged directory structure
    for split in ['train', 'valid', 'test']:
        (merged / split / 'images').mkdir(parents=True, exist_ok=True)
        (merged / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Copy files from both datasets
    for split in ['train', 'valid', 'test']:
        print(f"\n📦 Merging {split} split...")
        
        count = 0
        # Copy from dataset 1
        for img_path in (dataset1 / split / 'images').glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy2(img_path, merged / split / 'images' / img_path.name)
                
                # Copy corresponding label
                label_path = dataset1 / split / 'labels' / f"{img_path.stem}.txt"
                if label_path.exists():
                    shutil.copy2(label_path, merged / split / 'labels' / label_path.name)
                count += 1
        
        print(f"  ✅ Copied {count} images from dataset 1")
        
        # Copy from dataset 2
        count2 = 0
        for img_path in (dataset2 / split / 'images').glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Rename if conflict
                new_name = img_path.name
                if (merged / split / 'images' / new_name).exists():
                    new_name = f"v2_{img_path.name}"
                
                shutil.copy2(img_path, merged / split / 'images' / new_name)
                
                # Copy corresponding label
                label_path = dataset2 / split / 'labels' / f"{img_path.stem}.txt"
                if label_path.exists():
                    label_name = f"{Path(new_name).stem}.txt"
                    shutil.copy2(label_path, merged / split / 'labels' / label_name)
                count2 += 1
        
        print(f"  ✅ Copied {count2} images from dataset 2")
        print(f"  📊 Total: {count + count2} images")
    
    # Create data.yaml
    yaml_content = """train: train/images
val: valid/images
test: test/images

nc: 1
names: ['fingerling']
"""
    
    with open(merged / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("\n✅ Merge completed!")
    print(f"📁 Merged dataset: {merged}")
    print(f"📄 Config file: {merged / 'data.yaml'}")

if __name__ == "__main__":
    merge_datasets()

