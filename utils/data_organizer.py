import os
import shutil
import random

# config area
SOURCE_MAPPING = {
    "n02123394": "Persian cat", 
    "n02123597": "Siamese cat"
}

# 获取项目根目录 (假设此脚本在 utils/ 目录下，根目录为上一级)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_ROOT = os.path.join(PROJECT_ROOT, "my_experiment_data")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "origin")

TRAIN_RATIO = 0.8


def organize_dataset():
    print(f"Checking source directories in '{SOURCE_ROOT}'...")
    for source_name in SOURCE_MAPPING.keys():
        source_path = os.path.join(SOURCE_ROOT, source_name)
        if not os.path.exists(source_path):
            print(f"Error: Source directory '{source_name}' not found. Please ensure the extracted folders are in the same directory as the script, or modify SOURCE_ROOT.")
            return

    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning old output directory '{OUTPUT_DIR}'...")
        shutil.rmtree(OUTPUT_DIR)
    
    for split in ['train', 'val']:
        for class_name in SOURCE_MAPPING.values():
            os.makedirs(os.path.join(OUTPUT_DIR, split, class_name), exist_ok=True)

    print(f"Created directory structure in '{OUTPUT_DIR}'")

    total_images_count = 0
    
    for source_name, target_name in SOURCE_MAPPING.items():
        source_path = os.path.join(SOURCE_ROOT, source_name)
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPEG', '.JPG')
        all_files = [f for f in os.listdir(source_path) if f.lower().endswith(valid_extensions)]
        
        random.shuffle(all_files)
        
        split_idx = int(len(all_files) * TRAIN_RATIO)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]
        
        print(f"\nProcessing {target_name} ({source_name}):")
        print(f"  - Total: {len(all_files)} images")
        print(f"  - Train: {len(train_files)}")
        print(f"  - Val:   {len(val_files)}")
        
        def copy_files(file_list, split_type):
            count = 0
            for file_name in file_list:
                src = os.path.join(source_path, file_name)
                dst = os.path.join(OUTPUT_DIR, split_type, target_name, file_name)
                try:
                    shutil.copy2(src, dst)
                    count += 1
                except Exception as e:
                    print(f"    Warning: Failed to copy {file_name}. Error: {e}")
            return count

        t_count = copy_files(train_files, 'train')
        v_count = copy_files(val_files, 'val')
        total_images_count += (t_count + v_count)

    print("\n" + "="*30)
    print(f"Done! Processed a total of {total_images_count} images.")
    print(f"Dataset is ready at: {OUTPUT_DIR}")
    print("="*30)

if __name__ == "__main__":
    organize_dataset()