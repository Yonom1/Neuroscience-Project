import os
import shutil
import random

# ================= 配置区域 =================
# 1. 定义源文件夹的名字 (WNID) 和你想要映射的目标类别名
# 格式: "原始文件夹名": "目标类别名"
SOURCE_MAPPING = {
    "n02123394": "Persian cat",   # Persian Cat
    "n02123597": "Siamese cat" # Siamese Cat
}

# 2. 定义源文件所在的根目录 (默认是当前脚本所在目录)
SOURCE_ROOT = "./my_experiment_data" 

# 3. 定义输出目录
OUTPUT_DIR = "./dataset"

# 4. 训练集占比 (剩余的给验证集)
TRAIN_RATIO = 0.8

# ================= 执行逻辑 =================

def organize_dataset():
    # 0. 检查源文件夹是否存在
    print(f"Checking source directories in '{SOURCE_ROOT}'...")
    for source_name in SOURCE_MAPPING.keys():
        source_path = os.path.join(SOURCE_ROOT, source_name)
        if not os.path.exists(source_path):
            print(f"❌ Error: 找不到源文件夹 '{source_name}'。请确保解压后的文件夹和脚本在同一目录下，或者修改 SOURCE_ROOT。")
            return

    # 1. 清理/创建输出目录
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning old output directory '{OUTPUT_DIR}'...")
        shutil.rmtree(OUTPUT_DIR)
    
    # 创建 train 和 val 的子目录
    for split in ['train', 'val']:
        for class_name in SOURCE_MAPPING.values():
            os.makedirs(os.path.join(OUTPUT_DIR, split, class_name), exist_ok=True)

    print(f"Created directory structure in '{OUTPUT_DIR}'")

    # 2. 开始搬运图片
    total_images_count = 0
    
    for source_name, target_name in SOURCE_MAPPING.items():
        source_path = os.path.join(SOURCE_ROOT, source_name)
        
        # 获取所有图片文件 (过滤掉非图片文件)
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPEG', '.JPG')
        all_files = [f for f in os.listdir(source_path) if f.lower().endswith(valid_extensions)]
        
        # 随机打乱
        random.shuffle(all_files)
        
        # 计算分割点
        split_idx = int(len(all_files) * TRAIN_RATIO)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]
        
        print(f"\nProcessing {target_name} ({source_name}):")
        print(f"  - Total: {len(all_files)} images")
        print(f"  - Train: {len(train_files)}")
        print(f"  - Val:   {len(val_files)}")
        
        # 复制文件函数
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

        # 执行复制
        t_count = copy_files(train_files, 'train')
        v_count = copy_files(val_files, 'val')
        total_images_count += (t_count + v_count)

    print("\n" + "="*30)
    print(f"✅ 完成！共处理 {total_images_count} 张图片。")
    print(f"数据集已准备好，路径为: {OUTPUT_DIR}")
    print("="*30)

if __name__ == "__main__":
    organize_dataset()