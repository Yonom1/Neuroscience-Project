import os
import sys
import numpy as np
import cv2
import random
import warnings

# 忽略一些警告
warnings.filterwarnings("ignore")

# ================= 配置 =================
SOURCE_DIR = './dataset'
DIRS_TO_PROCESS = ['test'] # 只处理 test 子集
OUTPUT_ROOT = './dataset_variants'

# Jigsaw 档位设置 (网格数量 N)
# N=1: 原图 (不生成)
# N=2: 2x2 (4块)
# N=4: 4x4 (16块)
# N=8: 8x8 (64块)
# N=16: 16x16 (256块)
JIGSAW_LEVELS = [2, 4, 8, 16]

# ================= 图像处理函数 =================

def jigsaw_scramble(img, grid_n):
    """
    Apply Jigsaw Permutation.
    Args:
        img: Input image (numpy array, BGR or RGB).
        grid_n: The granularity of the scramble (N x N grid).
    """
    h, w = img.shape[:2]
    
    # Calculate block size
    bh, bw = h // grid_n, w // grid_n
    
    # 如果图片尺寸不能被 grid_n 整除，可能会丢弃边缘像素，或者 resize
    # 这里为了简单，我们先 resize 到能整除的大小，或者只取整除部分
    # 考虑到 dataset 通常是 224x224 或 256x256，通常能被 2,4,8,16,32 整除
    # 如果不能整除，resize 是比较安全的做法
    
    target_h = bh * grid_n
    target_w = bw * grid_n
    
    if target_h != h or target_w != w:
        img = cv2.resize(img, (target_w, target_h))
        h, w = target_h, target_w

    # Split into blocks
    blocks = []
    for y in range(0, h, bh):
        for x in range(0, w, bw):
            blocks.append(img[y:y+bh, x:x+bw])
    
    # Shuffle blocks (The corruption step)
    # 使用固定的 seed 还是随机？通常为了多样性是随机的。
    # 但为了实验可复现性，我们可以对每张图使用基于其文件名的 seed，或者完全随机。
    # 这里使用完全随机，因为每张图都是独立的样本。
    random.shuffle(blocks)
    
    # Reassemble
    new_img = np.zeros_like(img)
    idx = 0
    for y in range(0, h, bh):
        for x in range(0, w, bw):
            new_img[y:y+bh, x:x+bw] = blocks[idx]
            idx += 1
                
    return new_img

def process_dataset():
    for subset in DIRS_TO_PROCESS:
        subset_path = os.path.join(SOURCE_DIR, subset)
        if not os.path.exists(subset_path):
            continue
            
        print(f"Processing {subset}...")
        
        for class_name in os.listdir(subset_path):
            class_path = os.path.join(subset_path, class_name)
            if not os.path.isdir(class_path):
                continue
                
            print(f"  Processing class: {class_name}")
            
            for img_name in os.listdir(class_path):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                    
                img_path = os.path.join(class_path, img_name)
                
                # 使用 cv2 读取 (BGR)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"    Error loading {img_name}")
                    continue
                
                # 生成 Jigsaw 变体
                for i, grid_n in enumerate(JIGSAW_LEVELS):
                    # 文件夹命名: dataset_variants/jigsaw_level_1/test/class_name/
                    # level_1 对应 JIGSAW_LEVELS[0] (2x2)
                    save_dir = os.path.join(OUTPUT_ROOT, f'jigsaw_level_{i+1}', subset, class_name)
                    save_path = os.path.join(save_dir, img_name)
                    
                    if os.path.exists(save_path):
                        continue
                        
                    try:
                        # 生成拼图
                        img_jigsaw = jigsaw_scramble(img, grid_n)
                        
                        # 确保目录存在
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        
                        # 保存 (cv2 使用 BGR，直接保存即可)
                        cv2.imwrite(save_path, img_jigsaw)
                        
                    except Exception as e:
                        print(f"    Error processing {img_name} at grid {grid_n}: {e}")

    print("All Jigsaw processing done!")

if __name__ == "__main__":
    process_dataset()
