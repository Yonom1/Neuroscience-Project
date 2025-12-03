import os
import sys
import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from PIL import Image
import warnings
from tempfile import mkstemp

# 忽略一些警告
warnings.filterwarnings("ignore")

# ================= 配置 =================
SOURCE_DIR = './dataset'
DIRS_TO_PROCESS = ['test'] # 只处理 test 子集
OUTPUT_ROOT = './dataset_variants'

# Eidolon 扭曲档位设置 (5档)
# 调整 reach 参数，reach 越大扭曲越严重
# coherence 保持 1.0
# grain 保持 10.0
EIDOLON_LEVELS = [
    {'reach': 2.0, 'coherence': 1.0, 'grain': 10.0},  # Level 1
    {'reach': 4.0, 'coherence': 1.0, 'grain': 10.0},  # Level 2
    {'reach': 8.0, 'coherence': 1.0, 'grain': 10.0},  # Level 3
    {'reach': 16.0, 'coherence': 1.0, 'grain': 10.0}, # Level 4
    {'reach': 32.0, 'coherence': 1.0, 'grain': 10.0}, # Level 5
]

# ================= Eidolon 设置 =================
sys.path.append(os.path.abspath('./Eidolon'))

try:
    from eidolon import picture as pic
    from eidolon import helpers as hel
    from eidolon import scalespaces as scl
    EIDOLON_AVAILABLE = True
    print("Eidolon library loaded successfully.")
except ImportError:
    print("Error: Eidolon library not found. Please ensure the 'Eidolon' folder is in the root directory.")
    sys.exit(1)

# ================= 图像处理函数 =================

def imload_rgb(path):
    """Load and return an RGB image in the range [0, 1]."""
    img = imread(path)
    if img.dtype == np.uint8:
        img = img / 255.0
    if len(img.shape) == 2:
        img = np.stack((img,)*3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:,:,:3]
    return img

def save_img(image, path):
    """Save image."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if image.dtype != np.uint8:
        image = img_as_ubyte(np.clip(image, 0, 1))
    imsave(path, image)

# ================= Eidolon 核心函数 =================

SZ = 256
MIN_STD = (1 / np.sqrt(2))
MAX_STD = SZ / 4.0
STD_FAC = np.sqrt(2)

def load_pic(fname, sz=SZ, min_s=MIN_STD, max_s=MAX_STD, s_factor=STD_FAC):
    return pic.Picture(fname, sz, min_s, max_s, s_factor)

def data_to_pic(dat, sz=SZ, min_s=MIN_STD, max_s=MAX_STD, s_factor=STD_FAC):
    (outfd, fname) = mkstemp('.png')
    os.close(outfd)
    
    if dat.dtype != np.uint8:
        dat_uint8 = img_as_ubyte(np.clip(dat, 0, 1))
    else:
        dat_uint8 = dat
        
    imsave(fname, dat_uint8)
    
    try:
        p = load_pic(fname, sz, min_s, max_s, s_factor)
    finally:
        if os.path.exists(fname):
            os.remove(fname)
    return p

def partially_coherent_disarray(image, reach, coherence, grain):
    # Eidolon 处理灰度图
    gray = rgb2gray(image)
    
    p = data_to_pic(gray)
    
    fiducialDOGScaleSpace = scl.DOGScaleSpace(p)
    (h, w) = p.fatFiducialDataPlane.shape

    eidolonDataPlane = hel.PartiallyCoherentDisarray(fiducialDOGScaleSpace,
                                                     reach, coherence, grain,
                                                     w, h, p.MAX_SIGMA,
                                                     p.numScaleLevels,
                                                     p.scaleLevels)

    eidolon = p.DisembedDataPlane(eidolonDataPlane)
    # 转回 [0,1] float 3通道
    res = np.asarray(Image.fromarray(eidolon, 'L')) / 255.0
    return np.stack((res,)*3, axis=-1)

# ================= 主逻辑 =================

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
                try:
                    img = imload_rgb(img_path)
                except Exception as e:
                    print(f"    Error loading {img_name}: {e}")
                    continue
                
                # 生成 5 个档位的 Eidolon 变体
                for i, params in enumerate(EIDOLON_LEVELS):
                    level_idx = i + 1
                    save_dir = os.path.join(OUTPUT_ROOT, f'eidolon_level_{level_idx}', subset, class_name)
                    save_path = os.path.join(save_dir, img_name)
                    
                    if os.path.exists(save_path):
                        continue
                        
                    try:
                        img_eidolon = partially_coherent_disarray(
                            img, 
                            reach=params['reach'], 
                            coherence=params['coherence'], 
                            grain=params['grain']
                        )
                        save_img(img_eidolon, save_path)
                    except Exception as e:
                        print(f"    Error processing {img_name} at level {level_idx}: {e}")

    print("All Eidolon processing done!")

if __name__ == "__main__":
    process_dataset()
