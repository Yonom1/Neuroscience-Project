import os
import sys
import numpy as np
from skimage.io import imread, imsave
from skimage import img_as_ubyte
import warnings

# 忽略一些警告
warnings.filterwarnings("ignore")

# ================= 配置 =================
# 目标图片路径
TARGET_IMAGES = [
    r'E:\南京大学\课程\大三上学期\NS\project\dataset\test\persian_cat\000001.jpg',
    r'E:\南京大学\课程\大三上学期\NS\project\dataset\test\siamese_cat\000001.jpg'
]

# 输出目录
OUTPUT_DIR = r'E:\南京大学\课程\大三上学期\NS\project\FurtherExploration\noise_samples'

# 噪声参数
NOISE_LEVEL = 0.35
NUM_SAMPLES = 50

# ================= 图像处理函数 =================

def imload_rgb(path):
    """Load and return an RGB image in the range [0, 1]."""
    img = imread(path)
    if img.dtype == np.uint8:
        img = img / 255.0
    # 处理灰度图转RGB
    if len(img.shape) == 2:
        img = np.stack((img,)*3, axis=-1)
    elif img.shape[2] == 4: # 去掉 alpha 通道
        img = img[:,:,:3]
    return img

def save_img(image, path):
    """Save image."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if image.dtype != np.uint8:
        image = img_as_ubyte(np.clip(image, 0, 1))
    imsave(path, image)

def get_uniform_noise(low, high, nrow, ncol, rng=None):
    if rng is None:
        return np.random.uniform(low=low, high=high, size=(nrow, ncol))
    else:
        return rng.uniform(low=low, high=high, size=(nrow, ncol))

def apply_uniform_noise(image, width, rng=None):
    nrow = image.shape[0]
    ncol = image.shape[1]
    
    # 生成噪声 (单通道，然后广播到3通道)
    # 这里的 width 对应 generate_variants.py 中的 NOISE_LEVELS
    # generate_variants.py 中调用是 apply_uniform_noise(img, -width, width, rng)
    low = -width
    high = width
    
    noise = get_uniform_noise(low, high, nrow, ncol, rng)
    noise = np.stack((noise,)*3, axis=-1) # 扩展到3通道
    
    image = image + noise
    
    # clip values
    image = np.where(image < 0, 0, image)
    image = np.where(image > 1, 1, image)
    
    return image

# ================= 主逻辑 =================

def generate_noise_samples():
    # 使用随机种子，但为了每次生成的10张不一样，我们只在循环外初始化一次 rng
    # 或者每次循环不设种子
    rng = np.random.RandomState() 
    
    for img_path in TARGET_IMAGES:
        if not os.path.exists(img_path):
            print(f"Error: Image not found at {img_path}")
            continue
            
        print(f"Processing {os.path.basename(img_path)}...")
        
        try:
            img = imload_rgb(img_path)
        except Exception as e:
            print(f"  Error loading image: {e}")
            continue
            
        # 获取类别名作为子文件夹名
        # 假设路径结构是 .../class_name/img_name.jpg
        class_name = os.path.basename(os.path.dirname(img_path))
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        for i in range(NUM_SAMPLES):
            # 生成加噪图片
            noisy_img = apply_uniform_noise(img, NOISE_LEVEL, rng)
            
            # 保存路径: Further Exploration/noise_samples/class_name/img_name_sample_01.jpg
            save_name = f"{img_name}_noise_{NOISE_LEVEL}_sample_{i+1:02d}.jpg"
            save_path = os.path.join(OUTPUT_DIR, class_name, save_name)
            
            save_img(noisy_img, save_path)
            print(f"  Saved {save_name}")

    print("All samples generated!")

if __name__ == "__main__":
    generate_noise_samples()
