import os
import sys
import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from PIL import Image
import warnings

# 忽略一些警告
warnings.filterwarnings("ignore")

# ================= 配置 =================
SOURCE_DIR = './dataset'
DIRS_TO_PROCESS = ['train', 'val', 'test'] # 处理所有子集
OUTPUT_ROOT = './dataset_variants'

# 档位设置
# 对比度：1.0 是原图，0.0 是全灰。我们设置5个档位，逐渐降低对比度。
# 例如：0.9, 0.7, 0.5, 0.3, 0.1
CONTRAST_LEVELS = [0.9, 0.7, 0.5, 0.3, 0.1]

# 噪声：0.0 是无噪声。我们设置6个档位，逐渐增加噪声。
# 例如：0.05, 0.20, 0.35, 0.5, 0.75, 1.0
NOISE_LEVELS = [0.05, 0.20, 0.35, 0.5, 0.75, 1.0]

# ================= Eidolon 设置 =================
# 将 Eidolon 库加入路径
sys.path.append(os.path.abspath('./Eidolon'))

try:
    from eidolon import picture as pic
    from eidolon import helpers as hel
    from eidolon import scalespaces as scl
    from eidolon import noise as noi
    # import wrapper_utils as wr # Removed
    EIDOLON_AVAILABLE = False
    print("Eidolon library loaded successfully.")
except ImportError:
    print("Eidolon library not found or failed to load. Eidolon transformation will be skipped or replaced.")
    EIDOLON_AVAILABLE = False

# ================= 图像处理函数 (来自 image-manipulation.py) =================

def imload_rgb(path):
    """Load and return an RGB image in the range [0, 1]."""
    img = imread(path)
    if img.dtype == np.uint8:
        img = img / 255.0
    # 处理灰度图转RGB (如果dataset里混入了灰度图)
    if len(img.shape) == 2:
        img = np.stack((img,)*3, axis=-1)
    elif img.shape[2] == 4: # 去掉 alpha 通道
        img = img[:,:,:3]
    return img

def save_img(image, path):
    """Save image."""
    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 转换回 uint8
    if image.dtype != np.uint8:
        image = img_as_ubyte(np.clip(image, 0, 1))
    
    imsave(path, image)

def adjust_contrast(image, contrast_level):
    """Return the image scaled to a certain contrast level in [0, 1]."""
    assert(contrast_level >= 0.0), "contrast_level too low."
    assert(contrast_level <= 1.0), "contrast_level too high."
    return (1-contrast_level)/2.0 + image * contrast_level # image.dot(contrast_level) for scalar is just multiply

def grayscale_contrast(image, contrast_level):
    """Convert to grayscale. Adjust contrast."""
    # rgb2gray 返回 [0, 1]
    gray = rgb2gray(image)
    # 保持3通道以便后续处理或保存为RGB格式的灰度图
    gray_3ch = np.stack((gray,)*3, axis=-1)
    return adjust_contrast(gray_3ch, contrast_level)

def get_uniform_noise(low, high, nrow, ncol, rng=None):
    if rng is None:
        return np.random.uniform(low=low, high=high, size=(nrow, ncol))
    else:
        return rng.uniform(low=low, high=high, size=(nrow, ncol))

def apply_uniform_noise(image, low, high, rng=None):
    nrow = image.shape[0]
    ncol = image.shape[1]
    
    # 生成噪声 (单通道，然后广播到3通道，或者生成3通道噪声？)
    # 原代码似乎是针对灰度图设计的。如果是彩色图，我们通常对每个通道加噪声或者加同一种噪声。
    # 这里我们先转灰度再加噪声，符合 object-recognition 的逻辑 (uniform_noise 函数里调用了 grayscale_contrast)
    
    noise = get_uniform_noise(low, high, nrow, ncol, rng)
    noise = np.stack((noise,)*3, axis=-1) # 扩展到3通道
    
    image = image + noise
    
    # clip values
    image = np.where(image < 0, 0, image)
    image = np.where(image > 1, 1, image)
    
    return image

def uniform_noise_transform(image, width, contrast_level, rng):
    """Convert to grayscale. Adjust contrast. Apply uniform noise."""
    # 先灰度+对比度
    img_gray_contrast = grayscale_contrast(image, contrast_level)
    return apply_uniform_noise(img_gray_contrast, -width, width, rng)

# ================= Eidolon 辅助函数 (简化版 wrapper.py) =================
if EIDOLON_AVAILABLE:
    from tempfile import mkstemp
    
    SZ = 256
    MIN_STD = (1 / np.sqrt(2))
    MAX_STD = SZ / 4.0
    STD_FAC = np.sqrt(2)

    def load_pic(fname, sz=SZ, min_s=MIN_STD, max_s=MAX_STD, s_factor=STD_FAC):
        return pic.Picture(fname, sz, min_s, max_s, s_factor)

    def data_to_pic(dat, sz=SZ, min_s=MIN_STD, max_s=MAX_STD, s_factor=STD_FAC):
        # Eidolon 需要从文件读取，所以我们创建一个临时文件
        (outfd, fname) = mkstemp('.png')
        os.close(outfd) # Close the file descriptor immediately so imsave can use the file

        # 保存 dat (假设是 [0,1] float)
        if dat.dtype != np.uint8:
            dat_uint8 = img_as_ubyte(np.clip(dat, 0, 1))
        else:
            dat_uint8 = dat
            
        imsave(fname, dat_uint8)
        
        try:
            p = load_pic(fname, sz, min_s, max_s, s_factor)
        finally:
            if os.path.exists(fname):
                os.remove(fname) # 删除临时文件
        return p

    def partially_coherent_disarray(image, reach, coherence, grain):
        # image 是 numpy array [0,1] RGB
        # Eidolon 通常处理灰度。我们先转灰度。
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
        # Eidolon 返回的是 uint8 数组 (from Image.fromarray(..., 'L'))
        # 我们把它转回 [0,1] float 3通道
        res = np.asarray(Image.fromarray(eidolon, 'L')) / 255.0
        return np.stack((res,)*3, axis=-1)

# ================= 主逻辑 =================

def process_dataset():
    rng = np.random.RandomState(seed=42)
    
    # 遍历 dataset 下的 train, val, test
    for subset in DIRS_TO_PROCESS:
        subset_path = os.path.join(SOURCE_DIR, subset)
        if not os.path.exists(subset_path):
            continue
            
        print(f"Processing {subset}...")
        
        # 遍历类别文件夹
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
                
                # 1. Contrast Levels (RGB)
                for i, level in enumerate(CONTRAST_LEVELS):
                    # 文件夹命名: dataset_variants/contrast_level_1/train/class_name/
                    # level_1 对应 CONTRAST_LEVELS[0]
                    save_dir = os.path.join(OUTPUT_ROOT, f'contrast_level_{i+1}', subset, class_name)
                    save_path = os.path.join(save_dir, img_name)
                    
                    if os.path.exists(save_path):
                        continue

                    # 保持 RGB，不转灰度
                    img_contrast = adjust_contrast(img, level)
                    save_img(img_contrast, save_path)

                # 2. Noise Levels (RGB)
                for i, width in enumerate(NOISE_LEVELS):
                    save_dir = os.path.join(OUTPUT_ROOT, f'noise_level_{i+1}', subset, class_name)
                    save_path = os.path.join(save_dir, img_name)
                    
                    if os.path.exists(save_path):
                        continue

                    # 保持 RGB，不转灰度，不降对比度，只加噪
                    img_noise = apply_uniform_noise(img, -width, width, rng)
                    save_img(img_noise, save_path)
                
                # 3. Eidolon (Optional, keep separate if needed, or remove if not requested)
                # User didn't explicitly ask to remove Eidolon, but focused on Contrast/Noise.
                # I'll comment it out to save time/space unless requested, or keep it in a separate folder.
                # Let's skip it for now to focus on the requested task.
                
    print("All processing done!")

if __name__ == "__main__":
    process_dataset()
