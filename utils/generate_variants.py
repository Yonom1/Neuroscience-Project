import os
import sys
import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from PIL import Image
import warnings
import cv2
import random
from tempfile import mkstemp

warnings.filterwarnings("ignore")


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_DIR = os.path.join(PROJECT_ROOT, 'dataset')
DIRS_TO_PROCESS = ['train', 'val', 'test']
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'dataset_variants')

CONTRAST_LEVELS = [0.9, 0.7, 0.5, 0.3, 0.1]

NOISE_LEVELS = [0.05, 0.20, 0.35, 0.5, 0.75, 1.0]

EIDOLON_LEVELS = [
    {'reach': 2.0, 'coherence': 1.0, 'grain': 10.0}, 
    {'reach': 4.0, 'coherence': 1.0, 'grain': 10.0}, 
    {'reach': 8.0, 'coherence': 1.0, 'grain': 10.0}, 
    {'reach': 16.0, 'coherence': 1.0, 'grain': 10.0},
    {'reach': 32.0, 'coherence': 1.0, 'grain': 10.0},
]

JIGSAW_LEVELS = [2, 4, 8, 16]

# eidolon setup
sys.path.append(os.path.abspath('./Eidolon'))

try:
    from eidolon import picture as pic
    from eidolon import helpers as hel
    from eidolon import scalespaces as scl
    EIDOLON_AVAILABLE = True
    print("Eidolon library loaded successfully.")
except ImportError:
    print("Eidolon library not found. Eidolon transformation will be skipped.")
    EIDOLON_AVAILABLE = False


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

def adjust_contrast(image, contrast_level):
    """Return the image scaled to a certain contrast level in [0, 1]."""
    assert(contrast_level >= 0.0), "contrast_level too low."
    assert(contrast_level <= 1.0), "contrast_level too high."
    return (1-contrast_level)/2.0 + image * contrast_level

def grayscale_contrast(image, contrast_level):
    """Convert to grayscale. Adjust contrast."""
    gray = rgb2gray(image)
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
    noise = get_uniform_noise(low, high, nrow, ncol, rng)
    noise = np.stack((noise,)*3, axis=-1)
    image = image + noise
    image = np.where(image < 0, 0, image)
    image = np.where(image > 1, 1, image)
    return image

def jigsaw_scramble(img, grid_n):
    """Apply Jigsaw Permutation."""
    h, w = img.shape[:2]
    bh, bw = h // grid_n, w // grid_n
    target_h = bh * grid_n
    target_w = bw * grid_n
    
    if target_h != h or target_w != w:
        img = cv2.resize(img, (target_w, target_h))
        h, w = target_h, target_w

    blocks = []
    for y in range(0, h, bh):
        for x in range(0, w, bw):
            blocks.append(img[y:y+bh, x:x+bw])
    
    random.shuffle(blocks)
    
    new_img = np.zeros_like(img)
    idx = 0
    for y in range(0, h, bh):
        for x in range(0, w, bw):
            new_img[y:y+bh, x:x+bw] = blocks[idx]
            idx += 1
    return new_img

# eidolon utils
if EIDOLON_AVAILABLE:
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
        res = np.asarray(Image.fromarray(eidolon, 'L')) / 255.0
        return np.stack((res,)*3, axis=-1)

# main logic

def process_dataset():
    rng = np.random.RandomState(seed=42)
    
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
                
                # 1. Contrast Levels
                for i, level in enumerate(CONTRAST_LEVELS):
                    save_dir = os.path.join(OUTPUT_ROOT, f'contrast_level_{i+1}', subset, class_name)
                    save_path = os.path.join(save_dir, img_name)
                    if not os.path.exists(save_path):
                        img_contrast = adjust_contrast(img, level)
                        save_img(img_contrast, save_path)

                # 2. Noise Levels
                for i, width in enumerate(NOISE_LEVELS):
                    save_dir = os.path.join(OUTPUT_ROOT, f'noise_level_{i+1}', subset, class_name)
                    save_path = os.path.join(save_dir, img_name)
                    if not os.path.exists(save_path):
                        img_noise = apply_uniform_noise(img, -width, width, rng)
                        save_img(img_noise, save_path)
                
                # 3. Eidolon Levels
                if EIDOLON_AVAILABLE:
                    for i, params in enumerate(EIDOLON_LEVELS):
                        save_dir = os.path.join(OUTPUT_ROOT, f'eidolon_level_{i+1}', subset, class_name)
                        save_path = os.path.join(save_dir, img_name)
                        if not os.path.exists(save_path):
                            try:
                                img_eidolon = partially_coherent_disarray(
                                    img, 
                                    reach=params['reach'], 
                                    coherence=params['coherence'], 
                                    grain=params['grain']
                                )
                                save_img(img_eidolon, save_path)
                            except Exception as e:
                                print(f"    Error processing Eidolon {img_name} at level {i+1}: {e}")

                # 4. Jigsaw Levels
                for i, grid_n in enumerate(JIGSAW_LEVELS):
                    save_dir = os.path.join(OUTPUT_ROOT, f'jigsaw_level_{i+1}', subset, class_name)
                    save_path = os.path.join(save_dir, img_name)
                    if not os.path.exists(save_path):
                        try:
                            img_jigsaw = jigsaw_scramble(img, grid_n)
                            save_img(img_jigsaw, save_path)
                        except Exception as e:
                            print(f"    Error processing Jigsaw {img_name} at grid {grid_n}: {e}")

    print("All processing done!")

if __name__ == "__main__":
    process_dataset()
