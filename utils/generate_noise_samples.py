import os
import sys
import numpy as np
from skimage.io import imread, imsave
from skimage import img_as_ubyte
import warnings

warnings.filterwarnings("ignore")

# config area
# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGET_IMAGES = [
    os.path.join(PROJECT_ROOT, 'dataset', 'test', 'persian_cat', '000001.jpg'),
    os.path.join(PROJECT_ROOT, 'dataset', 'test', 'siamese_cat', '000001.jpg')
]

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'noise_samples')

NOISE_LEVEL = 0.75
NUM_SAMPLES = 50


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

def get_uniform_noise(low, high, nrow, ncol, rng=None):
    if rng is None:
        return np.random.uniform(low=low, high=high, size=(nrow, ncol))
    else:
        return rng.uniform(low=low, high=high, size=(nrow, ncol))

def apply_uniform_noise(image, width, rng=None):
    nrow = image.shape[0]
    ncol = image.shape[1]

    low = -width
    high = width
    
    noise = get_uniform_noise(low, high, nrow, ncol, rng)
    noise = np.stack((noise,)*3, axis=-1)
    
    image = image + noise
    
    image = np.where(image < 0, 0, image)
    image = np.where(image > 1, 1, image)
    
    return image


def generate_noise_samples():

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

        class_name = os.path.basename(os.path.dirname(img_path))
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        for i in range(NUM_SAMPLES):
            noisy_img = apply_uniform_noise(img, NOISE_LEVEL, rng)
            
            save_name = f"{img_name}_noise_{NOISE_LEVEL}_sample_{i+1:02d}.jpg"
            save_path = os.path.join(OUTPUT_DIR, class_name, save_name)
            
            save_img(noisy_img, save_path)
            print(f"  Saved {save_name}")

    print("All samples generated!")

if __name__ == "__main__":
    generate_noise_samples()
