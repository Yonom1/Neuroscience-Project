import torch
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import glob
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'noise_samples')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'FurtherExploration', 'vit_base_features.npy')
OUTPUT_META_FILE = os.path.join(PROJECT_ROOT, 'FurtherExploration', 'vit_base_features_meta.csv')
LOCAL_DIR = "/data1/sht/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_vit_feature_extractor():
    print("Loading ViT-Base (Hugging Face)...")
    
    model_path = LOCAL_DIR + "models/vit-base-patch16-224"
    
    processor = ViTImageProcessor.from_pretrained(model_path)
    
    model = ViTModel.from_pretrained(model_path)
    
    model.to(DEVICE)
    model.eval()
    
    return model, processor

def extract_features():
    model, processor = get_vit_feature_extractor()
    
    features_list = []
    meta_data = [] 

    print(f"Scanning images in {INPUT_DIR}...")
    image_paths = glob.glob(os.path.join(INPUT_DIR, '*', '*.jpg'))
    
    if not image_paths:
        print("No images found!")
        return

    image_paths = sorted(image_paths, reverse=False)

    print(f"Found {len(image_paths)} images. Extracting features...")
    
    with torch.no_grad():
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                inputs = processor(images=img, return_tensors="pt").to(DEVICE)
                outputs = model(**inputs)
                feature_vector = outputs.pooler_output
                feature_np = feature_vector.cpu().numpy().flatten()
                features_list.append(feature_np)
                
                class_name = os.path.basename(os.path.dirname(img_path))
                file_name = os.path.basename(img_path)
                meta_data.append((class_name, file_name))
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    if len(features_list) > 0:
        features_array = np.array(features_list)
        print(f"Features shape: {features_array.shape}")
        
        np.save(OUTPUT_FILE, features_array)
        print(f"Features saved to {OUTPUT_FILE}")
        
        df = pd.DataFrame(meta_data, columns=['Class', 'Filename'])
        df.to_csv(OUTPUT_META_FILE, index=False)
        print(f"Metadata saved to {OUTPUT_META_FILE}")
    else:
        print("No features extracted.")

if __name__ == "__main__":
    extract_features()