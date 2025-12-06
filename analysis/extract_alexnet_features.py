import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import glob

INPUT_DIR = r'dataset/noise_samples'
OUTPUT_FILE = r'output/FurtherExploration/alexnet_features.npy'
OUTPUT_META_FILE = r'output/FurtherExploration/alexnet_features_meta.csv'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_alexnet_feature_extractor():
    print("Loading AlexNet...")
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    
    new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    model.classifier = new_classifier
    
    model.to(DEVICE)
    model.eval()
    return model

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img)
    return img_tensor.unsqueeze(0)

def extract_features():
    model = get_alexnet_feature_extractor()
    
    features_list = []
    meta_data = []
    
    
    print(f"Scanning images in {INPUT_DIR}...")
    image_paths = glob.glob(os.path.join(INPUT_DIR, '*', '*.jpg'))
    
    if not image_paths:
        print("No images found!")
        return

    print(f"Found {len(image_paths)} images. Extracting features...")
    
    with torch.no_grad():
        for img_path in image_paths:
            try:
                input_tensor = load_and_preprocess_image(img_path).to(DEVICE)
                
                feature_vector = model(input_tensor)
                
                feature_np = feature_vector.cpu().numpy().flatten()
                
                features_list.append(feature_np)
                
                class_name = os.path.basename(os.path.dirname(img_path))
                file_name = os.path.basename(img_path)
                meta_data.append((class_name, file_name))
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    features_array = np.array(features_list)
    print(f"Features shape: {features_array.shape}")
    
    np.save(OUTPUT_FILE, features_array)
    print(f"Features saved to {OUTPUT_FILE}")
    
    import pandas as pd
    df = pd.DataFrame(meta_data, columns=['Class', 'Filename'])
    df.to_csv(OUTPUT_META_FILE, index=False)
    print(f"Metadata saved to {OUTPUT_META_FILE}")

if __name__ == "__main__":
    extract_features()
