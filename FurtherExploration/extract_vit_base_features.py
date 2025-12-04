import torch
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import glob
import pandas as pd

INPUT_DIR = 'dataset_variants/noise_samples'
OUTPUT_FILE = 'FurtherExploration/vit_base_features.npy'
OUTPUT_META_FILE = 'FurtherExploration/vit_base_features_meta.csv'
LOCAL_DIR = "/data1/sht/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 模型准备 =================
def get_vit_feature_extractor():
    print("Loading ViT-Base (Hugging Face)...")
    
    model_path = LOCAL_DIR + "models/vit-base-patch16-224"
    
    # 1. 加载 Processor (负责 Resize, Normalize 等预处理)
    # 相当于 torchvision 中的 weights.transforms()
    processor = ViTImageProcessor.from_pretrained(model_path)
    
    # 2. 加载模型
    # ViTModel 是纯粹的 Backbone，输出就是特征，没有分类层
    model = ViTModel.from_pretrained(model_path)
    
    model.to(DEVICE)
    model.eval()
    
    return model, processor

# ================= 主逻辑 =================
def extract_features():
    # 获取模型和对应的预处理函数
    model, processor = get_vit_feature_extractor()
    
    features_list = []
    meta_data = [] # 记录文件名和类别
    
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
                # ================= 数据预处理 =================
                # 读取图片
                img = Image.open(img_path).convert('RGB')
                
                # 使用 HF 的 processor 进行预处理
                # return_tensors="pt" 返回 pytorch tensor
                inputs = processor(images=img, return_tensors="pt").to(DEVICE)
                
                # ================= 前向传播 =================
                # HuggingFace 模型的输入通常是字典形式 (pixel_values)
                outputs = model(**inputs)
                
                # 获取特征：
                # outputs.pooler_output: [Batch, 768] -> [CLS] token 经过线性层+Tanh后的特征
                # outputs.last_hidden_state[:, 0, :]: [Batch, 768] -> 原始的 [CLS] token
                # 通常做特征提取使用 pooler_output 即可
                feature_vector = outputs.pooler_output
                
                # 转为 numpy
                feature_np = feature_vector.cpu().numpy().flatten()
                
                features_list.append(feature_np)
                
                # 记录元数据
                class_name = os.path.basename(os.path.dirname(img_path))
                file_name = os.path.basename(img_path)
                meta_data.append((class_name, file_name))
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    # 保存结果
    if len(features_list) > 0:
        features_array = np.array(features_list)
        print(f"Features shape: {features_array.shape}") # 预期应该是 (N, 768)
        
        np.save(OUTPUT_FILE, features_array)
        print(f"Features saved to {OUTPUT_FILE}")
        
        # 保存元数据 CSV
        df = pd.DataFrame(meta_data, columns=['Class', 'Filename'])
        df.to_csv(OUTPUT_META_FILE, index=False)
        print(f"Metadata saved to {OUTPUT_META_FILE}")
    else:
        print("No features extracted.")

if __name__ == "__main__":
    extract_features()