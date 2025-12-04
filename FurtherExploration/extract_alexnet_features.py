import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import glob

# ================= 配置 =================
# 图片所在目录
INPUT_DIR = r'E:\南京大学\课程\大三上学期\NS\project\FurtherExploration\noise_samples'
# 输出文件
OUTPUT_FILE = r'E:\南京大学\课程\大三上学期\NS\project\FurtherExploration\alexnet_features.npy'
OUTPUT_META_FILE = r'E:\南京大学\课程\大三上学期\NS\project\FurtherExploration\alexnet_features_meta.csv'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 模型准备 =================
def get_alexnet_feature_extractor():
    print("Loading AlexNet...")
    # 加载预训练的 AlexNet
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    
    # 我们需要“倒数第二层”的输出，即分类头之前的向量
    # AlexNet 的 classifier 结构如下：
    # (0): Dropout
    # (1): Linear(9216 -> 4096)
    # (2): ReLU
    # (3): Dropout
    # (4): Linear(4096 -> 4096)
    # (5): ReLU  <-- 我们通常取这里的输出作为特征
    # (6): Linear(4096 -> 1000) <-- 最后一层
    
    # 方法：移除最后一层 Linear
    # 创建一个新的 classifier，包含原 classifier 的前 6 层 (索引 0 到 5)
    new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    model.classifier = new_classifier
    
    model.to(DEVICE)
    model.eval()
    return model

# ================= 数据预处理 =================
# 使用与训练时相同的预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img)
    return img_tensor.unsqueeze(0) # Add batch dimension

# ================= 主逻辑 =================
def extract_features():
    model = get_alexnet_feature_extractor()
    
    features_list = []
    meta_data = [] # 记录文件名和类别
    
    # 遍历 noise_samples 下的所有子文件夹
    # 结构: noise_samples/class_name/image_name.jpg
    
    print(f"Scanning images in {INPUT_DIR}...")
    image_paths = glob.glob(os.path.join(INPUT_DIR, '*', '*.jpg'))
    
    if not image_paths:
        print("No images found!")
        return

    print(f"Found {len(image_paths)} images. Extracting features...")
    
    with torch.no_grad():
        for img_path in image_paths:
            try:
                # 准备数据
                input_tensor = load_and_preprocess_image(img_path).to(DEVICE)
                
                # 前向传播
                # 此时 model(input) 返回的就是 classifier[5] 的输出 (4096维)
                feature_vector = model(input_tensor)
                
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
    features_array = np.array(features_list)
    print(f"Features shape: {features_array.shape}")
    
    np.save(OUTPUT_FILE, features_array)
    print(f"Features saved to {OUTPUT_FILE}")
    
    # 保存元数据 CSV
    import pandas as pd
    df = pd.DataFrame(meta_data, columns=['Class', 'Filename'])
    df.to_csv(OUTPUT_META_FILE, index=False)
    print(f"Metadata saved to {OUTPUT_META_FILE}")

if __name__ == "__main__":
    extract_features()
