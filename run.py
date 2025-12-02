import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import sys
import gc # 引入垃圾回收
from tqdm import tqdm # 引入进度条

# 环境变量配置
os.environ['CUDA_VISIBLE_DEVICES'] = '4' 
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from src.vit_models import ViTClassifier, BLIPClassifier, LLaVAClassifier, QwenVLClassifier, load_images_from_folder
from torchvision.transforms import ToPILImage
to_pil = ToPILImage()

# =================配置区域=================
BATCH_SIZE = 32
NUM_CLASSES = 2
CLASS_NAMES = ['Persian', 'Siamese']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================1. 数据准备=================
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def get_dataloaders(data_dir):
    # 增加健壮性检查
    if not os.path.exists(os.path.join(data_dir, 'train')):
        print(f"Error: {data_dir} structure incorrect (missing train/val/test).")
        return None, None, None
        
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x=='train'), num_workers=4)
                   for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names

# =================2. 模型获取工厂=================
def get_model(model_name):
    print(f"Initializing {model_name}...", end=" ", flush=True)
    model = None
    
    if model_name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        for param in model.parameters(): param.requires_grad = False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)
        
    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for param in model.parameters(): param.requires_grad = False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)
        
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in model.parameters(): param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    elif model_name == "vit_base":
        model = ViTClassifier(NUM_CLASSES, DEVICE)

    elif model_name == "blip":
        model = BLIPClassifier(CLASS_NAMES, DEVICE)

    elif model_name == "llava":
        model = LLaVAClassifier(CLASS_NAMES, DEVICE)

    elif model_name == "qwen3-vl":
        model = QwenVLClassifier(CLASS_NAMES, DEVICE)
    
    else:
        raise ValueError("Unknown model name: " + model_name)
    
    # 确保 CNN/ViT 在设备上 (VLM 通常在 init 里已经 to(device) 或 device_map='auto')
    if model_name not in ["blip", "llava", "qwen3-vl"]:
        model = model.to(DEVICE)
        
    print("Done.")
    return model

# =================3. 核心功能函数=================

def run_evaluation(model, dataloaders, is_vlm, model_name):
    """
    统一的评估函数，集成了显存优化逻辑
    """
    model.eval()
    all_preds = []
    all_labels = []

    print(f"Evaluating {model_name}...", end=" ")
    
    with torch.no_grad():
        if is_vlm:
            # === VLM 评估逻辑 (集成 Batch 处理防止 OOM) ===
            folder_path = os.path.join(dataloaders['test'].dataset.root)
            images, labels, _ = load_images_from_folder(folder_path)
            
            # 使用较小的 Batch Size 进行推理
            eval_batch_size = 100 # 如果是 Qwen/LLaVA 显存不够，请改为 1 或 2
            
            # 进度条
            for i in tqdm(range(0, len(images), eval_batch_size), desc="VLM Inference", leave=False):
                batch_imgs = images[i : i + eval_batch_size]
                
                # 调用 predict_logits
                logits = model.predict_logits(batch_imgs)
                
                # 立即转移到 CPU
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
            
            all_labels.extend(labels)

        else:
            # === CNN/ViT 评估逻辑 ===
            for inputs, labels in dataloaders['test']:
                inputs = inputs.to(DEVICE)
                logits = model(inputs)
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

    # 计算指标
    cm = confusion_matrix(all_labels, all_preds)
    # 处理可能的 shape 不匹配 (防止只有一个类别时 ravel 报错)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # 简单的 fallback，或者根据实际 label 补全
        tn, fp, fn, tp = 0, 0, 0, 0 
        print(f"Warning: Confusion matrix shape is {cm.shape}, expected (2,2).")

    print("Done.")
    return tn, fp, fn, tp


def run_training(model, dataloaders):
    """
    微调训练循环
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=0.001, momentum=0.9)
    
    model.train()
    epochs = 3
    print(f"Fine-tuning (Epochs: {epochs})...", end=" ")
    
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    print("Done.")

# =================4. 主执行逻辑=================
if __name__ == '__main__':
    # 定义数据集
    datasets_list = ['dataset'] 
    for i in range(1, 6):
        datasets_list.append(f'dataset_variants/contrast_level_{i}')
    for i in range(1, 7):
        datasets_list.append(f'dataset_variants/noise_level_{i}')

    # 定义模型
    models_to_test = [
        # 'resnet18', 
        'vit_base', 
        # 'blip', 
        # 'llava', 
        # 'qwen3-vl'
    ]

    log_file = "experiment_results_optimized.csv"
    
    # 写入表头 (如果文件不存在)
    if not os.path.exists(log_file):
        with open(log_file, "w", encoding='utf-8') as f:
            f.write("Dataset,Model,TN,FP,FN,TP\n")

    # === 外层循环改为模型 ===
    for model_name in models_to_test:
        print(f"\n{'='*40}")
        print(f"Starting experiments for model: {model_name}")
        print(f"{'='*40}")
        
        is_vlm = model_name in ["blip", "llava", "qwen3-vl"]
        
        # --- 策略 A: 对于 VLM (Zero-Shot)，加载一次，跑遍所有数据集 ---
        if is_vlm:
            # 1. 加载模型 (耗时操作)
            try:
                model = get_model(model_name)
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                continue

            # 2. 遍历所有数据集
            for data_path in datasets_list:
                data_dir = f"./{data_path}"
                print(f"\n[Dataset: {data_path}]")
                
                dataloaders, _, _ = get_dataloaders(data_dir)
                if dataloaders is None: 
                    continue

                # 3. 直接评估 (无需训练)
                tn, fp, fn, tp = run_evaluation(model, dataloaders, is_vlm=True, model_name=model_name)
                
                # 4. 记录日志
                with open(log_file, "a", encoding='utf-8') as f:
                    f.write(f"{data_path},{model_name},{tn},{fp},{fn},{tp}\n")
            
            # 5. 模型跑完所有数据后，彻底清理显存
            print(f"Finished {model_name}, cleaning up GPU memory...")
            del model
            torch.cuda.empty_cache()
            gc.collect()

        # --- 策略 B: 对于可训练模型 (CNN/ViT)，必须每个数据集都重新加载/微调 ---
        else:
            for data_path in datasets_list:
                data_dir = f"./{data_path}"
                print(f"\n[Dataset: {data_path}]")
                
                dataloaders, _, _ = get_dataloaders(data_dir)
                if dataloaders is None: continue

                # 1. 每次都重新加载模型 (保证初始权重一致，没有数据泄露)
                model = get_model(model_name)
                
                # 2. 训练
                run_training(model, dataloaders)
                
                # 3. 评估
                tn, fp, fn, tp = run_evaluation(model, dataloaders, is_vlm=False, model_name=model_name)
                
                # 4. 记录
                with open(log_file, "a", encoding='utf-8') as f:
                    f.write(f"{data_path},{model_name},{tn},{fp},{fn},{tp}\n")
                
                # 5. 清理当前数据集的模型
                del model
                torch.cuda.empty_cache()

    print(f"\nAll experiments finished. Results saved to {log_file}")