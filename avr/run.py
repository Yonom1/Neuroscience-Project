import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

# =================配置区域=================
BATCH_SIZE = 32
NUM_CLASSES = 2         # 波斯猫 (0) 和 暹罗猫 (1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================1. 数据准备=================
# ImageNet 标准预处理
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
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x=='train'), num_workers=2)
                   for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names

# =================2. 模型获取工厂=================
def get_model(model_name):
    print(f"\nInitializing {model_name}...")
    model = None
    
    if model_name == "alexnet":
        # AlexNet: 2012年的经典
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        # 冻结前面的层（我们只想测它的特征提取能力，不想改变它的‘大脑’）
        for param in model.parameters():
            param.requires_grad = False
        # 修改最后一层分类器 (从1000改到2)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)
        
    elif model_name == "vgg16":
        # VGG: 更深更强
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)
        
    elif model_name == "resnet18":
        # ResNet: 现代结构
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        
    return model.to(DEVICE)

# =================3. 快速训练 & 评估函数=================
def train_model(model_name, train_loader):
    model = get_model(model_name)
    
    # 定义损失函数和优化器（只优化最后一层）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # --- 简单的训练循环 (Fine-tuning) ---
    print(f"Fine-tuning last layer for {model_name} on Original Dataset...")
    model.train()
    for epoch in range(3): # 跑3-5轮通常就够了，因为只需对齐最后一层
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")
    
    return model

def evaluate_model(model, model_name, test_loader, dataset_name):
    # --- 评估并生成混淆矩阵 ---
    print(f"Evaluating {model_name} on {dataset_name}...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() # 0=Persian, 1=Siamese
    
    print(f"\n--- Results for {model_name} on {dataset_name} ---")
    print(f"Confusion Matrix:\n{cm}")
    print(f"TN(Persian correct):{tn}, FP(Persian->Siamese):{fp}")
    print(f"FN(Siamese->Persian):{fn}, TP(Siamese correct):{tp}")
    
    return tn, fp, fn, tp

# =================4. 执行所有实验=================
if __name__ == '__main__':
    # 1. 准备原始数据集用于训练
    original_data_dir = "./dataset"
    if not os.path.exists(original_data_dir):
        print(f"Error: Original dataset not found at {original_data_dir}")
        sys.exit(1)
        
    print(f"\n{'='*20} Loading Original Dataset for Training {'='*20}")
    # 获取原始数据集的 dataloaders
    original_dataloaders, _, class_names = get_dataloaders(original_data_dir)
    print(f"Classes: {class_names}")

    # 2. 定义要测试的所有数据集（包括原始的和变体）
    test_datasets_list = ['dataset'] # 原数据集
    
    # 添加对比度变体 (5个档位)
    for i in range(1, 6):
        test_datasets_list.append(f'dataset_variants/contrast_level_{i}')
        
    # 添加噪声变体 (6个档位)
    for i in range(1, 7):
        test_datasets_list.append(f'dataset_variants/noise_level_{i}')

    # 定义要跑的三个模型
    models_to_test = ['alexnet', 'vgg16', 'resnet18']
    
    # 使用脚本所在目录来存放结果，避免路径错误
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, "experiment_results.csv")
    
    # 初始化日志文件
    with open(log_file, "w", encoding='utf-8') as f:
        f.write("Dataset,Model,TN,FP,FN,TP\n")
        
    # 3. 遍历每个模型：训练一次 -> 测试所有数据集
    for m in models_to_test:
        print(f"\n\n{'#'*30} Processing Model: {m} {'#'*30}")
        
        # A. 在原始数据集上训练模型
        trained_model = train_model(m, original_dataloaders['train'])
        
        # B. 在所有数据集上评估该模型
        for data_path in test_datasets_list:
            data_dir = f"./{data_path}"
            
            if not os.path.exists(data_dir):
                print(f"Warning: {data_dir} not found, skipping.")
                continue
            
            # 获取当前测试集的 dataloader (只需要 test 部分)
            # 注意：这里我们重新调用 get_dataloaders 只是为了获取 test loader
            # 实际上 train/val loader 在这里不会被用到
            current_dataloaders, _, _ = get_dataloaders(data_dir)
            
            tn, fp, fn, tp = evaluate_model(trained_model, m, current_dataloaders['test'], data_path)
            
            # Log to file
            with open(log_file, "a", encoding='utf-8') as f:
                f.write(f"{data_path},{m},{tn},{fp},{fn},{tp}\n")

    print(f"\nAll experiments finished. Results saved to {log_file}")