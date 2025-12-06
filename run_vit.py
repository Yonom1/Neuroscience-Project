import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import sys
import gc
from tqdm import tqdm

# 引入之前定义的模型类
# 确保 src/vit_models.py 存在，或者将之前的类定义粘贴到同一个文件中
from utils.vit_models import ViTClassifier, BLIPClassifier, LLaVAClassifier, QwenVLClassifier, load_images_from_folder

# =================配置区域=================
BATCH_SIZE_CNN = 32     # CNN 和 ViT 的批大小
BATCH_SIZE_VLM = 4      # LLaVA / Qwen 的批大小 (显存不够请改为1)
NUM_CLASSES = 2         # 类别数量
CLASS_NAMES = ['Persian', 'Siamese'] # 具体的类名，VLM 需要用到

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================1. 数据准备 (CNN/ViT专用)=================
# ImageNet 标准预处理 - 用于 CNN 和 ViT
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

def get_dataloaders(data_dir, is_variant=False):
    """
    仅用于 CNN 和 ViT 的 DataLoader 获取
    """
    if is_variant:
        # 变体数据集结构通常是 dataset_variants/type_level_x/test/class_name/img.jpg
        # 这里假设传入的 data_dir 已经是 .../test 了，或者根据你的目录结构调整
        # 如果 data_dir 是 'dataset_variants/contrast_level_1'
        test_dir = os.path.join(data_dir, 'test')
        if not os.path.exists(test_dir):
             # 容错：有些变体可能没有 train/test 分割，直接就是图片
             test_dir = data_dir 
        
        image_datasets = {'test': datasets.ImageFolder(test_dir, data_transforms['test'])}
        dataloaders = {'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE_CNN, shuffle=False, num_workers=4)}
        return dataloaders
    
    # 原始数据集
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val', 'test']}
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE_CNN, shuffle=(x=='train'), num_workers=4) 
        for x in ['train', 'val', 'test']}
    return dataloaders

# =================2. 模型获取工厂=================
def get_model(model_name):
    print(f"\nInitializing {model_name}...")
    model = None
    
    # --- 传统 CNN 模型 ---
    if model_name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        for param in model.parameters(): 
            param.requires_grad = False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)
        return model.to(DEVICE)
        
    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for param in model.parameters(): 
            param.requires_grad = False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)
        return model.to(DEVICE)
        
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in model.parameters(): 
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        return model.to(DEVICE)

    # --- 新加入的模型 ---
    elif model_name == "vit_base":
        # ViT 虽然是 Transformer，但通常可以像 CNN 一样微调
        return ViTClassifier(NUM_CLASSES, DEVICE)

    elif model_name == "blip":
        return BLIPClassifier(CLASS_NAMES, DEVICE)

    elif model_name == "llava":
        return LLaVAClassifier(CLASS_NAMES, DEVICE)

    elif model_name == "qwen3-vl":
        return QwenVLClassifier(CLASS_NAMES, DEVICE)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

# =================3. 训练 & 评估逻辑=================

def train_model(model, model_name, train_loader):
    """
    仅适用于 CNN 和 ViT 的微调。
    VLM (LLaVA/Qwen) 通常太大，无法在此脚本中简单微调，故跳过。
    """
    # 判断是否为 VLM (Zero-shot 模型)
    if model_name in ["blip", "llava", "qwen3-vl"]:
        print(f"Skipping training for {model_name} (Zero-shot mode).")
        return model
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 过滤出 requires_grad=True 的参数（ViTClassifier 和 CNN 修改层后的参数）
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=0.001, momentum=0.9)
    
    print(f"Fine-tuning {model_name} on Original Dataset...")
    model.train()
    
    epochs = 3
    for epoch in range(epochs): 
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            # ViTClassifier 的 forward 输入也是 tensor，所以兼容
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")
    
    return model

def evaluate_model(model, model_name, data_dir, is_variant):
    """
    统一评估函数，处理 CNN(Tensor) 和 VLM(PIL Image) 的数据差异
    """
    print(f"Evaluating {model_name} on {data_dir}...")
    model.eval()
    all_preds = []
    all_labels = []
    
    # 判断是否为 VLM
    is_vlm = model_name in ["blip", "llava", "qwen3-vl"]
    
    # 确定测试集路径
    if is_variant:
        test_path = os.path.join(data_dir, 'test')
        if not os.path.exists(test_path): 
            test_path = data_dir
    else:
        test_path = os.path.join(data_dir, 'test')

    with torch.no_grad():
        # --- 分支 A: 多模态大模型 (使用 raw images) ---
        if is_vlm:
            # 使用之前的 load_images_from_folder 读取 PIL 图片
            # 注意：load_images_from_folder 需要 src.vit_models 中的定义
            images, labels, _ = load_images_from_folder(test_path)
            
            # 批量处理防止爆显存
            batch_size = BATCH_SIZE_VLM
            for i in tqdm(range(0, len(images), batch_size), desc="VLM Infer"):
                batch_imgs = images[i : i + batch_size]
                
                # 调用 predict_logits (之前实现的 Multiple Choice 或 Loss 方法)
                logits = model.predict_logits(batch_imgs)
                
                _, preds = torch.max(logits, 1)
                
                all_preds.extend(preds.cpu().numpy())
            
            all_labels.extend(labels)

        # --- 分支 B: 传统 CNN / ViT (使用 DataLoader Tensor) ---
        else:
            # 获取 DataLoader
            dataloaders = get_dataloaders(data_dir, is_variant)
            test_loader = dataloaders['test']
            
            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)
                
                # ViTClassifier 内部做了处理，CNN 也是直接接收 Tensor
                outputs = model(inputs) 
                
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
            
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    # 容错处理：如果数据集只有1个类别，confusion_matrix 可能不是 2x2
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        print(f"Warning: Confusion Matrix shape is {cm.shape}, skipping detail extraction.")
        tn, fp, fn, tp = 0, 0, 0, 0
    
    print(f"Result: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    return tn, fp, fn, tp

# =================4. 执行所有实验=================
if __name__ == '__main__':
    # 1. 准备原始数据集用于训练
    original_data_dir = "./dataset"
    if not os.path.exists(original_data_dir):
        print(f"Error: Original dataset not found at {original_data_dir}")
        sys.exit(1)
        
    print(f"\n{'='*20} Setup {'='*20}")
    # 为了获取 train loader (仅用于 CNN/ViT 训练)
    original_dataloaders = get_dataloaders(original_data_dir, is_variant=False)
    train_loader = original_dataloaders['train']

    # 2. 定义要测试的所有数据集
    test_datasets_list = ['dataset'] # 原数据集
    
    # 你的变体数据集
    # for i in range(1, 6): 
    #     test_datasets_list.append(f'dataset_variants/contrast_level_{i}')
    # for i in range(1, 7): 
    #     test_datasets_list.append(f'dataset_variants/noise_level_{i}')
    # for i in range(1, 5): 
    #     test_datasets_list.append(f'dataset_variants/jigsaw_level_{i}')
    # for i in range(1, 6): 
    #     test_datasets_list.append(f'dataset_variants/eidolon_level_{i}')
    test_datasets_list.append('dataset_variants/noise_level_6')

    # 3. 定义模型列表
    models_to_test = [
        # 'alexnet', 
        # 'vgg16', 
        # 'resnet18', 
        'vit_base',
        'blip',
        'llava',
        'qwen3-vl'
    ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, "experiment_results_full.csv")
    
    # 初始化日志
    if not os.path.exists(log_file):
        with open(log_file, "w", encoding='utf-8') as f:
            f.write("Dataset,Model,TN,FP,FN,TP\n")
        
    # 4. 主循环：模型 -> 训练(可选) -> 遍历所有数据集
    for m in models_to_test:
        print(f"\n\n{'#'*30} Processing Model: {m} {'#'*30}")
        
        # 实例化模型
        try:
            current_model = get_model(m)
        except Exception as e:
            print(f"Error loading model {m}: {e}")
            continue

        # A. 训练 (如果是 CNN/ViT)
        # 如果是 blip/llava/qwen，函数内部会自动跳过
        current_model = train_model(current_model, m, train_loader)
        
        # B. 评估所有数据集
        for data_path in test_datasets_list:
            data_dir = f"./{data_path}"
            if not os.path.exists(data_dir):
                print(f"Skipping {data_dir} (not found)")
                continue
            
            is_variant = 'dataset_variants' in data_path
            
            try:
                tn, fp, fn, tp = evaluate_model(current_model, m, data_dir, is_variant)
                
                # 写入结果
                with open(log_file, "a", encoding='utf-8') as f:
                    f.write(f"{data_path},{m},{tn},{fp},{fn},{tp}\n")
            except Exception as e:
                print(f"Error evaluating {m} on {data_path}: {e}")
                # 遇到错误不要中断整个脚本，继续下一个数据集
                continue

        # C. 清理显存 (非常重要，特别是跑完 LLaVA 后换其他模型)
        del current_model
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nAll experiments finished. Results saved to {log_file}")