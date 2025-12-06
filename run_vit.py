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


from utils.vit_models import ViTClassifier, BLIPClassifier, LLaVAClassifier, QwenVLClassifier, load_images_from_folder

BATCH_SIZE_CNN = 32     
BATCH_SIZE_VLM = 4     
NUM_CLASSES = 2         
CLASS_NAMES = ['Persian', 'Siamese']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        test_dir = os.path.join(data_dir, 'test')
        if not os.path.exists(test_dir):
             test_dir = data_dir 
        
        image_datasets = {'test': datasets.ImageFolder(test_dir, data_transforms['test'])}
        dataloaders = {'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE_CNN, shuffle=False, num_workers=4)}
        return dataloaders
    
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val', 'test']}
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE_CNN, shuffle=(x=='train'), num_workers=4) 
        for x in ['train', 'val', 'test']}
    return dataloaders

def get_model(model_name):
    print(f"\nInitializing {model_name}...")
    model = None
    
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

    elif model_name == "vit_base":
        return ViTClassifier(NUM_CLASSES, DEVICE)

    elif model_name == "blip":
        return BLIPClassifier(CLASS_NAMES, DEVICE)

    elif model_name == "llava":
        return LLaVAClassifier(CLASS_NAMES, DEVICE)

    elif model_name == "qwen3-vl":
        return QwenVLClassifier(CLASS_NAMES, DEVICE)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_model(model, model_name, train_loader):
    """
    仅适用于 CNN 和 ViT 的微调。
    VLM (LLaVA/Qwen) 通常太大，无法在此脚本中简单微调，故跳过。
    """
    if model_name in ["blip", "llava", "qwen3-vl"]:
        print(f"Skipping training for {model_name} (Zero-shot mode).")
        return model
    
    criterion = nn.CrossEntropyLoss()
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
    
    is_vlm = model_name in ["blip", "llava", "qwen3-vl"]
    
    if is_variant:
        test_path = os.path.join(data_dir, 'test')
        if not os.path.exists(test_path): 
            test_path = data_dir
    else:
        test_path = os.path.join(data_dir, 'test')

    with torch.no_grad():
        if is_vlm:

            images, labels, _ = load_images_from_folder(test_path)
            
            batch_size = BATCH_SIZE_VLM
            for i in tqdm(range(0, len(images), batch_size), desc="VLM Infer"):
                batch_imgs = images[i : i + batch_size]
                
                logits = model.predict_logits(batch_imgs)
                
                _, preds = torch.max(logits, 1)
                
                all_preds.extend(preds.cpu().numpy())
            
            all_labels.extend(labels)

        else:
            dataloaders = get_dataloaders(data_dir, is_variant)
            test_loader = dataloaders['test']
            
            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)
                
                outputs = model(inputs) 
                
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        print(f"Warning: Confusion Matrix shape is {cm.shape}, skipping detail extraction.")
        tn, fp, fn, tp = 0, 0, 0, 0
    
    print(f"Result: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    return tn, fp, fn, tp

if __name__ == '__main__':
    original_data_dir = "./dataset"
    if not os.path.exists(original_data_dir):
        print(f"Error: Original dataset not found at {original_data_dir}")
        sys.exit(1)
        
    print(f"\n{'='*20} Setup {'='*20}")
    original_dataloaders = get_dataloaders(original_data_dir, is_variant=False)
    train_loader = original_dataloaders['train']

    test_datasets_list = ['dataset'] 
    
    for i in range(1, 6): 
        test_datasets_list.append(f'dataset_variants/contrast_level_{i}')
    for i in range(1, 7): 
        test_datasets_list.append(f'dataset_variants/noise_level_{i}')
    for i in range(1, 5): 
        test_datasets_list.append(f'dataset_variants/jigsaw_level_{i}')
    for i in range(1, 6): 
        test_datasets_list.append(f'dataset_variants/eidolon_level_{i}')
    test_datasets_list.append('dataset_variants/noise_level_6')

    models_to_test = [
        'alexnet', 
        'vgg16', 
        'resnet18', 
        'vit_base',
        'blip',
        'llava',
        'qwen3-vl'
    ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, "experiment_results_full.csv")
    
    if not os.path.exists(log_file):
        with open(log_file, "w", encoding='utf-8') as f:
            f.write("Dataset,Model,TN,FP,FN,TP\n")
        
    for m in models_to_test:
        print(f"\n\n{'#'*30} Processing Model: {m} {'#'*30}")
        
        try:
            current_model = get_model(m)
        except Exception as e:
            print(f"Error loading model {m}: {e}")
            continue

        current_model = train_model(current_model, m, train_loader)
        
        for data_path in test_datasets_list:
            data_dir = f"./{data_path}"
            if not os.path.exists(data_dir):
                print(f"Skipping {data_dir} (not found)")
                continue
            
            is_variant = 'dataset_variants' in data_path
            
            try:
                tn, fp, fn, tp = evaluate_model(current_model, m, data_dir, is_variant)
                
                with open(log_file, "a", encoding='utf-8') as f:
                    f.write(f"{data_path},{m},{tn},{fp},{fn},{tp}\n")
            except Exception as e:
                print(f"Error evaluating {m} on {data_path}: {e}")
                continue

        del current_model
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nAll experiments finished. Results saved to {log_file}")