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

# config area
BATCH_SIZE = 32
NUM_CLASSES = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet standard preprocessing
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
    if is_variant:
        image_datasets = {'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])}
        dataloaders = {'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=2)}
        return dataloaders, None, None
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x=='train'), num_workers=2)
                   for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names

def get_model(model_name):
    print(f"\nInitializing {model_name}...")
    model = None
    
    if model_name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)
        
    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)
        
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        
    return model.to(DEVICE)

def train_model(model_name, train_loader):
    model = get_model(model_name)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    print(f"Fine-tuning last layer for {model_name} on Original Dataset...")
    model.train()
    for epoch in range(3):
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
            
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n--- Results for {model_name} on {dataset_name} ---")
    print(f"Confusion Matrix:\n{cm}")
    print(f"TN(Persian correct):{tn}, FP(Persian->Siamese):{fp}")
    print(f"FN(Siamese->Persian):{fn}, TP(Siamese correct):{tp}")
    
    return tn, fp, fn, tp

if __name__ == '__main__':
    original_data_dir = "./dataset"
    if not os.path.exists(original_data_dir):
        print(f"Error: Original dataset not found at {original_data_dir}")
        sys.exit(1)
        
    print(f"\n{'='*20} Loading Original Dataset for Training {'='*20}")
    original_dataloaders, _, class_names = get_dataloaders(original_data_dir)
    print(f"Classes: {class_names}")

    test_datasets_list = ['dataset']
    
    for i in range(1, 6):
        test_datasets_list.append(f'dataset_variants/contrast_level_{i}')
        
    for i in range(1, 7):
        test_datasets_list.append(f'dataset_variants/noise_level_{i}')

    for i in range (1, 5):
        test_datasets_list.append(f'dataset_variants/jigsaw_level_{i}')

    for i in range(1, 6):
        test_datasets_list.append(f'dataset_variants/eidolon_level_{i}')

    models_to_test = ['alexnet', 'vgg16', 'resnet18']
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, "experiment_results2.csv")
    
    with open(log_file, "w", encoding='utf-8') as f:
        f.write("Dataset,Model,TN,FP,FN,TP\n")
        
    for m in models_to_test:
        print(f"\n\n{'#'*30} Processing Model: {m} {'#'*30}")
        
        trained_model = train_model(m, original_dataloaders['train'])
        
        for data_path in test_datasets_list:
            data_dir = f"./{data_path}"
            
            if not os.path.exists(data_dir):
                print(f"Warning: {data_dir} not found, skipping.")
                continue
            
            is_variant = 'dataset_variants' in data_path
            current_dataloaders, _, _ = get_dataloaders(data_dir, is_variant=is_variant)
            
            tn, fp, fn, tp = evaluate_model(trained_model, m, current_dataloaders['test'], data_path)
            
            with open(log_file, "a", encoding='utf-8') as f:
                f.write(f"{data_path},{m},{tn},{fp},{fn},{tp}\n")

    print(f"\nAll experiments finished. Results saved to {log_file}")