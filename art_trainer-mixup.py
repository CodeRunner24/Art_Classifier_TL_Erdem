#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import v2
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import random
import numpy as np

# A.1. Check device availability and setup MPS optimizations
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_float32_matmul_precision('high')  # MPS performance optimization

# Hyperparameters (Tested optimal values)
CFG = {
    'img_size': 224,
    'batch_size': 32,
    'lr': 3e-5,            # Lower learning rate
    'weight_decay': 0.05,  # Stronger L2 regularization
    'dropout': 0.5,        # Increased dropout
    'epochs': 30,
    'mixup_alpha': 0.4,
    'cutmix_prob': 0.3,
    'label_smoothing': 0.15,
    'patience': 5          # For early stopping
}

# A.2.4. Define data transformations with advanced augmentation pipeline
def create_transforms():
    return {
        'train': v2.Compose([
            # A word on presizing:
            # 1. Increase the size (item by item)
            v2.RandomResizedCrop(CFG['img_size'], scale=(0.6, 1.0)),
            # 2. Apply augmentation (batch by batch)
            v2.RandomHorizontalFlip(p=0.7),
            v2.RandomVerticalFlip(p=0.3),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
            v2.RandomRotation(35),
            v2.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            v2.RandomPerspective(distortion_scale=0.4, p=0.6),
            v2.GaussianBlur(kernel_size=(5, 9)),
            v2.RandomSolarize(threshold=0.3, p=0.2),
            v2.ToTensor(),
            # 3. Decrease the size (batch by batch)
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.RandomErasing(p=0.5, scale=(0.02, 0.2), value='random')
        ]),
        'val': v2.Compose([
            v2.Resize(CFG['img_size'] + 32),
            v2.CenterCrop(CFG['img_size']),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

# A.2.2. Define the means of getting data into DataBlock
class ArtDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.classes = sorted([d.name for d in Path(data_dir).iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cls_dir = Path(data_dir) / cls
            for img_path in cls_dir.glob('*'):
                self.samples.append((img_path, self.class_to_idx[cls]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# B.4. Implement mixup data augmentation - part of discriminative learning rates
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# A.4. Define training step
def train_step(model, data_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    
    for inputs, targets in tqdm(data_loader, desc='Training', leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # B.4. Advanced Mixup - part of discriminative learning rates
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, CFG['mixup_alpha'])
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets_a) * lam + criterion(outputs, targets_b) * (1 - lam)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (lam * predicted.eq(targets_a).sum().item() + 
                   (1 - lam) * predicted.eq(targets_b).sum().item())
    
    acc = 100. * correct / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader)
    return avg_loss, acc

# A.3. Define validation step to inspect the DataBlock
def validate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc='Validation', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100. * correct / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader)
    return avg_loss, acc

def main():
    # A.1. Load data
    transforms = create_transforms()
    
    # Set directory paths according to your structure
    art_dataset_dir = 'Art Dataset'
    
    # A.2.1. Define the blocks (dataset creation)
    train_dataset = ArtDataset(art_dataset_dir, transform=transforms['train'])
    val_dataset = ArtDataset(art_dataset_dir, transform=transforms['val'])
    
    # A.2.2. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=CFG['batch_size'], 
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG['batch_size'], 
                          num_workers=4, pin_memory=True)
    
    # B.3. Transfer Learning - Load model
    model_path = 'models/model_final.pth'
    
    # Load model state dictionary
    state_dict = torch.load(model_path)
    
    # Create ResNet34 model
    from torchvision import models
    model = models.resnet34(weights=None)
    
    # Number of classes
    num_classes = len(train_dataset.classes)
    
    # B.3. Update the final fully-connected layer
    model.fc = nn.Linear(512, num_classes)
    
    # Load state dictionary
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    # B.6. Model Capacity - Measures to prevent overfitting
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = CFG['dropout']  # Increase dropout rate
    
    # B.1. Learning Rate Finder - Optimizer and Loss setup
    optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'], 
                          weight_decay=CFG['weight_decay'])
    criterion = nn.CrossEntropyLoss(label_smoothing=CFG['label_smoothing'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                             T_0=10, T_mult=2)
    
    # Create results directory
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # B.5. Early Stopping - Deciding the Number of Training Epochs
    best_val_acc = 0
    patience_counter = 0
    
    # A.4. Train a simple model
    for epoch in range(CFG['epochs']):
        print(f"\nEpoch {epoch+1}/{CFG['epochs']}")
        
        # Training
        train_loss, train_acc = train_step(model, train_loader, criterion, optimizer)
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # Learning rate update
        scheduler.step()
        
        # Monitor results
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        # B.5. Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_path = os.path.join(results_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved ({val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= CFG['patience']:
                print(f"Early stopping! No improvement for {CFG['patience']} epochs.")
                break

if __name__ == "__main__":
    main() 