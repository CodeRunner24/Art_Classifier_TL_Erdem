#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm.auto import tqdm
from pathlib import Path
from collections import Counter
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models import resnet34, ResNet34_Weights

# A.1. Enable CPU fallback for MPS device
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Enable MPS optimizations for PyTorch 2.2+
if hasattr(torch.backends.mps, 'enable_workflow_compiling'):
    print("Enabling MPS workflow compiling...")
    torch.backends.mps.enable_workflow_compiling = True

# A.1. Check Metal 3 / MPS support
def setup_device():
    """Checks Metal 3 / MPS support and returns appropriate device"""
    print("PyTorch version:", torch.__version__)
    
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Metal Performance Shaders (MPS) available.")
        print("PYTORCH_ENABLE_MPS_FALLBACK=1 set - CPU will be used for unsupported operations.")
        device = torch.device("mps")
        
        # Force GPU usage
        dummy_tensor = torch.ones(1, device=device)
        result = dummy_tensor + 1
        is_mps_working = (result.device.type == 'mps')
        
        if is_mps_working:
            print(f"MPS successfully tested: {result}")
            print(f"Training device: {device}")
            return device
        else:
            print("MPS is available but simple operation failed, using CPU.")
            return torch.device("cpu")
    else:
        print("MPS not available, using CPU.")
        device = torch.device("cpu")
        print(f"Training device: {device}")
        return device

# A.1.1. Dataset analysis
def analyze_dataset(data_path):
    """Analyzes the dataset and calculates the number of samples per class"""
    data_path = Path(data_path)
    classes = [d.name for d in data_path.iterdir() if d.is_dir()]
    class_counts = {}
    
    # Calculate the number of samples in each class
    for cls in tqdm(classes, desc="Analyzing classes"):
        class_path = data_path / cls
        class_counts[cls] = len(list(class_path.glob('*.jpg')))
    
    # Display results
    df = pd.DataFrame({'Class': list(class_counts.keys()), 
                       'Number of Samples': list(class_counts.values())})
    df = df.sort_values('Number of Samples', ascending=False).reset_index(drop=True)
    
    # Calculate statistics
    total_samples = df['Number of Samples'].sum()
    mean_samples = df['Number of Samples'].mean()
    min_samples = df['Number of Samples'].min()
    max_samples = df['Number of Samples'].max()
    
    print(f"Total number of samples: {total_samples}")
    print(f"Average number of samples: {mean_samples:.1f}")
    print(f"Minimum number of samples: {min_samples} ({df.iloc[-1]['Class']})")
    print(f"Maximum number of samples: {max_samples} ({df.iloc[0]['Class']})")
    
    # Visualize class distribution
    plt.figure(figsize=(14, 8))
    plt.bar(df['Class'], df['Number of Samples'])
    plt.xticks(rotation=90)
    plt.title('Art Styles - Sample Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    plt.savefig('results/class_distribution.png')
    plt.close()
    
    return df, classes

# A.2.2. Custom dataset class - Performs data augmentation on CPU
class ArtStyleDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None, train=True, valid_pct=0.2, seed=42):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        
        # Get all images and labels
        all_imgs = []
        class_names = [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(class_names))}
        
        # Collect images and labels for each class
        for cls_name in class_names:
            cls_path = self.root_dir / cls_name
            cls_idx = self.class_to_idx[cls_name]
            for img_path in cls_path.glob('*.jpg'):
                all_imgs.append((str(img_path), cls_idx))
                
        # Shuffle data
        random.seed(seed)
        random.shuffle(all_imgs)
        
        # Split into training and validation sets
        n_valid = int(len(all_imgs) * valid_pct)
        if train:
            self.imgs = all_imgs[n_valid:]
        else:
            self.imgs = all_imgs[:n_valid]
        
        self.classes = sorted(class_names)
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        if self.target_transform:
            label = self.target_transform(label)
            
        return img, label

# A.2. Creating DataLoaders using PyTorch native structures
def create_dataloaders(data_path, batch_size=32, img_size=224, augment=True, 
                       balance_method='weighted', valid_pct=0.2, seed=42):
    """Creates PyTorch DataLoaders"""
    
    # A.2.4. Define data transformations
    # Transformations to run on CPU
    if augment:
        # A word on presizing:
        # 1. Increase the size (item by item) - done by RandomResizedCrop
        # 2. Apply augmentation (batch by batch) - done by various transforms
        # 3. Decrease the size (batch by batch) - handled by normalization
        # 4. Presizing avoids artifacts when applying augmentations (e.g., rotation)
        train_transforms = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),  # Increase size item by item
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),  # Apply augmentation batch by batch
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Decrease size batch by batch
        ])
    else:
        train_transforms = T.Compose([
            T.Resize(int(img_size*1.14)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    valid_transforms = T.Compose([
        T.Resize(int(img_size*1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # A.2.1. Define the blocks (dataset creation)
    train_dataset = ArtStyleDataset(data_path, transform=train_transforms, train=True, valid_pct=valid_pct, seed=seed)
    valid_dataset = ArtStyleDataset(data_path, transform=valid_transforms, train=False, valid_pct=valid_pct, seed=seed)
    
    # A.2.2. Define the means of getting data into DataBlock
    # Calculate weights for weighted sampling
    if balance_method == 'weighted' and train_dataset:
        # Count classes
        class_counts = Counter([label for _, label in train_dataset.imgs])
        total = sum(class_counts.values())
        
        # Calculate weights (classes with fewer examples will get higher weights)
        weights = [total / class_counts[train_dataset.imgs[i][1]] for i in range(len(train_dataset))]
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    class_names = train_dataset.classes
    
    # Display data loader summary
    print(f"Training dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(valid_dataset)} images")
    print(f"Classes: {len(class_names)}")
    
    # Return the data loaders
    return train_loader, valid_loader, class_names

# PyTorch native training loop
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_times = []
    
    # Show progress with tqdm
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    # Monitor MPS memory usage
    if device.type == 'mps':
        print(f"MPS memory usage (start): {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")
    
    start_time = time.time()
    for inputs, labels in progress_bar:
        batch_start = time.time()
        
        # Move data to device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Verify training device
        if total == 0:
            print(f"Training tensor device: {inputs.device}, Model device: {next(model.parameters()).device}")
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward propagation
        loss.backward()
        optimizer.step()
        
        # Measure processing time
        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_times.append(batch_time)
        
        # Update statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
    
    # Calculate final statistics
    avg_loss = running_loss / len(dataloader.dataset)
    avg_acc = 100 * correct / total
    avg_time = sum(batch_times) / len(batch_times)
    total_time = time.time() - start_time
    
    # Monitoring memory usage
    if device.type == 'mps':
        print(f"MPS memory usage (end): {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")
        
    # Print statistics
    print(f"Training - Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%, Time: {total_time:.1f}s, Avg batch: {avg_time:.3f}s")
    
    return avg_loss, avg_acc

# A.3. Inspect the DataBlock via dataloader
def validate_epoch(model, dataloader, criterion, device):
    # Set model to evaluation mode
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Disable gradient calculation
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        
        for inputs, labels in progress_bar:
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
    
    # Calculate final statistics
    avg_loss = running_loss / len(dataloader.dataset)
    avg_acc = 100 * correct / total
    
    # Print statistics
    print(f"Validation - Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%")
    
    return avg_loss, avg_acc

# A.4. Train a simple model
def train_model(train_loader, valid_loader, class_names, device, 
                model_name="resnet34", lr=1e-3, epochs=10, 
                freeze_epochs=3, unfreeze_epochs=7):
    """Trains a model using transfer learning with discriminative learning rates"""
    print(f"\nTraining {model_name} model for {epochs} epochs (freeze: {freeze_epochs}, unfreeze: {unfreeze_epochs})")
    
    # B.3. Transfer Learning setup
    # Create ResNet34 model with pretrained weights
    if model_name == "resnet34":
        model = resnet34(weights=ResNet34_Weights.DEFAULT)
        
        # Replace the final layer with a new one for our classes
        num_classes = len(class_names)
        model.fc = nn.Linear(512, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Move model to device
    model = model.to(device)
    
    # B.3. Freeze all weights except the final layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # Set up loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training history for plotting
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training in two phases: first frozen, then unfrozen
    total_start_time = time.time()
    
    # Phase 1: Train with frozen layers
    if freeze_epochs > 0:
        print("\n=== Phase 1: Training with frozen feature extractor ===")
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
        
        for epoch in range(freeze_epochs):
            print(f"\nEpoch {epoch+1}/{freeze_epochs}")
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate_epoch(model, valid_loader, criterion, device)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
    
    # Phase 2: Unfreeze and train with discriminative learning rates
    if unfreeze_epochs > 0:
        print("\n=== Phase 2: Fine-tuning with discriminative learning rates ===")
        
        # B.3. Unfreeze all weights for fine-tuning
        for param in model.parameters():
            param.requires_grad = True
        
        # B.4. Discriminative learning rates
        # Group parameters by layer to apply different learning rates
        # Earlier layers get smaller learning rates (already well-trained)
        # Later layers get higher learning rates (need more adaptation)
        layer_params = [
            {'params': model.layer1.parameters(), 'lr': lr/9},  # Earlier layers - smaller learning rate
            {'params': model.layer2.parameters(), 'lr': lr/3},
            {'params': model.layer3.parameters(), 'lr': lr/3},
            {'params': model.layer4.parameters(), 'lr': lr},    # Later layers - higher learning rate
            {'params': model.fc.parameters(), 'lr': lr*3}       # New classification layer - highest learning rate
        ]
        
        optimizer = torch.optim.Adam(layer_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr*3, total_steps=unfreeze_epochs * len(train_loader)
        )
        
        for epoch in range(unfreeze_epochs):
            print(f"\nEpoch {freeze_epochs+epoch+1}/{epochs}")
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate_epoch(model, valid_loader, criterion, device)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
    
    total_time = time.time() - total_start_time
    print(f"\nTotal training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models/model_final.pth')
    print(f"Model saved to models/model_final.pth")
    
    # A.4.2. Visualize training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.close()
    
    # A.4.3. Create confusion matrix
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader, desc="Creating confusion matrix"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Create and plot confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(20, 20))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    
    return model, history

def main():
    # Setup environment
    device = setup_device()
    
    # A.1. Download and analyze the data
    data_path = "Art Dataset"
    os.makedirs('results', exist_ok=True)
    
    # A.1.1. Inspect the data layout
    print("\n===== A.1.1. Inspecting data layout =====")
    df, classes = analyze_dataset(data_path)
    
    # A.2. Create the DataBlock and dataloaders
    print("\n===== A.2. Creating DataLoaders =====")
    train_loader, valid_loader, class_names = create_dataloaders(
        data_path, batch_size=32, img_size=224, augment=True, 
        balance_method='weighted', valid_pct=0.2
    )
    
    # A.3. Inspect the DataBlock via dataloader
    print("\n===== A.3. Inspecting DataBlock =====")
    
    # A.3.1. Show batch
    def visualize_batch(dataloader, num_images=16):
        """Display a batch of images from the dataloader"""
        # Get a batch
        images, labels = next(iter(dataloader))
        images = images[:num_images]
        labels = labels[:num_images]
        
        # Convert tensors back to images
        # (unnormalize first)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        # Create a grid of images
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
        for i, (img, label) in enumerate(zip(images, labels)):
            # Unnormalize
            img = img.cpu() * std[:, None, None] + mean[:, None, None]
            # Convert to numpy
            img = img.permute(1, 2, 0).numpy()
            # Clip values to valid range
            img = np.clip(img, 0, 1)
            
            # Get class name
            class_name = class_names[label]
            class_name = class_name.replace('_', ' ')
            
            # Plot
            row, col = i // 4, i % 4
            axes[row, col].imshow(img)
            axes[row, col].set_title(class_name)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/batch_preview.png')
        plt.close()
        print("Batch preview saved to results/batch_preview.png")
    
    # A.3.1. Show batch: dataloader.show_batch()
    print("\n===== A.3.1. Showing batch =====")
    visualize_batch(train_loader)
    
    # A.3.2. Check the labels
    print("\n===== A.3.2. Checking labels =====")
    print(f"Class names: {class_names}")
    
    # A.3.3. Summarize the DataBlock
    print("\n===== A.3.3. Summarizing DataBlock =====")
    print(f"Number of classes: {len(class_names)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(valid_loader)}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Total training samples: {len(train_loader.dataset)}")
    print(f"Total validation samples: {len(valid_loader.dataset)}")
    
    # A.4. Train a simple model
    print("\n===== A.4. Training a simple model =====")
    model, history = train_model(
        train_loader, valid_loader, class_names, device,
        model_name="resnet34", lr=1e-3, 
        epochs=10, freeze_epochs=3, unfreeze_epochs=7
    )
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main() 