import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import pandas as pd
import random
from collections import defaultdict

# MPS (Metal Performance Shaders) check - Apple GPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"Using Metal GPU: {DEVICE}")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Metal GPU not found, using device: {DEVICE}")

# Constants
IMG_SIZE = 224
BATCH_SIZE = 64  # Batch size increased for GPU
NUM_WORKERS = 6  # Number of threads increased
MAX_SAMPLES_PER_CLASS = 30  # Maximum number of samples per class (for quick testing)

# Transformation for test dataset
test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ArtDataset(Dataset):
    def __init__(self, samples, transform=None, class_to_idx=None):
        self.samples = samples
        self.transform = transform
        
        if class_to_idx is None:
            # Extract classes from samples
            classes = set([Path(str(s[0])).parent.name for s in samples])
            self.classes = sorted(list(classes))
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = sorted(class_to_idx.keys(), key=lambda x: class_to_idx[x])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        label = self.class_to_idx[class_name]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def create_test_set(data_dir, test_ratio=0.2, max_per_class=None):
    """Create test set by taking a certain percentage of samples from each class"""
    class_samples = defaultdict(list)
    
    # Collect all examples by their classes
    for class_dir in Path(data_dir).iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            for img_path in class_dir.glob('*'):
                class_samples[class_name].append((img_path, class_name))
    
    # Select a certain percentage and maximum number of examples from each class
    test_samples = []
    for class_name, samples in class_samples.items():
        random.shuffle(samples)
        n_test = max(1, int(len(samples) * test_ratio))
        
        # Limit the maximum number of examples
        if max_per_class and n_test > max_per_class:
            n_test = max_per_class
            
        test_samples.extend(samples[:n_test])
    
    print(f"Total of {len(test_samples)} test samples selected from {len(class_samples)} different art movements.")
    
    # Create class-index mapping
    classes = sorted(class_samples.keys())
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    return test_samples, class_to_idx

def load_model(model_path, num_classes):
    """Load model file"""
    print(f"Loading model: {model_path}")
    # Create ResNet34 model
    model = models.resnet34(weights=None)
    # Update the last fully-connected layer
    model.fc = nn.Linear(512, num_classes)
    
    # Special loading for Metal GPU availability check
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    
    return model

def evaluate_model(model, test_loader, classes):
    """Evaluate model and return metrics"""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluation"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Run directly on MPS device (without using autocast)
            outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            
            # Move results to CPU
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    # Class-based accuracy
    class_accuracy = {}
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    for i, class_name in enumerate(classes):
        class_samples = np.sum(np.array(all_labels) == i)
        class_correct = conf_matrix[i, i]
        if class_samples > 0:
            class_accuracy[class_name] = class_correct / class_samples
    
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'class_accuracy': class_accuracy,
        'confusion_matrix': conf_matrix,
        'predictions': all_preds,
        'ground_truth': all_labels
    }
    
    return results

def plot_confusion_matrix(conf_matrix, classes, model_name, save_dir):
    """Plot confusion matrix graph"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    
    # Save the graph
    save_path = Path(save_dir) / f"conf_matrix_{Path(model_name).stem}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_class_accuracy(class_acc, model_name, save_dir):
    """Plot class-based accuracy graph"""
    plt.figure(figsize=(14, 8))
    
    # Sort classes by accuracy value
    sorted_items = sorted(class_acc.items(), key=lambda x: x[1], reverse=True)
    classes = [item[0] for item in sorted_items]
    accuracies = [item[1] for item in sorted_items]
    
    bars = plt.bar(classes, accuracies)
    plt.xlabel('Art Movement')
    plt.ylabel('Accuracy')
    plt.title(f'Class-Based Accuracy - {model_name}')
    plt.xticks(rotation=90)
    plt.ylim(0, 1.0)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    
    # Save the graph
    save_path = Path(save_dir) / f"class_accuracy_{Path(model_name).stem}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_model_comparison(all_results, save_dir):
    """Plot model comparison graph"""
    model_names = list(all_results.keys())
    metrics = ['accuracy', 'f1_score', 'precision', 'recall']
    
    # Collect metrics
    metric_data = {metric: [all_results[model][metric] for model in model_names] for metric in metrics}
    
    # Compare metrics
    plt.figure(figsize=(12, 7))
    x = np.arange(len(model_names))
    width = 0.2
    multiplier = 0
    
    for metric, values in metric_data.items():
        offset = width * multiplier
        bars = plt.bar(x + offset, values, width, label=metric)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        multiplier += 1
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width, model_names)
    plt.legend(loc='lower right')
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    
    # Save the graph
    save_path = Path(save_dir) / "model_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    # Data directory and results directory
    art_dataset_dir = 'Art Dataset'
    models_dir = 'models'
    results_dir = 'evaluation_results'
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Create test data - limit maximum number of examples from each class
    test_samples, class_to_idx = create_test_set(art_dataset_dir, test_ratio=0.2, max_per_class=MAX_SAMPLES_PER_CLASS)
    test_dataset = ArtDataset(test_samples, transform=test_transform, class_to_idx=class_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    
    classes = test_dataset.classes
    num_classes = len(classes)
    print(f"Art classes: {len(classes)}")
    
    # Find model files (exclude files like .DS_Store)
    model_paths = [os.path.join(models_dir, f) for f in os.listdir(models_dir) 
                 if f.endswith('.pth') and not f.startswith('.')]
    
    # Dictionary to store results
    all_results = {}
    
    # Evaluate each model
    for model_path in model_paths:
        model_name = Path(model_path).name
        print(f"\nEvaluating {model_name}...")
        
        # Load model
        model = load_model(model_path, num_classes)
        
        # Evaluate model
        results = evaluate_model(model, test_loader, classes)
        all_results[model_name] = results
        
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        
        # Plot confusion matrix graph
        plot_confusion_matrix(results['confusion_matrix'], classes, model_name, results_dir)
        
        # Plot class-based accuracy graph
        plot_class_accuracy(results['class_accuracy'], model_name, results_dir)
        
        # Save detailed class report
        report = classification_report(results['ground_truth'], results['predictions'], 
                                     target_names=classes, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f"{results_dir}/classification_report_{Path(model_name).stem}.csv")
    
    # Compare models
    if len(all_results) > 1:
        plot_model_comparison(all_results, results_dir)
    
    # Save results to CSV file
    results_summary = []
    for model_name, results in all_results.items():
        row = {
            'model': model_name,
            'accuracy': results['accuracy'],
            'f1_score': results['f1_score'],
            'precision': results['precision'],
            'recall': results['recall']
        }
        results_summary.append(row)
    
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(f"{results_dir}/model_comparison_summary.csv", index=False)
    
    print(f"\nEvaluation completed. Results are in '{results_dir}' directory.")

if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()