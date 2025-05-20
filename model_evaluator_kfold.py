import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
from tqdm import tqdm
import pandas as pd
import random
from collections import defaultdict

# MPS (Metal Performance Shaders) kontrolü - Apple GPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"Metal GPU kullanılıyor: {DEVICE}")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Metal GPU bulunamadı, şu cihaz kullanılıyor: {DEVICE}")

# Sabit değerler
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 6
MAX_SAMPLES_PER_CLASS = 20  # Her sınıftan maksimum örnek sayısı (hızlı test için)
K_FOLDS = 5  # 5-fold cross validation

# Test veri seti için dönüşüm
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
            # Sınıfları örneklerden çıkar
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

def create_balanced_dataset(data_dir, max_per_class=None):
    """Her sınıftan eşit sayıda örnek içeren dengeli bir veri seti oluştur"""
    class_samples = defaultdict(list)
    
    # Tüm örnekleri sınıflarına göre topla
    for class_dir in Path(data_dir).iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            for img_path in class_dir.glob('*'):
                class_samples[class_name].append((img_path, class_name))
    
    # Her sınıftan maksimum sayıda örnek seç
    balanced_samples = []
    for class_name, samples in class_samples.items():
        random.shuffle(samples)
        
        # Maksimum örnek sayısını sınırla
        if max_per_class and len(samples) > max_per_class:
            samples = samples[:max_per_class]
            
        balanced_samples.extend(samples)
    
    print(f"Toplam {len(balanced_samples)} örnek, {len(class_samples)} farklı sanat akımından seçildi.")
    
    # Sınıf-indeks eşleştirmesini oluştur
    classes = sorted(class_samples.keys())
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    return balanced_samples, class_to_idx

def load_model(model_path, num_classes):
    """Model dosyasını yükle"""
    print(f"Model yükleniyor: {model_path}")
    # ResNet34 modelini oluştur
    model = models.resnet34(weights=None)
    # Son fully-connected katmanını güncelle
    model.fc = nn.Linear(512, num_classes)
    
    # Metal GPU kullanılabilirliği kontrolü için özel yükleme
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    
    return model

def evaluate_model(model, test_loader, classes):
    """Modeli değerlendir ve metrikleri döndür"""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Değerlendirme", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # MPS cihazında çalıştır
            outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            
            # Sonuçları CPU'ya taşı
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Temel metrikleri hesapla - uyarıları engellemek için zero_division=1 parametresi eklendi
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
    
    # Sınıf bazında doğruluk
    class_accuracy = {}
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    for i, class_name in enumerate(classes):
        class_samples = np.sum(np.array(all_labels) == i)
        class_correct = conf_matrix[i, i] if i < len(conf_matrix) else 0
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

def k_fold_cross_validation(dataset, model_paths, num_classes, k=5):
    """K-fold cross validation ile modelleri değerlendir"""
    
    # K-fold nesnesi oluştur
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Her model için sonuçları sakla
    all_model_results = {}
    for model_path in model_paths:
        model_name = Path(model_path).name
        all_model_results[model_name] = {
            'fold_results': [],
            'accuracy': [],
            'f1_score': [],
            'precision': [],
            'recall': []
        }
    
    # K-fold cross validation
    for fold, (_, test_indices) in enumerate(kfold.split(dataset)):
        print(f"\nFold {fold+1}/{k} değerlendiriliyor...")
        
        # Test veri setini oluştur
        test_subset = Subset(dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
        
        # Her model için değerlendirme yap
        for model_path in model_paths:
            model_name = Path(model_path).name
            print(f"  {model_name} değerlendiriliyor...")
            
            # Modeli yükle
            model = load_model(model_path, num_classes)
            
            # Modeli değerlendir
            results = evaluate_model(model, test_loader, dataset.classes)
            
            # Sonuçları kaydet
            all_model_results[model_name]['fold_results'].append(results)
            all_model_results[model_name]['accuracy'].append(results['accuracy'])
            all_model_results[model_name]['f1_score'].append(results['f1_score'])
            all_model_results[model_name]['precision'].append(results['precision'])
            all_model_results[model_name]['recall'].append(results['recall'])
            
            print(f"    Fold {fold+1} - Doğruluk: {results['accuracy']:.4f}, F1: {results['f1_score']:.4f}")
    
    # Her model için ortalama sonuçları hesapla
    summary_results = {}
    for model_name, results in all_model_results.items():
        summary_results[model_name] = {
            'mean_accuracy': np.mean(results['accuracy']),
            'std_accuracy': np.std(results['accuracy']),
            'mean_f1': np.mean(results['f1_score']),
            'std_f1': np.std(results['f1_score']),
            'mean_precision': np.mean(results['precision']),
            'std_precision': np.std(results['precision']),
            'mean_recall': np.mean(results['recall']),
            'std_recall': np.std(results['recall']),
            'fold_accuracy': results['accuracy'],
            'fold_f1': results['f1_score']
        }
    
    return summary_results

def plot_kfold_results(summary_results, save_dir):
    """K-fold cross validation sonuçlarını gösteren grafikler oluştur"""
    
    # Accuracy ve F1 için ortalama değerleri çiz
    plt.figure(figsize=(14, 7))
    
    # Model isimlerini ve ortalama değerleri çıkart
    model_names = list(summary_results.keys())
    model_names = [Path(name).stem for name in model_names]  # .pth uzantısını kaldır
    
    # Doğruluk ve F1 skorları
    mean_accuracy = [summary_results[model]['mean_accuracy'] for model in summary_results]
    std_accuracy = [summary_results[model]['std_accuracy'] for model in summary_results]
    mean_f1 = [summary_results[model]['mean_f1'] for model in summary_results]
    std_f1 = [summary_results[model]['std_f1'] for model in summary_results]
    
    # X ekseni konumları
    x = np.arange(len(model_names))
    width = 0.35
    
    # Çubuk grafikleri
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, mean_accuracy, width, yerr=std_accuracy, 
                    label='Accuracy', capsize=5, color='cornflowerblue')
    rects2 = ax.bar(x + width/2, mean_f1, width, yerr=std_f1, 
                    label='F1 Score', capsize=5, color='lightcoral')
    
    # Grafik özellikleri
    ax.set_ylabel('Skor')
    ax.set_title('5-Fold Cross Validation Ortalama Performans (Ortalama ± Std)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    # Çubukların üstüne değerleri ekle
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(rects1)
    add_labels(rects2)
    
    plt.tight_layout()
    
    # Grafiği kaydet
    save_path = Path(save_dir) / "kfold_mean_performance.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Her bir fold için performansı çiz
    plt.figure(figsize=(18, 12))
    
    # Accuracy için
    plt.subplot(2, 1, 1)
    for model_name in summary_results:
        model_stem = Path(model_name).stem
        plt.plot(range(1, K_FOLDS + 1), summary_results[model_name]['fold_accuracy'], 
                 marker='o', linestyle='-', label=model_stem)
    
    plt.title('Her Fold için Accuracy Değerleri')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, K_FOLDS + 1))
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # F1 Skor için
    plt.subplot(2, 1, 2)
    for model_name in summary_results:
        model_stem = Path(model_name).stem
        plt.plot(range(1, K_FOLDS + 1), summary_results[model_name]['fold_f1'], 
                 marker='o', linestyle='-', label=model_stem)
    
    plt.title('Her Fold için F1 Değerleri')
    plt.xlabel('Fold')
    plt.ylabel('F1 Score')
    plt.xticks(range(1, K_FOLDS + 1))
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    
    # Grafiği kaydet
    save_path = Path(save_dir) / "kfold_all_folds_performance.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    # Veri dizini ve sonuç dizini
    art_dataset_dir = 'Art Dataset'
    models_dir = 'models'
    results_dir = 'kfold_evaluation_results'
    
    # Sonuç dizinini oluştur
    os.makedirs(results_dir, exist_ok=True)
    
    # Dengeli veri setini oluştur - her sınıftan maksimum örnek sayısını sınırla
    samples, class_to_idx = create_balanced_dataset(art_dataset_dir, max_per_class=MAX_SAMPLES_PER_CLASS)
    dataset = ArtDataset(samples, transform=test_transform, class_to_idx=class_to_idx)
    
    num_classes = len(dataset.classes)
    print(f"Sanat sınıfları: {len(dataset.classes)}")
    
    # Model dosyalarını bul (.DS_Store gibi dosyaları hariç tut)
    model_paths = [os.path.join(models_dir, f) for f in os.listdir(models_dir) 
                 if f.endswith('.pth') and not f.startswith('.')]
    
    # K-fold cross validation ile modelleri değerlendir
    summary_results = k_fold_cross_validation(dataset, model_paths, num_classes, k=K_FOLDS)
    
    # Sonuçları görselleştir
    plot_kfold_results(summary_results, results_dir)
    
    # Sonuçları yazdır
    print("\n5-Fold Cross Validation Sonuçları:")
    for model_name, results in summary_results.items():
        print(f"\n{model_name}:")
        print(f"  Ortalama Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        print(f"  Ortalama F1 Score: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
        print(f"  Ortalama Precision: {results['mean_precision']:.4f} ± {results['std_precision']:.4f}")
        print(f"  Ortalama Recall: {results['mean_recall']:.4f} ± {results['std_recall']:.4f}")
    
    # Sonuçları CSV dosyasına kaydet
    results_summary = []
    for model_name, results in summary_results.items():
        row = {
            'model': model_name,
            'mean_accuracy': results['mean_accuracy'],
            'std_accuracy': results['std_accuracy'],
            'mean_f1': results['mean_f1'],
            'std_f1': results['std_f1'],
            'mean_precision': results['mean_precision'],
            'std_precision': results['std_precision'],
            'mean_recall': results['mean_recall'],
            'std_recall': results['std_recall']
        }
        results_summary.append(row)
    
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(f"{results_dir}/kfold_model_comparison_summary.csv", index=False)
    
    print(f"\nDeğerlendirme tamamlandı. Sonuçlar '{results_dir}' dizininde.")

if __name__ == "__main__":
    # Tekrar üretilebilirlik için seed ayarla
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main() 