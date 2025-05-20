#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from pprint import pprint
import os

# Model dosyasını yükle
model_path = 'models/model_final.pth'
checkpoint = torch.load(model_path)

# FC katmanı ağırlıklarının boyutunu kontrol et
if 'fc.weight' in checkpoint:
    print(f"FC katmanı ağırlık boyutu: {checkpoint['fc.weight'].shape}")
    print(f"FC katmanı bias boyutu: {checkpoint['fc.bias'].shape}")

# Modelin yapısını anlama
layer_types = {}
for key in checkpoint.keys():
    parts = key.split('.')
    if len(parts) >= 2:
        layer_type = parts[0]
        if layer_type not in layer_types:
            layer_types[layer_type] = []
        if len(parts) >= 3:
            sublayer = parts[1]
            if sublayer not in layer_types[layer_type]:
                layer_types[layer_type].append(sublayer)

print("\nModel katman yapısı:")
for layer, sublayers in layer_types.items():
    if isinstance(sublayers, list) and len(sublayers) > 0:
        print(f"{layer}: {sublayers[:5]}{' ...' if len(sublayers) > 5 else ''}")
    else:
        print(f"{layer}")

# Model mimari özellikleri
print("\nLast layer shapes:")
for key, value in checkpoint.items():
    if key.startswith('layer4.2'):
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")

# Checkpoint'un tipini kontrol et
print(f"Checkpoint tipi: {type(checkpoint)}")

# Eğer bir sözlükse, anahtarları görüntüle
if isinstance(checkpoint, dict):
    print("\nCheckpoint anahtarları:")
    pprint(list(checkpoint.keys())[:10])  # İlk 10 anahtarı göster
    
    # Katmanları ve boyutları görüntüle
    print("\nBazı katmanların boyutları:")
    for key, value in list(checkpoint.items())[:20]:  # İlk 20 katmanı göster
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
else:
    # Model sınıfını ve mimari bilgilerini görüntüle
    print(f"\nModel sınıfı: {type(checkpoint)}")
    print(f"Model mimarisi:")
    print(checkpoint) 