#!/usr/bin/env python3
"""
Training-Skript mit Activity Logging für Manifold-Analysen.
Verwendet dasselbe Preprocessing wie im Notebook.
"""

import sys
import os

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import random
import numpy as np

# Project imports
from data.dataloader import load_filtered_shd_dataloader
from data.preprocessing import get_manifold_preprocessing
from models.sffnn_batched import Net
from training.trainer import train_one_epoch_batched
from utils.metrics import print_full_dataloader_accuracy_batched
from training.callbacks import RasterPlotCallback
from training.activity_logger import BinnedActivityLoggerCallback

# Seed für Reproduzierbarkeit
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device setup
device = torch.device("cuda") if torch.cuda.is_available() else \
         torch.device("mps") if torch.backends.mps.is_available() else \
         torch.device("cpu")

print(f"Using device: {device}")

# ============================================================================
# PREPROCESSING (wie im Notebook für Manifold-Analysen)
# ============================================================================
transform = get_manifold_preprocessing(n_time_bins=80)

# Data loading
train_dataloader = load_filtered_shd_dataloader(
    label_range=range(0, 10), 
    transform=transform, 
    train=True, 
    batch_size=64
)

test_dataloader = load_filtered_shd_dataloader(
    label_range=range(0, 10), 
    transform=transform, 
    train=False, 
    batch_size=64
)

# ============================================================================
# MODEL (angepasst an 70 Neuronen Input und 80 Zeitschritte)
# ============================================================================
net = Net(
    num_inputs=70,      # Nach Downsample1D(0.1): 700 -> 70
    num_hidden=1000, 
    num_outputs=10, 
    num_steps=80,       # 80 Zeitbins
    beta=0.9
).to(device)

# ============================================================================
# TRAINING SETUP
# ============================================================================
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)

# ============================================================================
# CALLBACKS
# ============================================================================
# 1. Rasterplots (wie vorher)
raster_cb = RasterPlotCallback(
    dataloader=test_dataloader, 
    device=device, 
    out_dir="./plots", 
    max_neurons_hidden=1000
)

# 2. Gebinnte Aktivitäten für Manifold-Analysen
activity_cb = BinnedActivityLoggerCallback(
    dataloader=test_dataloader,
    device=device,
    out_dir="./activity_logs",
    max_samples_per_epoch=20,  # Speichere 20 Samples pro Epoche
    layer_names={'spk1': 'hidden', 'spk2': 'output'}
)

# ============================================================================
# TRAINING LOOP
# ============================================================================
if __name__ == "__main__":
    num_epochs = 6
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*80}")
        
        # Training
        train_one_epoch_batched(
            net=net,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=loss,
            device=device
        )
        
        # Callbacks am Ende der Epoche
        print("\nSpeichere Visualisierungen und Aktivitäten...")
        raster_cb.on_epoch_end(net, epoch)
        activity_cb.on_epoch_end(net, epoch)
    
    # ============================================================================
    # EVALUATION
    # ============================================================================
    print(f"\n{'='*80}")
    print("Finale Evaluation")
    print(f"{'='*80}")
    print_full_dataloader_accuracy_batched(net, test_dataloader)
    
    # ============================================================================
    # EXPORT
    # ============================================================================
    # Modell-Gewichte speichern
    torch.save(net.state_dict(), "./model_export/model_weights.pth")
    print("\n✅ Modell-Gewichte gespeichert: ./model_export/model_weights.pth")
    
    # NIR Export
    from snntorch.export_nir import export_to_nir
    import nir
    
    net_cpu = net.to("cpu")
    example_input_cpu = torch.rand((1, 80, 70))  # (batch, time, features)
    nir_graph = export_to_nir(net_cpu, example_input_cpu, ignore_dims=[0])
    nir.write("./model_export/my_model.nir", nir_graph)
    print("✅ NIR Graph gespeichert: ./model_export/my_model.nir")
    
    print(f"\n{'='*80}")
    print("Training abgeschlossen!")
    print(f"{'='*80}")
    print("\nGespeicherte Dateien:")
    print("  - Rasterplots: ./plots/")
    print("  - Aktivitäten (CSV): ./activity_logs/")
    print("  - Modell: ./model_export/")

