"""
Callback zum Speichern neuronaler Aktivitäten während des Trainings.
Speichert die gebinnten Aktivierungen in CSV-Dateien (Long Format).
"""

import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path


class ActivityLoggerCallback:
    """
    Speichert neuronale Aktivitäten pro Layer und Epoche als CSV.
    
    Format der CSV (Long Format):
    - sample_id: Index des Samples im Batch
    - epoch: Aktuelle Epoche
    - layer: Name des Layers ('hidden', 'output', etc.)
    - time_bin: Zeitindex
    - neuron_id: Neuron-Index
    - activity: Aktivierungswert
    - label: Ground-Truth Label
    """
    
    def __init__(self, dataloader, device, out_dir="./activity_logs", 
                 max_samples_per_epoch=10, layer_names=None):
        """
        Args:
            dataloader: DataLoader für Daten
            device: CPU oder CUDA Device
            out_dir: Ausgabeverzeichnis für CSVs
            max_samples_per_epoch: Max. Anzahl Samples die gespeichert werden
            layer_names: Dict mit Layer-Namen, z.B. {'spk1': 'hidden', 'spk2': 'output'}
        """
        self.dataloader = dataloader
        self.device = device
        self.out_dir = Path(out_dir)
        self.max_samples_per_epoch = max_samples_per_epoch
        self.layer_names = layer_names or {'spk1': 'hidden', 'spk2': 'output'}
        
        # Erstelle Ausgabeverzeichnis
        self.out_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, net, epoch: int):
        """
        Wird am Ende jeder Epoche aufgerufen.
        Speichert die Aktivierungen als CSV.
        """
        net.eval()
        
        with torch.no_grad():
            # Hole einen Batch von Daten
            x, labels = next(iter(self.dataloader))
            
            # Squeeze wenn nötig
            if x.ndim == 4:
                x = x.squeeze(2)
            
            # Forward Pass
            _ = net(x.to(self.device))
        
        # Sammle Aktivierungen aus net.recordings
        for recording_name, layer_name in self.layer_names.items():
            spk = net.recordings.get(recording_name)
            
            if spk is None:
                print(f"Warnung: Keine Aufzeichnung für '{recording_name}' gefunden.")
                continue
            
            # Speichere als CSV
            self._save_activity_csv(
                spk=spk.cpu().numpy(),
                labels=labels.numpy(),
                epoch=epoch,
                layer_name=layer_name
            )
    
    def _save_activity_csv(self, spk, labels, epoch, layer_name):
        """
        Speichert Spike-Aktivität als CSV im Long Format.
        
        Args:
            spk: Spike-Tensor (batch, time, neurons)
            labels: Ground-Truth Labels (batch,)
            epoch: Epoche
            layer_name: Name des Layers
        """
        batch_size, n_time_bins, n_neurons = spk.shape
        
        # Limitiere Anzahl Samples
        n_samples = min(batch_size, self.max_samples_per_epoch)
        
        data = []
        
        for sample_idx in range(n_samples):
            label = int(labels[sample_idx])
            
            # Extrahiere Aktivität für dieses Sample
            sample_activity = spk[sample_idx]  # (time, neurons)
            
            # Konvertiere zu Long Format
            for t in range(n_time_bins):
                for n in range(n_neurons):
                    activity_value = float(sample_activity[t, n])
                    
                    # Nur speichern wenn Aktivität vorhanden (optional)
                    # Entfernen Sie diese Zeile, um alle Werte zu speichern
                    if activity_value > 0:
                        data.append({
                            'sample_id': sample_idx,
                            'epoch': epoch,
                            'layer': layer_name,
                            'time_bin': t,
                            'neuron_id': n,
                            'activity': activity_value,
                            'label': label
                        })
        
        # Erstelle DataFrame
        df = pd.DataFrame(data)
        
        # Speichere als CSV
        filename = self.out_dir / f"epoch_{epoch:03d}_{layer_name}_activity.csv"
        df.to_csv(filename, index=False)
        
        print(f"✅ Aktivitäten gespeichert: {filename} ({len(df)} Einträge)")


class BinnedActivityLoggerCallback:
    """
    Speichert gebinnte neuronale Aktivitäten (wie im Preprocessing für Manifolds).
    Speichert die durchschnittliche Aktivität pro Zeitbin als CSV.
    """
    
    def __init__(self, dataloader, device, out_dir="./binned_activity_logs", 
                 max_samples_per_epoch=10, layer_names=None):
        """
        Args:
            dataloader: DataLoader für Daten
            device: CPU oder CUDA Device
            out_dir: Ausgabeverzeichnis für CSVs
            max_samples_per_epoch: Max. Anzahl Samples die gespeichert werden
            layer_names: Dict mit Layer-Namen
        """
        self.dataloader = dataloader
        self.device = device
        self.out_dir = Path(out_dir)
        self.max_samples_per_epoch = max_samples_per_epoch
        self.layer_names = layer_names or {'spk1': 'hidden', 'spk2': 'output'}
        
        self.out_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, net, epoch: int):
        """Speichert gebinnte Aktivitäten am Ende der Epoche."""
        net.eval()
        
        all_data = {layer_name: [] for layer_name in self.layer_names.values()}
        all_labels = []
        samples_processed = 0
        
        with torch.no_grad():
            # Verarbeite mehrere Batches bis max_samples_per_epoch erreicht
            for batch_idx, (x, labels) in enumerate(self.dataloader):
                if samples_processed >= self.max_samples_per_epoch:
                    break
                    
                if x.ndim == 4:
                    x = x.squeeze(2)
                
                _ = net(x.to(self.device))
                
                # Speichere für jeden Layer
                for recording_name, layer_name in self.layer_names.items():
                    spk = net.recordings.get(recording_name)
                    
                    if spk is None:
                        continue
                    
                    # Sammle Daten für diesen Batch
                    batch_data = self._extract_batch_data(
                        spk=spk.cpu().numpy(),
                        labels=labels.numpy(),
                        epoch=epoch,
                        layer_name=layer_name,
                        max_samples=self.max_samples_per_epoch - samples_processed
                    )
                    all_data[layer_name].extend(batch_data)
                
                all_labels.extend(labels.numpy().tolist())
                samples_processed += len(labels)
        
        # Speichere alle gesammelten Daten
        for layer_name, data in all_data.items():
            if data:  # Nur speichern wenn Daten vorhanden
                df = pd.DataFrame(data)
                filename = self.out_dir / f"epoch_{epoch:03d}_{layer_name}_binned.csv"
                df.to_csv(filename, index=False)
                print(f"✅ Gebinnte Aktivitäten: {filename} ({len(df)} Einträge, {samples_processed} Samples)")
    
    def _extract_batch_data(self, spk, labels, epoch, layer_name, max_samples):
        """
        Extrahiert Batch-Daten für gebinnte Aktivität.
        
        Args:
            spk: Spike-Tensor (batch, time, neurons)
            labels: Labels (batch,)
            epoch: Epoche
            layer_name: Layer-Name
            max_samples: Max. Anzahl Samples für diesen Batch
        """
        batch_size, n_time_bins, n_neurons = spk.shape
        n_samples = min(batch_size, max_samples)
        
        data = []
        
        for sample_idx in range(n_samples):
            label = int(labels[sample_idx])
            vec = spk[sample_idx]  # (time, neurons) - entspricht vec im Notebook
            
            # Für jedes Zeitbin: speichere als Vektor (flattened)
            for t in range(n_time_bins):
                # Speichere nur die Neuron-Indizes mit deren Werten
                neuron_values = vec[t]  # (neurons,)
                
                # Long Format: ein Eintrag pro Neuron
                for n_id, value in enumerate(neuron_values):
                    # Speichere ALLE Werte (auch 0) für vollständige Daten
                    data.append({
                        'sample_id': sample_idx,
                        'epoch': epoch,
                        'layer': layer_name,
                        'time_bin': t,
                        'neuron_id': n_id,
                        'value': float(value),
                        'label': label
                    })
        
        return data

