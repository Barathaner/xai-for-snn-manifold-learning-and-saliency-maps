#!/usr/bin/env python3
"""
Berechnet Isomap-Embeddings für alle SHD-Datenpunkte und speichert sie im Long Format als CSV.
"""

import tonic
import tonic.transforms as transforms
import numpy as np
import pandas as pd
from sklearn.manifold import Isomap
from torch.utils.data import Subset, Dataset
from tqdm import tqdm
import argparse
from pathlib import Path


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        events, label = self.dataset[idx]
        return self.transform(events), label


class Downsample1D:
    def __init__(self, spatial_factor=0.1):
        self.spatial_factor = spatial_factor

    def __call__(self, events):
        events = events.copy()
        events['x'] = (events['x'] * self.spatial_factor).astype(events['x'].dtype)
        return events


def compute_embeddings(dataset_split='train', method='isomap', n_neighbors=5, n_components=3,
                      n_time_bins=80, output_file=None, max_samples=None):
    """
    Berechnet Embeddings für alle Samples und speichert im Long Format.
    
    Args:
        dataset_split: 'train' oder 'test'
        method: 'isomap', 'tsne', 'umap', oder 'pca'
        n_neighbors: Anzahl Nachbarn für Isomap/UMAP
        n_components: Anzahl Dimensionen (2 oder 3)
        n_time_bins: Anzahl Zeitbins
        output_file: Pfad zur CSV-Datei
        max_samples: Optional, limitiert Anzahl Samples für Testing
    """
    
    print(f"Lade {dataset_split} Dataset...")
    if dataset_split == 'train':
        dataset = tonic.datasets.SHD(save_to="./data", train=True)
    else:
        dataset = tonic.datasets.SHD(save_to="./data", train=False)
    
    print(f"Original Dataset Größe: {len(dataset)} samples")
    
    # Filter auf Labels 0-9
    label_range = set(range(0, 10))
    filtered_indices = [
        i for i in range(len(dataset)) if dataset[i][1] in label_range
    ]
    dataset = Subset(dataset, filtered_indices)
    print(f"Nach Filterung: {len(dataset)} samples")
    
    # Transform Pipeline
    trans = transforms.Compose([
        Downsample1D(spatial_factor=0.1),   
        tonic.transforms.ToFrame(sensor_size=(70, 1, 1), n_time_bins=n_time_bins),
    ])
    
    # Modell erstellen
    if method == 'isomap':
        model = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        model = PCA(n_components=n_components)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        model = TSNE(n_components=n_components, perplexity=min(30, n_time_bins-1))
    elif method == 'umap':
        import umap
        model = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components)
    else:
        raise ValueError(f"Unbekannte Methode: {method}")
    
    # Parameter als Dict für Metadaten
    params = {
        'n_components': n_components,
        'n_neighbors': n_neighbors if method in ['isomap', 'umap'] else None,
        'n_time_bins': n_time_bins
    }
    
    # Daten sammeln
    all_data = []
    
    # Limitiere für Testing
    n_samples = min(max_samples, len(dataset)) if max_samples else len(dataset)
    
    print(f"Berechne {method.upper()} für {n_samples} samples...")
    
    for i in tqdm(range(n_samples), desc="Processing samples"):
        events, label = dataset[i]
        
        # Transform anwenden
        frames = trans(events)
        vec = frames[:, 0, :]  # Shape: (n_time_bins, n_neurons)
        
        # Embedding berechnen
        try:
            emb = model.fit_transform(vec)  # Shape: (n_time_bins, n_components)
        except Exception as e:
            print(f"\nWarnung: Sample {i} übersprungen. Fehler: {e}")
            continue
        
        # In Long Format konvertieren
        for t in range(emb.shape[0]):
            row = {
                'sample_id': i,
                'label': int(label),
                'method': method,
                'time_bin': t,
                'x': float(emb[t, 0]),
                'y': float(emb[t, 1]) if n_components > 1 else None,
                'z': float(emb[t, 2]) if n_components > 2 else None,
                'n_neighbors': n_neighbors if method in ['isomap', 'umap'] else None,
                'n_components': n_components,
                'n_time_bins': n_time_bins,
                'dataset_split': dataset_split
            }
            all_data.append(row)
    
    # DataFrame erstellen
    df = pd.DataFrame(all_data)
    
    # Output-Datei bestimmen
    if output_file is None:
        output_file = f"data/embeddings_{method}_{dataset_split}_n{n_neighbors}_c{n_components}.csv"
    
    # Verzeichnis erstellen falls nötig
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Speichern
    df.to_csv(output_file, index=False)
    print(f"\n✅ Gespeichert: {output_file}")
    print(f"   Shape: {df.shape}")
    print(f"   Spalten: {list(df.columns)}")
    print(f"\nErste Zeilen:")
    print(df.head(10))
    print(f"\nStatistiken:")
    print(f"   Samples: {df['sample_id'].nunique()}")
    print(f"   Labels: {sorted(df['label'].unique())}")
    print(f"   Zeitbins pro Sample: {df.groupby('sample_id')['time_bin'].count().iloc[0]}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Berechnet Embeddings für SHD-Dataset')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                       help='Dataset Split')
    parser.add_argument('--method', type=str, default='isomap', 
                       choices=['isomap', 'pca', 'tsne', 'umap'],
                       help='Embedding-Methode')
    parser.add_argument('--n-neighbors', type=int, default=5,
                       help='Anzahl Nachbarn (für Isomap/UMAP)')
    parser.add_argument('--n-components', type=int, default=3,
                       help='Anzahl Dimensionen')
    parser.add_argument('--n-time-bins', type=int, default=80,
                       help='Anzahl Zeitbins')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV Datei')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximale Anzahl Samples (für Testing)')
    
    args = parser.parse_args()
    
    df = compute_embeddings(
        dataset_split=args.split,
        method=args.method,
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        n_time_bins=args.n_time_bins,
        output_file=args.output,
        max_samples=args.max_samples
    )

