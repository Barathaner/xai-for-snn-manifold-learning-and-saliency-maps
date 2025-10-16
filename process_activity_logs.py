#!/usr/bin/env python3
"""
Verarbeitet Activity-Logs von SNN-Training zu Manifold-Embeddings.
1. Lädt CSV-Dateien
2. Rekonstruiert vec-Matrix (wie im Notebook)
3. Wendet PCA/Isomap an
4. Speichert Embeddings als CSV (Long Format für Dash)
"""

import pandas as pd
import numpy as np
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime


def reconstruct_vec_from_csv(df, n_time_bins=80, n_neurons=70):
    """
    Rekonstruiert vec-Matrix aus CSV (wie im Notebook).
    
    Args:
        df: DataFrame für ein Sample (gefiltert auf sample_id)
        n_time_bins: Anzahl Zeitbins
        n_neurons: Anzahl Neuronen
    
    Returns:
        vec: numpy array (n_time_bins, n_neurons)
    """
    vec = np.zeros((n_time_bins, n_neurons))
    
    for _, row in df.iterrows():
        t = int(row['time_bin'])
        n = int(row['neuron_id'])
        vec[t, n] = row['value']
    
    return vec


def process_activity_logs(csv_file, method='isomap', n_neighbors=5, n_components=3):
    """
    Verarbeitet Activity-Log CSV zu Embeddings.
    
    Args:
        csv_file: Pfad zur CSV-Datei
        method: 'isomap', 'pca', 'tsne', 'umap'
        n_neighbors: Für Isomap/UMAP
        n_components: Anzahl Dimensionen
    
    Returns:
        DataFrame mit Embeddings im Long Format
    """
    print(f"\n📂 Lade: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Extrahiere Metadaten
    epoch = df['epoch'].iloc[0]
    layer = df['layer'].iloc[0]
    
    # Bestimme Anzahl Neuronen und Zeitbins
    n_time_bins = df['time_bin'].max() + 1
    n_neurons = df['neuron_id'].max() + 1
    
    print(f"   Epoche: {epoch}, Layer: {layer}")
    print(f"   Zeitbins: {n_time_bins}, Neuronen: {n_neurons}")
    
    # Alle Samples
    sample_ids = df['sample_id'].unique()
    print(f"   Samples: {len(sample_ids)}")
    
    # Erstelle Modell
    if method == 'isomap':
        model = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    elif method == 'pca':
        model = PCA(n_components=n_components)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        model = TSNE(n_components=n_components, perplexity=min(30, n_time_bins-1))
    elif method == 'umap':
        import umap
        model = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components)
    else:
        raise ValueError(f"Unbekannte Methode: {method}")
    
    # KORRIGIERT: Alle Daten sammeln und zusammen transformieren
    all_vectors = []
    all_metadata = []
    
    print(f"   Sammle Daten für {len(sample_ids)} Samples...")
    
    for sample_id in tqdm(sample_ids, desc="  Sammle Daten"):
        # Filter auf dieses Sample
        sample_df = df[df['sample_id'] == sample_id]
        label = sample_df['label'].iloc[0]
        
        # Rekonstruiere vec
        vec = reconstruct_vec_from_csv(sample_df, n_time_bins, n_neurons)
        
        # Für jeden Zeitpunkt einen separaten Datenpunkt
        for t in range(vec.shape[0]):
            all_vectors.append(vec[t, :])  # Ein Zeitpunkt = ein Datenpunkt
            all_metadata.append({
                'sample_id': sample_id,
                'time_bin': t,
                'label': int(label)
            })
    
    print(f"   Gesammelt: {len(all_vectors)} Datenpunkte aus {len(sample_ids)} Samples")
    
    # ALLE Datenpunkte zusammen transformieren
    print(f"   Berechne {method.upper()} für alle {len(all_vectors)} Datenpunkte zusammen...")
    
    # Konvertiere zu numpy array
    X = np.array(all_vectors)  # Shape: (n_data_points, n_neurons)
    print(f"   Input Shape: {X.shape}")
    
    # KORRIGIERT: Einmalige Transformation aller Daten
    try:
        X_embedded = model.fit_transform(X)  # Shape: (n_data_points, n_components)
        print(f"   Output Shape: {X_embedded.shape}")
    except Exception as e:
        print(f"\n❌ Fehler bei {method.upper()}: {e}")
        return None
    
    # Ergebnisse in Long Format konvertieren
    all_embeddings = []
    for i, (point, meta) in enumerate(zip(X_embedded, all_metadata)):
        all_embeddings.append({
            'sample_id': meta['sample_id'],
            'epoch': epoch,
            'layer': layer,
            'label': meta['label'],
            'method': method,
            'time_bin': meta['time_bin'],
            'x': float(point[0]),
            'y': float(point[1]) if n_components > 1 else None,
            'z': float(point[2]) if n_components > 2 else None,
            'n_neighbors': n_neighbors if method in ['isomap', 'umap'] else None,
            'n_components': n_components,
        })
    
    # DataFrame erstellen
    embeddings_df = pd.DataFrame(all_embeddings)
    
    print(f"   ✅ {len(embeddings_df)} Einträge erstellt")
    
    return embeddings_df


def process_all_activity_logs(activity_dir='./activity_logs', 
                               output_dir='./manifold_embeddings',
                               method='isomap',
                               n_neighbors=5,
                               n_components=3,
                               layer_filter=None):
    """
    Verarbeitet alle Activity-Log CSVs im Verzeichnis.
    
    Args:
        activity_dir: Verzeichnis mit Activity-Logs
        output_dir: Ausgabeverzeichnis für Embeddings
        method: Embedding-Methode
        n_neighbors: Parameter
        n_components: Dimensionen
        layer_filter: Nur bestimmte Layer ('hidden', 'output', oder None für alle)
    """
    activity_path = Path(activity_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Finde alle CSV-Dateien
    csv_files = list(activity_path.glob('epoch_*_*_binned.csv'))
    
    if layer_filter:
        csv_files = [f for f in csv_files if layer_filter in f.name]
    
    csv_files = sorted(csv_files)
    
    print(f"\n{'='*80}")
    print(f"Verarbeite {len(csv_files)} Activity-Log Dateien")
    print(f"Methode: {method.upper()}, Komponenten: {n_components}")
    if method in ['isomap', 'umap']:
        print(f"Nachbarn: {n_neighbors}")
    print(f"{'='*80}")
    
    all_embeddings = []
    
    for csv_file in csv_files:
        embeddings_df = process_activity_logs(
            csv_file=csv_file,
            method=method,
            n_neighbors=n_neighbors,
            n_components=n_components
        )
        all_embeddings.append(embeddings_df)
    
    # Kombiniere alle Embeddings
    final_df = pd.concat(all_embeddings, ignore_index=True)
    
    # Speichere als CSV mit Zeitstempel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"embeddings_{method}_n{n_neighbors}_c{n_components}_{timestamp}.csv"
    final_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ FERTIG!")
    print(f"{'='*80}")
    print(f"Gespeichert: {output_file}")
    print(f"Shape: {final_df.shape}")
    print(f"\nSpalten: {list(final_df.columns)}")
    print(f"\nEpochen: {sorted(final_df['epoch'].unique())}")
    print(f"Layer: {sorted(final_df['layer'].unique())}")
    print(f"Samples: {final_df['sample_id'].nunique()}")
    print(f"Labels: {sorted(final_df['label'].unique())}")
    
    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verarbeitet Activity-Logs zu Manifold-Embeddings')
    parser.add_argument('--activity-dir', type=str, default='./activity_logs',
                       help='Verzeichnis mit Activity-Logs')
    parser.add_argument('--output-dir', type=str, default='./manifold_embeddings',
                       help='Ausgabeverzeichnis')
    parser.add_argument('--method', type=str, default='isomap',
                       choices=['isomap', 'pca', 'tsne', 'umap'],
                       help='Embedding-Methode')
    parser.add_argument('--n-neighbors', type=int, default=5,
                       help='Anzahl Nachbarn (für Isomap/UMAP)')
    parser.add_argument('--n-components', type=int, default=3,
                       help='Anzahl Dimensionen')
    parser.add_argument('--layer', type=str, default=None,
                       choices=['hidden', 'output'],
                       help='Nur bestimmten Layer verarbeiten')
    
    args = parser.parse_args()
    
    df = process_all_activity_logs(
        activity_dir=args.activity_dir,
        output_dir=args.output_dir,
        method=args.method,
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        layer_filter=args.layer
    )
    
    print(f"\n📊 Vorschau:")
    print(df.head(20))

