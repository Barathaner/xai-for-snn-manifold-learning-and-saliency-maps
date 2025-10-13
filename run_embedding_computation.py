#!/usr/bin/env python3
"""
Einfaches Skript zum Berechnen der Embeddings für alle Daten.
Berechnet verschiedene Konfigurationen nacheinander.
"""

from compute_and_save_embeddings import compute_embeddings

# Konfigurationen die berechnet werden sollen
configs = [
    # Isomap mit verschiedenen Parametern
    {'method': 'isomap', 'n_neighbors': 5, 'n_components': 3, 'split': 'train'},
    {'method': 'isomap', 'n_neighbors': 10, 'n_components': 3, 'split': 'train'},
    
    # PCA
    {'method': 'pca', 'n_neighbors': 5, 'n_components': 3, 'split': 'train'},
    
    # Optional: Test-Set (kommentieren Sie aus, falls gewünscht)
    # {'method': 'isomap', 'n_neighbors': 5, 'n_components': 3, 'split': 'test'},
]

print("=" * 80)
print("Starte Embedding-Berechnung für alle Konfigurationen")
print("=" * 80)

for i, config in enumerate(configs, 1):
    print(f"\n\n{'='*80}")
    print(f"Konfiguration {i}/{len(configs)}: {config}")
    print(f"{'='*80}\n")
    
    try:
        df = compute_embeddings(
            dataset_split=config['split'],
            method=config['method'],
            n_neighbors=config.get('n_neighbors', 5),
            n_components=config['n_components'],
            n_time_bins=80,
            max_samples=None  # None = alle Samples
        )
        print(f"✅ Erfolgreich abgeschlossen!")
    except Exception as e:
        print(f"❌ Fehler bei Konfiguration {i}: {e}")
        import traceback
        traceback.print_exc()

print("\n\n" + "=" * 80)
print("FERTIG! Alle Embeddings wurden berechnet.")
print("=" * 80)
print("\nGespeicherte Dateien:")
import os
for file in os.listdir('data'):
    if file.startswith('embeddings_') and file.endswith('.csv'):
        filepath = os.path.join('data', file)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  - {file} ({size_mb:.2f} MB)")

