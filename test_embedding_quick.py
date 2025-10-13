#!/usr/bin/env python3
"""
Schneller Test mit nur wenigen Samples zum Überprüfen der Pipeline.
"""

from compute_and_save_embeddings import compute_embeddings

print("=" * 80)
print("SCHNELLTEST: Berechne nur 10 Samples zum Testen")
print("=" * 80)

df = compute_embeddings(
    dataset_split='train',
    method='isomap',
    n_neighbors=5,
    n_components=3,
    n_time_bins=80,
    output_file='data/embeddings_test_quick.csv',
    max_samples=10  # Nur 10 Samples zum Testen
)

print("\n" + "=" * 80)
print("TEST ERFOLGREICH!")
print("=" * 80)
print("\nDataFrame Info:")
print(df.info())
print("\nNumerische Statistiken:")
print(df[['x', 'y', 'z']].describe())

