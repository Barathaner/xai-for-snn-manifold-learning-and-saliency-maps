#!/usr/bin/env python3
"""
Test-Skript für die korrigierte compute_embeddings.py
"""

from compute_and_save_embeddings import compute_embeddings

def test_corrected_embeddings():
    """
    Testet die korrigierte Embedding-Berechnung mit wenigen Samples
    """
    
    print("🧪 Teste korrigierte Embedding-Berechnung...")
    print("=" * 60)
    
    # Test 1: t-SNE mit wenigen Samples
    print("\n1️⃣ Test: t-SNE mit 5 Samples")
    df_tsne = compute_embeddings(
        dataset_split='train',
        method='tsne',
        n_components=3,
        n_time_bins=80,
        max_samples=5,
        output_file='data/test_tsne_corrected.csv'
    )
    
    if df_tsne is not None:
        print(f"✅ t-SNE erfolgreich!")
        print(f"   Shape: {df_tsne.shape}")
        print(f"   Samples: {df_tsne['sample_id'].nunique()}")
        print(f"   Labels: {sorted(df_tsne['label'].unique())}")
        print(f"   Zeitbins pro Sample: {df_tsne.groupby('sample_id')['time_bin'].count().iloc[0]}")
    else:
        print("❌ t-SNE fehlgeschlagen!")
    
    print("\n" + "=" * 60)
    
    # Test 2: PCA mit wenigen Samples
    print("\n2️⃣ Test: PCA mit 5 Samples")
    df_pca = compute_embeddings(
        dataset_split='train',
        method='pca',
        n_components=3,
        n_time_bins=80,
        max_samples=5,
        output_file='data/test_pca_corrected.csv'
    )
    
    if df_pca is not None:
        print(f"✅ PCA erfolgreich!")
        print(f"   Shape: {df_pca.shape}")
        print(f"   Samples: {df_pca['sample_id'].nunique()}")
        print(f"   Labels: {sorted(df_pca['label'].unique())}")
        print(f"   Zeitbins pro Sample: {df_pca.groupby('sample_id')['time_bin'].count().iloc[0]}")
    else:
        print("❌ PCA fehlgeschlagen!")
    
    print("\n" + "=" * 60)
    
    # Test 3: Isomap mit wenigen Samples
    print("\n3️⃣ Test: Isomap mit 5 Samples")
    df_isomap = compute_embeddings(
        dataset_split='train',
        method='isomap',
        n_neighbors=5,
        n_components=3,
        n_time_bins=80,
        max_samples=5,
        output_file='data/test_isomap_corrected.csv'
    )
    
    if df_isomap is not None:
        print(f"✅ Isomap erfolgreich!")
        print(f"   Shape: {df_isomap.shape}")
        print(f"   Samples: {df_isomap['sample_id'].nunique()}")
        print(f"   Labels: {sorted(df_isomap['label'].unique())}")
        print(f"   Zeitbins pro Sample: {df_isomap.groupby('sample_id')['time_bin'].count().iloc[0]}")
    else:
        print("❌ Isomap fehlgeschlagen!")
    
    print("\n" + "=" * 60)
    print("🎉 Test abgeschlossen!")
    
    return df_tsne, df_pca, df_isomap

if __name__ == "__main__":
    test_corrected_embeddings()
