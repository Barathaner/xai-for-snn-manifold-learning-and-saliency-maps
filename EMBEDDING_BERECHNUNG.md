# Embedding-Berechnung für SHD-Dataset

## Überblick

Diese Skripte berechnen Dimensionsreduktions-Embeddings (Isomap, PCA, t-SNE, UMAP) für das SHD-Dataset und speichern die Ergebnisse im **Long Format** als CSV für die Verwendung in Plotly Dash.

## CSV-Format (Long Format)

Jede Zeile = ein Punkt zu einem Zeitpunkt:

```csv
sample_id,label,method,time_bin,x,y,z,n_neighbors,n_components,n_time_bins,dataset_split
0,5,isomap,0,-16.61,-2.92,-4.90,5,3,80,train
0,5,isomap,1,-16.19,-2.78,-4.72,5,3,80,train
0,5,isomap,2,-15.69,-2.66,-4.47,5,3,80,train
...
```

### Spalten:
- **sample_id**: Index des Samples (0, 1, 2, ...)
- **label**: Klassen-Label (0-9 für Ziffern)
- **method**: Embedding-Methode ('isomap', 'pca', 'tsne', 'umap')
- **time_bin**: Zeitindex (0 bis n_time_bins-1)
- **x, y, z**: 3D-Koordinaten im reduzierten Raum
- **n_neighbors**: Anzahl Nachbarn (für Isomap/UMAP)
- **n_components**: Anzahl Dimensionen (2 oder 3)
- **n_time_bins**: Anzahl Zeitbins (Standard: 80)
- **dataset_split**: 'train' oder 'test'

## Verwendung

### 1. Schnelltest (10 Samples)

```bash
source venv/bin/activate
python3 test_embedding_quick.py
```

**Output:** `data/embeddings_test_quick.csv`

### 2. Einzelne Konfiguration berechnen

```bash
source venv/bin/activate

# Isomap für Train-Set
python3 compute_and_save_embeddings.py --split train --method isomap --n-neighbors 5 --n-components 3

# PCA für Train-Set
python3 compute_and_save_embeddings.py --split train --method pca --n-components 3

# Mit begrenzter Anzahl Samples (zum Testen)
python3 compute_and_save_embeddings.py --split train --method isomap --max-samples 100
```

### 3. Alle Konfigurationen berechnen (empfohlen)

```bash
source venv/bin/activate
python3 run_embedding_computation.py
```

Dieser Befehl berechnet automatisch:
- Isomap (n_neighbors=5, n_components=3) für Train
- Isomap (n_neighbors=10, n_components=3) für Train
- PCA (n_components=3) für Train

**Dauer:** Ca. 1-2 Stunden für alle ~4000 Train-Samples (abhängig von CPU)

### 4. Nur bestimmte Anzahl Samples

Für schnellere Tests können Sie die Anzahl limitieren:

```bash
python3 compute_and_save_embeddings.py --split train --method isomap --max-samples 500
```

## Command-Line Optionen

```
--split {train,test}       Dataset Split (Standard: train)
--method {isomap,pca,tsne,umap}  Embedding-Methode (Standard: isomap)
--n-neighbors INT          Anzahl Nachbarn für Isomap/UMAP (Standard: 5)
--n-components INT         Anzahl Dimensionen (Standard: 3)
--n-time-bins INT          Anzahl Zeitbins (Standard: 80)
--output PATH              Output CSV Datei (optional)
--max-samples INT          Maximale Anzahl Samples (optional, für Testing)
```

## Gespeicherte Dateien

Nach der Berechnung finden Sie in `data/`:

```
data/
├── embeddings_isomap_train_n5_c3.csv    # Isomap, 5 Nachbarn, 3D
├── embeddings_isomap_train_n10_c3.csv   # Isomap, 10 Nachbarn, 3D
├── embeddings_pca_train_n5_c3.csv       # PCA, 3D
└── ...
```

## Verwendung in Plotly Dash

```python
import pandas as pd
import plotly.express as px

# CSV laden
df = pd.read_csv('data/embeddings_isomap_train_n5_c3.csv')

# Filter auf bestimmte Samples
samples = df[df['label'] == 5].sample_id.unique()[:10]
df_filtered = df[df['sample_id'].isin(samples)]

# 3D Scatter Plot mit Trajektorien
fig = px.line_3d(df_filtered, 
                 x='x', y='y', z='z',
                 color='sample_id',
                 line_group='sample_id',
                 hover_data=['label', 'time_bin'])
fig.show()
```

## Datenbank-Integration (Optional)

Für größere Datasets können Sie die CSV auch in SQLite importieren:

```python
import sqlite3
import pandas as pd

# CSV in SQLite laden
df = pd.read_csv('data/embeddings_isomap_train_n5_c3.csv')
conn = sqlite3.connect('data/embeddings.db')
df.to_sql('embeddings', conn, if_exists='append', index=False)

# Indizes für schnelle Abfragen
conn.execute('CREATE INDEX idx_method_label ON embeddings(method, label)')
conn.execute('CREATE INDEX idx_sample ON embeddings(sample_id)')
conn.close()
```

## Troubleshooting

### Warnung: "The number of connected components of the neighbors graph is 2 > 1"

Dies ist eine Warnung von sklearn bei Isomap. Lösungen:
- Erhöhen Sie `--n-neighbors` (z.B. auf 10 oder 15)
- Oder ignorieren Sie die Warnung (Ergebnis ist trotzdem nutzbar)

### Speicherprobleme

Falls der RAM nicht ausreicht:
- Verwenden Sie `--max-samples` um die Berechnung aufzuteilen
- Oder berechnen Sie nur bestimmte Labels:

```python
# Im Skript anpassen:
label_range = set([0, 1, 2])  # Nur Labels 0, 1, 2
```

## Performance

### Zeiten (ca., auf Standard-CPU):

- **Isomap**: ~0.1-0.3 Sekunden pro Sample
- **PCA**: ~0.01-0.05 Sekunden pro Sample  
- **t-SNE**: ~0.5-1 Sekunde pro Sample
- **UMAP**: ~0.2-0.5 Sekunden pro Sample

### Für 4000 Train-Samples:

- Isomap: ~10-20 Minuten
- PCA: ~2-5 Minuten
- t-SNE: ~30-60 Minuten
- UMAP: ~15-30 Minuten

