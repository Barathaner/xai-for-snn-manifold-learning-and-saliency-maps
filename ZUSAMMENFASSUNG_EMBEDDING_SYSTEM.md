# 📊 Zusammenfassung: Embedding-System für SHD-Daten

## ✅ Was wurde erstellt?

Ein vollständiges System zur Berechnung und Visualisierung von Dimensionsreduktions-Embeddings:

### 1. **Berechnung-Skripte** 
- `compute_and_save_embeddings.py` - Hauptskript für Embedding-Berechnung
- `run_embedding_computation.py` - Batch-Verarbeitung mehrerer Konfigurationen
- `test_embedding_quick.py` - Schnelltest mit 10 Samples

### 2. **Dash Visualisierungs-App**
- `visualizations/plotly_webapp_visu/app_with_embeddings.py` - Interaktive 3D-Visualisierung

### 3. **Dokumentation**
- `EMBEDDING_BERECHNUNG.md` - Vollständige Anleitung
- `ZUSAMMENFASSUNG_EMBEDDING_SYSTEM.md` - Diese Datei

## 📋 DataFrame-Struktur (Long Format)

### Spalten (11 total):

| Spalte | Typ | Beschreibung | Beispiel |
|--------|-----|--------------|----------|
| `sample_id` | int | Sample-Index | 0, 1, 2, ... |
| `label` | int | Klassen-Label | 0-9 (Ziffer) |
| `method` | str | Embedding-Methode | 'isomap', 'pca', 'tsne', 'umap' |
| `time_bin` | int | Zeitindex | 0-79 (bei 80 bins) |
| `x` | float | X-Koordinate | -16.61 |
| `y` | float | Y-Koordinate | -2.92 |
| `z` | float | Z-Koordinate | -4.90 |
| `n_neighbors` | int | Nachbarn-Parameter | 5, 10, ... |
| `n_components` | int | Anzahl Dimensionen | 2 oder 3 |
| `n_time_bins` | int | Anzahl Zeitbins | 80 |
| `dataset_split` | str | Train/Test | 'train' oder 'test' |

### Beispiel-Daten:

```csv
sample_id,label,method,time_bin,x,y,z,n_neighbors,n_components,n_time_bins,dataset_split
0,5,isomap,0,-16.61,-2.92,-4.90,5,3,80,train
0,5,isomap,1,-16.19,-2.78,-4.72,5,3,80,train
0,5,isomap,2,-15.69,-2.66,-4.47,5,3,80,train
...
```

### Größenordnung:

- **Pro Sample**: 80 Zeilen (bei 80 Zeitbins)
- **Für 100 Samples**: 8.000 Zeilen
- **Für alle 4.011 Train-Samples**: ~320.880 Zeilen
- **CSV-Größe**: ~15-30 MB (abhängig von Anzahl Samples)

## 🚀 Schnellstart

### Schritt 1: Test ausführen (10 Samples)

```bash
cd /home/karl/git/xai-for-snn-manifold-learning-and-saliency-maps
source venv/bin/activate
python3 test_embedding_quick.py
```

**Output**: `data/embeddings_test_quick.csv` (800 Zeilen)

### Schritt 2: Vollständige Berechnung für alle Daten

```bash
source venv/bin/activate
python3 run_embedding_computation.py
```

⏱️ **Dauer**: ~20-40 Minuten für alle Train-Samples

**Output**: 
- `data/embeddings_isomap_train_n5_c3.csv`
- `data/embeddings_isomap_train_n10_c3.csv`
- `data/embeddings_pca_train_n5_c3.csv`

### Schritt 3: Dash-App starten

```bash
source venv/bin/activate
python3 visualizations/plotly_webapp_visu/app_with_embeddings.py
```

🌐 Öffnen: http://127.0.0.1:8050

## 📊 Verwendung der Daten in Python

### CSV laden und visualisieren:

```python
import pandas as pd
import plotly.express as px

# Daten laden
df = pd.read_csv('data/embeddings_isomap_train_n5_c3.csv')

# Anschauen
print(df.head())
print(f"Shape: {df.shape}")
print(f"Samples: {df['sample_id'].nunique()}")
print(f"Labels: {df['label'].unique()}")

# Filter auf bestimmte Label
df_5 = df[df['label'] == 5]

# Nur erste 10 Samples
sample_ids = df_5['sample_id'].unique()[:10]
df_plot = df_5[df_5['sample_id'].isin(sample_ids)]

# 3D Plot
fig = px.line_3d(df_plot, 
                 x='x', y='y', z='z',
                 color='sample_id',
                 line_group='sample_id',
                 hover_data=['label', 'time_bin'])
fig.show()
```

### In SQLite importieren (optional):

```python
import sqlite3
import pandas as pd

# CSV laden
df = pd.read_csv('data/embeddings_isomap_train_n5_c3.csv')

# In SQLite speichern
conn = sqlite3.connect('data/embeddings.db')
df.to_sql('embeddings', conn, if_exists='replace', index=False)

# Indizes erstellen
conn.execute('CREATE INDEX idx_method_label ON embeddings(method, label)')
conn.execute('CREATE INDEX idx_sample ON embeddings(sample_id)')
conn.execute('CREATE INDEX idx_timebin ON embeddings(time_bin)')

conn.close()
print("✅ In SQLite importiert!")

# Später: Abfragen
conn = sqlite3.connect('data/embeddings.db')
df_query = pd.read_sql("""
    SELECT * FROM embeddings 
    WHERE method='isomap' AND label=5 
    LIMIT 800
""", conn)
conn.close()
```

## 🎯 Vorteile des Long Formats

### ✅ Perfekt für Plotly:
```python
# EINFACH - direkt verwendbar
fig = px.scatter_3d(df, x='x', y='y', z='z', color='time_bin')
```

### ✅ Einfache Filter:
```python
# Nach Label filtern
df[df['label'] == 5]

# Nach Zeit filtern
df[df['time_bin'] < 20]

# Nach Sample filtern
df[df['sample_id'].isin([0, 1, 2])]
```

### ✅ Groupby-Operationen:
```python
# Durchschnittliche Position pro Label
df.groupby('label')[['x', 'y', 'z']].mean()

# Trajektorienlänge berechnen
df.groupby('sample_id').apply(
    lambda g: np.sqrt(np.diff(g['x'])**2 + 
                     np.diff(g['y'])**2 + 
                     np.diff(g['z'])**2).sum()
)
```

## 🔄 Erweiterungen

### Weitere Methoden hinzufügen:

```bash
# t-SNE
python3 compute_and_save_embeddings.py --method tsne --n-components 3

# UMAP
python3 compute_and_save_embeddings.py --method umap --n-neighbors 15
```

### Test-Set berechnen:

```bash
python3 compute_and_save_embeddings.py --split test --method isomap
```

### Verschiedene Parameter testen:

```bash
# Mehr Nachbarn
python3 compute_and_save_embeddings.py --n-neighbors 20

# 2D statt 3D
python3 compute_and_save_embeddings.py --n-components 2
```

## 📁 Datei-Struktur

```
xai-for-snn-manifold-learning-and-saliency-maps/
├── compute_and_save_embeddings.py      # Haupt-Berechnungsskript
├── run_embedding_computation.py        # Batch-Verarbeitung
├── test_embedding_quick.py             # Schnelltest
├── EMBEDDING_BERECHNUNG.md             # Vollständige Doku
├── ZUSAMMENFASSUNG_EMBEDDING_SYSTEM.md # Diese Datei
├── requirements.txt                    # Dependencies (aktualisiert)
├── data/
│   ├── embeddings_isomap_train_n5_c3.csv
│   ├── embeddings_isomap_train_n10_c3.csv
│   ├── embeddings_pca_train_n5_c3.csv
│   └── embeddings_test_quick.csv
└── visualizations/
    └── plotly_webapp_visu/
        ├── app.py                      # Original
        └── app_with_embeddings.py      # Neue Version mit CSV-Support
```

## 🎨 Dash-App Features

Die neue Dash-App (`app_with_embeddings.py`) bietet:

1. **Datei-Auswahl** - Wähle verschiedene Embedding-Dateien
2. **Label-Filter** - Zeige nur bestimmte Ziffern
3. **Sample-Anzahl** - Slider für 1-100 Samples
4. **3 Visualisierungstypen**:
   - 🔹 Trajektorien (Linien mit Farbverlauf)
   - 🔹 Scatter (Punkte)
   - 🔹 Animation über Zeit
5. **Statistiken** - Echzeit-Infos über Daten

## ❓ FAQ

### Wie lange dauert die vollständige Berechnung?

- **Isomap (n=5)**: ~15-20 Minuten für 4.011 Samples
- **PCA**: ~3-5 Minuten
- **t-SNE**: ~45-60 Minuten
- **UMAP**: ~20-30 Minuten

### Kann ich die Berechnung unterbrechen und fortsetzen?

Aktuell nein, aber Sie können:
1. `--max-samples` verwenden um in Batches zu arbeiten
2. Oder das Skript anpassen um bereits berechnete Samples zu überspringen

### Wie groß werden die CSV-Dateien?

- **10 Samples**: ~70 KB
- **100 Samples**: ~700 KB
- **1.000 Samples**: ~7 MB
- **4.011 Samples (alle Train)**: ~28 MB

### Was wenn mir der Speicher ausgeht?

```python
# Teilweise laden mit pandas
chunks = pd.read_csv('data/embeddings_isomap_train_n5_c3.csv', 
                     chunksize=10000)
for chunk in chunks:
    # Verarbeite chunk
    pass
```

Oder in SQLite importieren und dann abfragen.

## ✨ Nächste Schritte

1. ✅ Test ausführen (`test_embedding_quick.py`)
2. ✅ Vollständige Berechnung starten (`run_embedding_computation.py`)
3. ✅ Dash-App starten und visualisieren
4. 🔄 Optional: In SQLite importieren für noch schnellere Abfragen
5. 📊 Optional: Weitere Analysen und Metriken berechnen

