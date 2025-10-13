# 🔄 Workflow: Activity-Logs → Manifold-Visualisierung

## Kompletter Workflow in 4 Schritten

### 📋 **Schritt 1: CSV laden in Pandas**

```bash
python3 process_activity_logs.py --activity-dir ./activity_logs --method isomap
```

Das Skript:
1. Lädt alle Activity-Log CSVs aus `./activity_logs/`
2. Rekonstruiert `vec`-Matrix für jedes Sample (wie im Notebook)
3. Wendet Isomap/PCA an
4. Speichert Embeddings als CSV

### 📊 **Schritt 2: Rekonstruiere vec (wie im Notebook)**

Automatisch im Skript:

```python
def reconstruct_vec_from_csv(df, n_time_bins=80, n_neurons=70):
    vec = np.zeros((n_time_bins, n_neurons))
    
    for _, row in df.iterrows():
        t = int(row['time_bin'])
        n = int(row['neuron_id'])
        vec[t, n] = row['value']
    
    return vec  # Genau wie im Notebook!
```

### 🎯 **Schritt 3: Wende PCA/Isomap an & speichere als CSV**

```bash
# Isomap
python3 process_activity_logs.py \
    --method isomap \
    --n-neighbors 5 \
    --n-components 3 \
    --layer hidden

# PCA  
python3 process_activity_logs.py \
    --method pca \
    --n-components 3 \
    --layer hidden

# Oder beides mit dem Shell-Skript:
bash run_full_pipeline.sh
```

**Output:** `./manifold_embeddings/embeddings_isomap_n5_c3.csv`

### 🎨 **Schritt 4: Visualisiere in Dash-App!**

```bash
python3 visualizations/plotly_webapp_visu/app_with_embeddings.py
```

Die App findet **automatisch** alle Embeddings aus:
- `./data/embeddings_*.csv` (Original-Daten)
- `./manifold_embeddings/embeddings_*.csv` (Activity-Log-Embeddings)

## 🚀 Schnellstart

### Komplette Pipeline ausführen:

```bash
# 1. Training mit Activity Logging (falls noch nicht gemacht)
python3 experiments/main_with_activity_logging.py

# 2. Verarbeite Activity-Logs zu Embeddings
bash run_full_pipeline.sh

# 3. Starte Dash-App
python3 visualizations/plotly_webapp_visu/app_with_embeddings.py
```

Dann öffnen: **http://127.0.0.1:8050**

## 📂 Dateistruktur

Nach der Pipeline:

```
activity_logs/              # Schritt 1: Training Output
├── epoch_001_hidden_binned.csv
├── epoch_002_hidden_binned.csv
└── ...

manifold_embeddings/        # Schritt 2: Embeddings
├── embeddings_isomap_n5_c3.csv
├── embeddings_pca_n0_c3.csv
└── ...

Dash-App zeigt alle!        # Schritt 3: Visualisierung
```

## 📊 CSV-Formate

### Activity-Log CSV (Input):
```csv
sample_id,epoch,layer,time_bin,neuron_id,value,label
0,1,hidden,0,42,0.87,5
0,1,hidden,0,123,0.65,5
...
```

### Embedding CSV (Output - Long Format):
```csv
sample_id,epoch,layer,label,method,time_bin,x,y,z,n_neighbors,n_components
0,1,hidden,5,isomap,0,-16.61,-2.92,-4.90,5,3
0,1,hidden,5,isomap,1,-16.19,-2.78,-4.72,5,3
...
```

**Perfekt für Dash!** Jede Zeile = 1 Punkt zu 1 Zeitpunkt

## 🔧 Alle Optionen

### process_activity_logs.py:

```bash
python3 process_activity_logs.py \
    --activity-dir ./activity_logs \      # Input-Verzeichnis
    --output-dir ./manifold_embeddings \  # Output-Verzeichnis
    --method isomap \                     # isomap, pca, tsne, umap
    --n-neighbors 5 \                     # Für Isomap/UMAP
    --n-components 3 \                    # 2D oder 3D
    --layer hidden                        # hidden, output, oder weglassen für beide
```

### Nur bestimmte Epochen:

```bash
# Manuell: Nur Epoche 6 verarbeiten
python3 -c "
from process_activity_logs import process_activity_logs
import pandas as pd

# Nur Epoche 6, Hidden Layer
df = process_activity_logs(
    csv_file='./activity_logs/epoch_006_hidden_binned.csv',
    method='isomap',
    n_neighbors=5,
    n_components=3
)

df.to_csv('./manifold_embeddings/epoch_006_isomap.csv', index=False)
"
```

## 📈 Vergleich über Epochen

Da jede CSV die Epoche enthält, können Sie in der Dash-App oder separat analysieren:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Lade Embeddings
df = pd.read_csv('./manifold_embeddings/embeddings_isomap_n5_c3.csv')

# Filter auf ein Sample
sample_df = df[df['sample_id'] == 0]

# Plot für verschiedene Epochen
fig = plt.figure(figsize=(15, 10))

for i, epoch in enumerate(sorted(sample_df['epoch'].unique())):
    epoch_df = sample_df[sample_df['epoch'] == epoch]
    
    ax = fig.add_subplot(2, 3, i+1, projection='3d')
    ax.scatter(epoch_df['x'], epoch_df['y'], epoch_df['z'], 
               c=epoch_df['time_bin'], cmap='viridis')
    ax.set_title(f'Epoche {epoch}')

plt.tight_layout()
plt.savefig('manifolds_evolution.png')
```

## 🎯 Use Cases

### 1. Training Evolution visualisieren:
- Verarbeite alle Epochen
- Vergleiche Manifolds über Zeit
- Sehe wie Netzwerk lernt

### 2. Layer-Vergleich:
```bash
# Hidden Layer
python3 process_activity_logs.py --layer hidden --method isomap

# Output Layer
python3 process_activity_logs.py --layer output --method isomap
```

### 3. Methoden-Vergleich:
```bash
# Verschiedene Methoden auf gleichen Daten
for method in isomap pca tsne umap; do
    python3 process_activity_logs.py --method $method --layer hidden
done
```

Dann in Dash alle vergleichen!

## ✅ Zusammenfassung

**Was Sie jetzt haben:**

1. ✅ **Activity-Logs** vom Training (CSV)
2. ✅ **Rekonstruktion** von vec-Matrix (wie im Notebook)
3. ✅ **Manifold-Learning** (Isomap/PCA/etc.)
4. ✅ **Embedding-CSVs** (Long Format)
5. ✅ **Dash-Visualisierung** (automatisch alle Dateien)

**Ein Befehl für alles:**

```bash
bash run_full_pipeline.sh && python3 visualizations/plotly_webapp_visu/app_with_embeddings.py
```

🎉 **Fertig!**

