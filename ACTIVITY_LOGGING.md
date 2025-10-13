# Activity Logging für SNN-Training

## Übersicht

Dieses System speichert neuronale Aktivitäten während des SNN-Trainings und verwendet **dasselbe Preprocessing** wie für die Manifold-Analysen. Die Aktivitäten werden als CSV-Dateien gespeichert, getrennt nach Layer und Epoche.

## 📁 Neue Dateien

### 1. **Preprocessing-Modul** (`data/preprocessing.py`)
Wiederverwendbare Preprocessing-Funktionen:
- `Downsample1D(spatial_factor=0.1)` - 1D Downsampling
- `SqrtTransform()` - Wurzel-Transformation
- `get_manifold_preprocessing()` - Komplettes Preprocessing für Manifold-Analysen

### 2. **Activity Logger** (`training/activity_logger.py`)
Zwei Callback-Klassen zum Speichern der Aktivitäten:
- `ActivityLoggerCallback` - Speichert Spike-Aktivitäten
- `BinnedActivityLoggerCallback` - Speichert gebinnte Aktivitäten (wie `vec` im Notebook)

### 3. **Training-Skript** (`experiments/main_with_activity_logging.py`)
Vollständiges Trainings-Beispiel mit Activity Logging

## 🚀 Schnellstart

### Training mit Activity Logging starten:

```bash
cd /home/karl/git/xai-for-snn-manifold-learning-and-saliency-maps
source venv/bin/activate
python3 experiments/main_with_activity_logging.py
```

**Output:**
- Rasterplots: `./plots/`
- CSV-Aktivitäten: `./activity_logs/`
- Modell: `./model_export/`

## 📊 CSV-Format

### Gebinnte Aktivitäten (BinnedActivityLoggerCallback):

```csv
sample_id,epoch,layer,time_bin,neuron_id,value,label
0,1,hidden,0,42,0.87,5
0,1,hidden,0,123,0.65,5
0,1,hidden,1,42,0.92,5
1,1,hidden,0,15,0.73,3
...
```

**Spalten:**
- `sample_id`: Index des Samples im Batch (0, 1, 2, ...)
- `epoch`: Trainings-Epoche (1, 2, 3, ...)
- `layer`: Name des Layers ('hidden', 'output')
- `time_bin`: Zeitindex (0-79 bei 80 Bins)
- `neuron_id`: Neuron-Index
- `value`: Aktivierungswert
- `label`: Ground-Truth Label (0-9)

## 🔧 Verwendung in Ihrem Code

### Grundlegendes Setup:

```python
from data.preprocessing import get_manifold_preprocessing
from training.activity_logger import BinnedActivityLoggerCallback

# 1. Preprocessing (wie im Notebook)
transform = get_manifold_preprocessing(n_time_bins=80)

# 2. DataLoader
train_dataloader = load_filtered_shd_dataloader(
    label_range=range(0, 10),
    transform=transform,
    train=True,
    batch_size=64
)

# 3. Modell (angepasst an 70 Input-Neuronen, 80 Zeitschritte)
net = Net(
    num_inputs=70,      # 700 * 0.1 = 70 nach Downsampling
    num_hidden=1000,
    num_outputs=10,
    num_steps=80,       # 80 Zeitbins
    beta=0.9
).to(device)

# 4. Activity Logger Callback
activity_cb = BinnedActivityLoggerCallback(
    dataloader=test_dataloader,
    device=device,
    out_dir="./activity_logs",
    max_samples_per_epoch=20,  # Anzahl Samples die gespeichert werden
    layer_names={'spk1': 'hidden', 'spk2': 'output'}
)

# 5. Training Loop
for epoch in range(1, num_epochs + 1):
    train_one_epoch_batched(net, train_dataloader, optimizer, loss_fn, device)
    
    # Callback am Ende der Epoche
    activity_cb.on_epoch_end(net, epoch)
```

## 📈 Preprocessing-Details

### Manifold-Preprocessing (wie im Notebook):

```python
transform = transforms.Compose([
    Downsample1D(spatial_factor=0.1),   # 700 → 70 Neuronen
    ToFrame(sensor_size=(70, 1, 1), n_time_bins=80),
    SqrtTransform()                      # Wurzel-Normalisierung
])
```

**Resultat:**
- Input: Event-Stream
- Output: Tensor `(time_bins=80, neurons=70)` 

Dies entspricht **exakt** dem `vec` aus dem Notebook!

### Training-Preprocessing (Standard):

```python
from data.preprocessing import get_training_preprocessing

transform = get_training_preprocessing(
    n_time_bins=250,
    spatial_factor=0.5
)
```

## 🔄 Workflow: Training → Manifold-Analyse

### 1. Training mit Activity Logging:

```bash
python3 experiments/main_with_activity_logging.py
```

**Output:** `activity_logs/epoch_001_hidden_binned.csv`, etc.

### 2. CSV laden und analysieren:

```python
import pandas as pd

# Lade Aktivitäten
df = pd.read_csv('activity_logs/epoch_001_hidden_binned.csv')

print(df.head())
#   sample_id  epoch  layer  time_bin  neuron_id  value  label
# 0         0      1 hidden         0         42   0.87      5
# 1         0      1 hidden         0        123   0.65      5
# ...

# Filter auf bestimmtes Sample
sample_0 = df[df['sample_id'] == 0]

# Rekonstruiere vec-Matrix (wie im Notebook)
import numpy as np

n_time_bins = 80
n_neurons = 70

vec = np.zeros((n_time_bins, n_neurons))

for _, row in sample_0.iterrows():
    t = int(row['time_bin'])
    n = int(row['neuron_id'])
    vec[t, n] = row['value']

# Jetzt haben Sie vec wie im Notebook!
print(f"vec shape: {vec.shape}")  # (80, 70)
```

### 3. PCA/Isomap auf Aktivitäten anwenden:

```python
from sklearn.manifold import Isomap

# vec ist bereits im richtigen Format!
isomap = Isomap(n_neighbors=5, n_components=3)
embedding = isomap.fit_transform(vec)

print(f"Embedding shape: {embedding.shape}")  # (80, 3)
```

### 4. Oder: Batch-Verarbeitung aller Epochen:

```python
import glob

all_csvs = glob.glob('activity_logs/epoch_*_hidden_binned.csv')

for csv_file in all_csvs:
    df = pd.read_csv(csv_file)
    epoch = df['epoch'].iloc[0]
    
    # Verarbeite alle Samples
    for sample_id in df['sample_id'].unique():
        sample_df = df[df['sample_id'] == sample_id]
        
        # Rekonstruiere vec
        vec = np.zeros((80, 70))
        for _, row in sample_df.iterrows():
            vec[int(row['time_bin']), int(row['neuron_id'])] = row['value']
        
        # Wende Manifold-Learning an
        # ...
```

## ⚙️ Konfiguration

### Anzahl gespeicherter Samples ändern:

```python
activity_cb = BinnedActivityLoggerCallback(
    dataloader=test_dataloader,
    device=device,
    out_dir="./activity_logs",
    max_samples_per_epoch=50,  # Speichere 50 statt 20 Samples
    layer_names={'spk1': 'hidden', 'spk2': 'output'}
)
```

### Nur bestimmte Layer speichern:

```python
activity_cb = BinnedActivityLoggerCallback(
    # ...
    layer_names={'spk1': 'hidden'}  # Nur Hidden Layer
)
```

### Verschiedene Zeitbins:

```python
# Im Preprocessing
transform = get_manifold_preprocessing(n_time_bins=100)

# Im Modell anpassen
net = Net(
    num_inputs=70,
    num_hidden=1000,
    num_outputs=10,
    num_steps=100,  # Muss mit n_time_bins übereinstimmen!
    beta=0.9
)
```

## 📂 Dateistruktur

Nach dem Training:

```
activity_logs/
├── epoch_001_hidden_binned.csv
├── epoch_001_output_binned.csv
├── epoch_002_hidden_binned.csv
├── epoch_002_output_binned.csv
├── ...
└── epoch_006_output_binned.csv

plots/
├── epoch_001_hidden_raster.png
├── epoch_001_output_raster.png
└── ...

model_export/
├── model_weights.pth
└── my_model.nir
```

## 🔍 Analyse-Beispiele

### 1. Vergleich über Epochen:

```python
import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5, 6]
embeddings_per_epoch = []

for epoch in epochs:
    df = pd.read_csv(f'activity_logs/epoch_{epoch:03d}_hidden_binned.csv')
    
    # Sample 0
    sample_df = df[df['sample_id'] == 0]
    vec = reconstruct_vec(sample_df)  # Siehe oben
    
    isomap = Isomap(n_neighbors=5, n_components=3)
    emb = isomap.fit_transform(vec)
    embeddings_per_epoch.append(emb)

# 3D Plot über Epochen
fig = plt.figure(figsize=(15, 10))

for i, (epoch, emb) in enumerate(zip(epochs, embeddings_per_epoch)):
    ax = fig.add_subplot(2, 3, i+1, projection='3d')
    ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=range(len(emb)), cmap='viridis')
    ax.set_title(f'Epoche {epoch}')

plt.tight_layout()
plt.savefig('manifolds_over_epochs.png')
```

### 2. Vergleich Hidden vs. Output Layer:

```python
df_hidden = pd.read_csv('activity_logs/epoch_006_hidden_binned.csv')
df_output = pd.read_csv('activity_logs/epoch_006_output_binned.csv')

# Beide Layers für Sample 0
sample_0_hidden = df_hidden[df_hidden['sample_id'] == 0]
sample_0_output = df_output[df_output['sample_id'] == 0]

# Manifold-Analyse für beide...
```

## 💡 Tipps

### Speicherplatz sparen:

Die CSVs speichern nur **aktive Neuronen** (`value > 0`). Falls Sie alle Werte brauchen, ändern Sie in `activity_logger.py`:

```python
# Zeile entfernen:
if value > 0:  # ← Diese Zeile löschen

# Dann werden alle Werte gespeichert (größere Dateien!)
```

### Nur bestimmte Epochen speichern:

```python
for epoch in range(1, num_epochs + 1):
    train_one_epoch_batched(...)
    
    # Nur jede 2. Epoche speichern
    if epoch % 2 == 0:
        activity_cb.on_epoch_end(net, epoch)
```

### Eigene Layer hinzufügen:

Falls Ihr Modell mehr Layer hat:

```python
activity_cb = BinnedActivityLoggerCallback(
    # ...
    layer_names={
        'spk1': 'hidden1',
        'spk2': 'hidden2',
        'spk3': 'output'
    }
)
```

## ✅ Zusammenfassung

**Was Sie jetzt haben:**

1. ✅ **Gleiches Preprocessing** wie im Notebook (Downsample1D, ToFrame, Sqrt)
2. ✅ **Aktivitäten während Training** werden gespeichert
3. ✅ **CSV-Format** kompatibel mit Manifold-Analysen
4. ✅ **Pro Layer und Epoche** separate Dateien
5. ✅ **Direkt nutzbar** für PCA, Isomap, UMAP, t-SNE

**Nächste Schritte:**

1. Training starten: `python3 experiments/main_with_activity_logging.py`
2. CSVs analysieren mit Pandas
3. Manifold-Learning anwenden
4. In Dash-App visualisieren!
