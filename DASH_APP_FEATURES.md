# 🎨 Dash-App Features

## Neue Filter (Activity-Log Support)

Die Dash-App wurde erweitert um **Epoche- und Layer-Filter** für Activity-Log-Embeddings!

### 📊 Alle verfügbaren Filter:

1. **Embedding-Datei** 📁
   - Wählt zwischen verschiedenen CSV-Dateien
   - Findet automatisch Dateien in `./data/` und `./manifold_embeddings/`

2. **Label (Ziffer)** 🏷️
   - Filtert nach Klassen-Label (0-9)
   - Option "Alle" zeigt alle Labels

3. **Layer** 🧠 (**NEU!**)
   - Wählt zwischen Hidden und Output Layer
   - Nur bei Activity-Log-Embeddings verfügbar
   - Optionen:
     - Alle
     - Hidden
     - Output

4. **Epoche** 📅 (**NEU!**)
   - Wählt spezifische Trainings-Epoche
   - **Dynamisch**: Zeigt nur verfügbare Epochen aus der gewählten Datei
   - Nur bei Activity-Log-Embeddings verfügbar
   - Optionen: Alle, Epoche 1, Epoche 2, ...

5. **Anzahl Samples** 📊
   - Slider: 1-100 Samples
   - Limitiert die angezeigten Trajektorien

6. **Visualisierungstyp** 🎬
   - Trajektorien (Linien): Mit zeitlichem Farbverlauf
   - Scatter (Punkte): Alle Punkte einzeln
   - Animation: Punkt läuft entlang Trajektorie

## 🎯 Anwendungsbeispiele

### Beispiel 1: Vergleich über Epochen

```
1. Wähle: "embeddings_pca_n5_c3.csv" (Activity-Log)
2. Layer: "Hidden"
3. Epoche: "Epoche 1"
4. Label: "5"
→ Sehe Manifold für Label 5 in Epoche 1

Dann ändere:
3. Epoche: "Epoche 6"
→ Vergleiche wie sich das Manifold über Training entwickelt!
```

### Beispiel 2: Layer-Vergleich

```
1. Wähle: "embeddings_pca_n5_c3.csv"
2. Epoche: "Epoche 6"
3. Layer: "Hidden"
4. Label: "Alle"
→ Sehe Hidden Layer Aktivität

Dann ändere:
3. Layer: "Output"
→ Vergleiche Output Layer!
```

### Beispiel 3: Original-Daten vs. Activity-Logs

```
1. Wähle: "embeddings_pca_train_n5_c3.csv" (Original)
   → Kein Layer/Epoche Filter (nicht verfügbar)
   
2. Wähle: "embeddings_pca_n5_c3.csv" (Activity-Log)
   → Layer und Epoche Filter erscheinen automatisch!
```

## 🔄 Dynamisches Verhalten

### Epoche-Dropdown ist dynamisch:

- **Keine Epochen in Daten**: Zeigt nur "Alle"
- **Mit Epochen**: Zeigt "Alle, Epoche 1, Epoche 2, ..." basierend auf Datei

### Filter sind intelligent:

- Wenn keine Daten für Filter-Kombination: Zeigt Info-Nachricht
- Statistiken zeigen nur relevante Info (Layer/Epoche nur wenn vorhanden)

## 📈 Statistik-Anzeige

Die Statistik-Box zeigt jetzt:

```
📁 Datei: embeddings_pca_n5_c3.csv
🔬 Methode: PCA
🧠 Layer: hidden                    ← NEU (nur wenn vorhanden)
📅 Epoche: 6                        ← NEU (nur wenn vorhanden)
📊 Anzahl Samples: 10
🏷️ Labels: [0, 1, 2, 5, ...]
📈 Datenpunkte gesamt: 800
⏱️ Zeitbins pro Sample: 80
---
X-Bereich: [-8.55, 4.25]
Y-Bereich: [-1.47, 6.39]
Z-Bereich: [-2.03, 2.12]
```

## 🚀 Starten

```bash
cd /home/karl/git/xai-for-snn-manifold-learning-and-saliency-maps
source venv/bin/activate
python3 visualizations/plotly_webapp_visu/app_with_embeddings.py
```

Dann öffnen: **http://127.0.0.1:8050**

## 🎨 Screenshots-Workflow

### Workflow: Training Evolution visualisieren

1. **Datei wählen**: Activity-Log-Embedding (z.B. `embeddings_pca_n5_c3.csv`)
2. **Layer**: Hidden
3. **Epoche**: Epoche 1
4. **Label**: 5
5. **Viz**: Trajektorien
   → Screenshot 1

6. **Epoche ändern**: Epoche 3
   → Screenshot 2

7. **Epoche ändern**: Epoche 6
   → Screenshot 3

→ **Vergleich**: Wie sich Manifold über Training entwickelt!

### Workflow: Layer-Vergleich

1. **Layer**: Hidden
   → Screenshot Hidden Layer

2. **Layer**: Output
   → Screenshot Output Layer

→ **Vergleich**: Unterschiede zwischen Layern!

## 💡 Tipps

### Kombination der Filter:

- **Epoche 1 + Hidden + Label 5**: Sehr spezifisch
- **Alle Epochen + Hidden + Label 5**: Vergleich über Zeit
- **Epoche 6 + Alle Layer + Label 5**: Layer-Vergleich

### Performance:

- Weniger Samples (10-20): Schnellere Visualisierung
- Mehr Samples (50-100): Besserer Überblick

### Fehlermeldungen:

```
"Keine Daten für Filter: Label=5, Layer=output, Epoche=1"
```
→ Diese Kombination existiert nicht in den Daten
→ Ändere einen Filter!

## 🔧 Technische Details

### Filter-Logik:

```python
# Filter werden nacheinander angewendet:
if label != 'all':
    df = df[df['label'] == label]

if layer != 'all' and 'layer' in df.columns:
    df = df[df['layer'] == layer]

if epoch != 'all' and 'epoch' in df.columns:
    df = df[df['epoch'] == epoch]
```

### Epoche-Optionen werden automatisch geladen:

```python
@callback(
    Output('epoch-dropdown', 'options'),
    Input('file-dropdown', 'value')
)
def update_epoch_options(filename):
    # Liest Datei und findet verfügbare Epochen
    if 'epoch' in df.columns:
        epochs = sorted(df['epoch'].unique())
        return [{'label': f'Epoche {e}', 'value': e} for e in epochs]
```

## ✅ Zusammenfassung

**Neue Features:**
- ✅ Layer-Filter (Hidden/Output)
- ✅ Epoche-Filter (dynamisch basierend auf Datei)
- ✅ Erweiterte Statistiken
- ✅ Intelligente Fehlermeldungen
- ✅ Automatische Erkennung von Activity-Log-Daten

**Kompatibilität:**
- ✅ Original-Embeddings (ohne Layer/Epoche)
- ✅ Activity-Log-Embeddings (mit Layer/Epoche)
- ✅ Beide Dateitypen funktionieren parallel!

🎉 **Perfekt für Analyse der Training-Evolution!**

