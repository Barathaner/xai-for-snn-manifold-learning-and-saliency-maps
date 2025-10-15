#!/bin/bash
# Vollständige Pipeline: Activity-Logs → Embeddings → Visualisierung

echo "================================================================================
VOLLSTÄNDIGE PIPELINE: Activity-Logs zu Manifold-Visualisierung
================================================================================
"

# 1. Verarbeite Activity-Logs zu Embeddings
echo "
Schritt 1: Verarbeite Activity-Logs..."
echo "--------------------------------------------------------------------------------"

# Hidden Layer
python3 process_activity_logs.py \
    --activity-dir ./activity_logs \
    --output-dir ./manifold_embeddings \
    --method isomap \
    --n-neighbors 5 \
    --n-components 3 \
    --layer hidden

python3 process_activity_logs.py \
    --activity-dir ./activity_logs \
    --output-dir ./manifold_embeddings \
    --method pca \
    --n-components 3 \
    --layer hidden

echo "
Schritt 2: Verarbeite auch Output Layer..."
echo "--------------------------------------------------------------------------------"

# Output Layer
python3 process_activity_logs.py \
    --activity-dir ./activity_logs \
    --output-dir ./manifold_embeddings \
    --method isomap \
    --n-neighbors 5 \
    --n-components 3 \
    --layer output

python3 process_activity_logs.py \
    --activity-dir ./activity_logs \
    --output-dir ./manifold_embeddings \
    --method pca \
    --n-components 3 \
    --layer output

echo "
================================================================================
FERTIG! Embeddings erstellt.
================================================================================

Gespeichert in: ./manifold_embeddings/

Jetzt starten Sie die Dash-App:
    python3 visualizations/plotly_webapp_visu/app_with_embeddings.py

Die App findet automatisch alle Embedding-Dateien!
"

