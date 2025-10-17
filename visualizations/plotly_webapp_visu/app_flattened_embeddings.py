#!/usr/bin/env python3
"""
Dash-App zur Visualisierung der geflatteten Embeddings (ein Sample = ein Datenpunkt).
Speziell für Klassentrennung optimiert.
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

app = dash.Dash(__name__)

# Verfügbare geflattete CSV-Dateien finden
data_dir = Path(__file__).parent.parent.parent / "data"
manifold_dir = Path(__file__).parent.parent.parent / "manifold_embeddings"

# Suche nach geflatteten Dateien
csv_files = list(data_dir.glob("embeddings_*_flattened_*.csv"))
manifold_files = list(manifold_dir.glob("embeddings_*_flattened_*.csv")) if manifold_dir.exists() else []

all_csv_files = csv_files + manifold_files

# Falls keine Dateien vorhanden
if not all_csv_files:
    print("⚠️  Keine geflatteten Embedding-Dateien gefunden!")
    print("   Bitte zuerst compute_and_save_embeddings.py mit --flattened ausführen")
    available_files = []
else:
    # Erstelle Dict: Dateiname -> vollständiger Pfad
    file_dict = {f.name: f for f in all_csv_files}
    available_files = list(file_dict.keys())

print(f"Gefundene geflattete Embedding-Dateien: {available_files}")

app.layout = html.Div([
    html.H1("SHD Geflattete Embeddings - Klassentrennung", 
            style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    html.Div([
        html.Div([
            html.Label("Embedding-Datei:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='file-dropdown',
                options=[{'label': f, 'value': f} for f in available_files],
                value=available_files[0] if available_files else None,
                style={'width': '100%'}
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'paddingRight': '2%'}),
        
        html.Div([
            html.Label("Label (Ziffer):", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='label-dropdown',
                options=[{'label': 'Alle', 'value': 'all'}] + 
                        [{'label': str(i), 'value': i} for i in range(10)],
                value='all',
                style={'width': '100%'}
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),
    ], style={'marginBottom': 20}),
    
    html.Div([
        html.Div([
            html.Label("Visualisierungstyp:", style={'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='viz-type',
                options=[
                    {'label': ' Scatter 3D (nach Label)', 'value': 'scatter3d'},
                    {'label': ' Scatter 2D (nach Label)', 'value': 'scatter2d'},
                    {'label': ' Scatter Matrix', 'value': 'scatter_matrix'},
                    {'label': ' Box Plot (nach Label)', 'value': 'boxplot'},
                    {'label': ' Violin Plot (nach Label)', 'value': 'violin'}
                ],
                value='scatter3d',
                inline=True
            ),
        ], style={'width': '100%', 'display': 'inline-block'}),
    ], style={'marginBottom': 20}),
    
    html.Div([
        html.Div([
            html.Label("Anzahl Samples:", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='n-samples-slider',
                min=1, max=1000, value=100, step=1,
                marks={i: str(i) for i in [1, 10, 25, 50, 100, 250, 500, 1000]},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'width': '100%', 'display': 'inline-block'}),
    ], style={'marginBottom': 30}),
    
    dcc.Graph(id='main-plot', style={'height': '700px'}),
    
    html.Div([
        html.H3("Statistiken:", style={'color': '#2c3e50'}),
        html.Div(id='stats-output', style={
            'backgroundColor': '#ecf0f1',
            'padding': '15px',
            'borderRadius': '5px',
            'fontFamily': 'monospace'
        })
    ], style={'marginTop': 30})
])

@callback(
    [Output('main-plot', 'figure'),
     Output('stats-output', 'children')],
    [Input('file-dropdown', 'value'),
     Input('label-dropdown', 'value'),
     Input('viz-type', 'value'),
     Input('n-samples-slider', 'value')]
)
def update_graph(filename, label, viz_type, n_samples):
    if not filename:
        return go.Figure(), "Keine Daten verfügbar"
    
    # Daten laden
    filepath = file_dict.get(filename)
    if filepath is None:
        return go.Figure(), f"Datei nicht gefunden: {filename}"
    
    df = pd.read_csv(filepath)
    
    # Filter nach Label
    if label != 'all':
        df = df[df['label'] == label]
    
    if df.empty:
        return go.Figure(), f"Keine Daten für Label: {label}"
    
    # Nur erste n_samples nehmen
    df = df.head(n_samples)
    
    # Methode extrahieren
    method = df['method'].iloc[0]
    
    # Visualisierung erstellen
    if viz_type == 'scatter3d':
        # 3D Scatter Plot
        fig = px.scatter_3d(df, 
                           x='x', y='y', z='z',
                           color='label',
                           symbol='label',
                           hover_data=['sample_id'],
                           title=f'{method.upper()} - 3D Klassentrennung',
                           color_continuous_scale='Viridis')
        
        fig.update_layout(
            scene=dict(
                xaxis_title=f'{method} Dim 1',
                yaxis_title=f'{method} Dim 2',
                zaxis_title=f'{method} Dim 3',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            )
        )
        
    elif viz_type == 'scatter2d':
        # 2D Scatter Plot (erste zwei Dimensionen)
        fig = px.scatter(df, 
                        x='x', y='y',
                        color='label',
                        symbol='label',
                        hover_data=['sample_id', 'z'],
                        title=f'{method.upper()} - 2D Klassentrennung (Dim 1 vs Dim 2)',
                        color_continuous_scale='Viridis')
        
        fig.update_layout(
            xaxis_title=f'{method} Dim 1',
            yaxis_title=f'{method} Dim 2'
        )
        
    elif viz_type == 'scatter_matrix':
        # Scatter Matrix für alle Dimensionen
        dimensions = ['x', 'y', 'z'] if 'z' in df.columns else ['x', 'y']
        fig = px.scatter_matrix(df, 
                               dimensions=dimensions,
                               color='label',
                               title=f'{method.upper()} - Scatter Matrix',
                               color_continuous_scale='Viridis')
        
    elif viz_type == 'boxplot':
        # Box Plot für jede Dimension
        dimensions = ['x', 'y', 'z'] if 'z' in df.columns else ['x', 'y']
        fig = go.Figure()
        
        for i, dim in enumerate(dimensions):
            fig.add_trace(go.Box(
                y=df[dim],
                x=[f'{method} {dim.upper()}'] * len(df),
                name=f'{method} {dim.upper()}',
                boxpoints='outliers',
                jitter=0.3,
                pointpos=-1.8
            ))
        
        fig.update_layout(
            title=f'{method.upper()} - Box Plot Verteilung',
            yaxis_title='Wert',
            showlegend=False
        )
        
    elif viz_type == 'violin':
        # Violin Plot für jede Dimension
        dimensions = ['x', 'y', 'z'] if 'z' in df.columns else ['x', 'y']
        fig = go.Figure()
        
        for i, dim in enumerate(dimensions):
            fig.add_trace(go.Violin(
                y=df[dim],
                x=[f'{method} {dim.upper()}'] * len(df),
                name=f'{method} {dim.upper()}',
                box_visible=True,
                meanline_visible=True
            ))
        
        fig.update_layout(
            title=f'{method.upper()} - Violin Plot Verteilung',
            yaxis_title='Wert',
            showlegend=False
        )
    
    # Statistiken
    stats_items = [
        html.P(f"📁 Datei: {filename}"),
        html.P(f"🔬 Methode: {method.upper()}"),
        html.P(f"📊 Anzahl Samples: {len(df)}"),
        html.P(f"🏷️  Labels: {sorted(df['label'].unique())}"),
        html.P(f"📈 Datenpunkte gesamt: {len(df):,}"),
    ]
    
    # Füge Layer hinzu falls vorhanden
    if 'layer' in df.columns:
        layers = sorted(df['layer'].unique())
        stats_items.append(html.P(f"🧠 Layer: {', '.join(layers)}"))
    
    # Füge Epoche hinzu falls vorhanden
    if 'epoch' in df.columns:
        epochs = sorted(df['epoch'].unique())
        if len(epochs) == 1:
            stats_items.append(html.P(f"📅 Epoche: {epochs[0]}"))
        else:
            stats_items.append(html.P(f"📅 Epochen: {epochs}"))
    
    # Geflattete Daten Info
    if 'flattened' in df.columns:
        stats_items.append(html.P(f"🔧 Geflattete Vektoren: {df['flattened'].iloc[0]}"))
    
    stats_items.extend([
        html.Hr(),
        html.P(f"X-Bereich: [{df['x'].min():.2f}, {df['x'].max():.2f}]"),
        html.P(f"Y-Bereich: [{df['y'].min():.2f}, {df['y'].max():.2f}]"),
    ])
    
    if 'z' in df.columns:
        stats_items.append(html.P(f"Z-Bereich: [{df['z'].min():.2f}, {df['z'].max():.2f}]"))
    
    # Label-Verteilung
    label_counts = df['label'].value_counts().sort_index()
    stats_items.append(html.Hr())
    stats_items.append(html.P("📊 Label-Verteilung:"))
    for label_val, count in label_counts.items():
        stats_items.append(html.P(f"   Label {label_val}: {count} Samples"))
    
    stats = html.Div(stats_items)
    
    return fig, stats

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 Starte Geflattete Embeddings Dash-App auf http://127.0.0.1:8051")
    print("="*80 + "\n")
    app.run(debug=True, port=8051)

