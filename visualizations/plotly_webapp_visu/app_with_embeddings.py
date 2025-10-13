#!/usr/bin/env python3
"""
Dash-App zur Visualisierung der vorberechneten Embeddings.
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from pathlib import Path

app = dash.Dash(__name__)

# Verfügbare CSV-Dateien finden
data_dir = Path(__file__).parent.parent.parent / "data"
csv_files = list(data_dir.glob("embeddings_*.csv"))

# Falls keine Dateien vorhanden, Beispiel laden
if not csv_files:
    print("⚠️  Keine Embedding-Dateien gefunden!")
    print("   Bitte zuerst compute_and_save_embeddings.py ausführen")
    # Fallback: Leere App
    available_files = []
else:
    available_files = [f.name for f in csv_files]

print(f"Gefundene Embedding-Dateien: {available_files}")

app.layout = html.Div([
    html.H1("SHD Neuronale Aktivität - Manifold Visualisierung", 
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
            html.Label("Anzahl Samples:", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='n-samples-slider',
                min=1, max=100, value=10, step=1,
                marks={i: str(i) for i in [1, 10, 25, 50, 100]},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'paddingRight': '2%'}),
        
        html.Div([
            html.Label("Visualisierungstyp:", style={'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='viz-type',
                options=[
                    {'label': ' Trajektorien (Linien)', 'value': 'lines'},
                    {'label': ' Scatter (Punkte)', 'value': 'scatter'},
                    {'label': ' Animation über Zeit', 'value': 'animation'}
                ],
                value='lines',
                inline=True
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),
    ], style={'marginBottom': 30}),
    
    dcc.Graph(id='3d-plot', style={'height': '700px'}),
    
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
    [Output('3d-plot', 'figure'),
     Output('stats-output', 'children')],
    [Input('file-dropdown', 'value'),
     Input('label-dropdown', 'value'),
     Input('n-samples-slider', 'value'),
     Input('viz-type', 'value')]
)
def update_graph(filename, label, n_samples, viz_type):
    if not filename:
        return go.Figure(), "Keine Daten verfügbar"
    
    # Daten laden
    filepath = data_dir / filename
    df = pd.read_csv(filepath)
    
    # Filter nach Label
    if label != 'all':
        df = df[df['label'] == label]
    
    if df.empty:
        return go.Figure(), f"Keine Daten für Label {label}"
    
    # Nur erste n_samples nehmen
    sample_ids = df['sample_id'].unique()[:n_samples]
    df = df[df['sample_id'].isin(sample_ids)]
    
    # Methode extrahieren
    method = df['method'].iloc[0]
    
    # Visualisierung erstellen
    if viz_type == 'lines':
        # Trajektorien mit Linien und Farbverlauf
        fig = go.Figure()
        
        # Colormap für Zeitverlauf (neue Matplotlib API)
        try:
            cmap = cm.colormaps['viridis']
        except AttributeError:
            # Fallback für ältere Matplotlib Versionen
            cmap = cm.get_cmap('viridis')
        
        for idx, sample_id in enumerate(sample_ids):
            sample_df = df[df['sample_id'] == sample_id].sort_values('time_bin')
            label_val = sample_df['label'].iloc[0]
            n_points = len(sample_df)
            
            # Normalisiere Zeit für Farbskala
            time_normalized = sample_df['time_bin'].values / sample_df['time_bin'].max()
            
            # Zeichne jedes Segment mit eigener Farbe
            for j in range(n_points - 1):
                # Farbe für dieses Segment (Mittelwert der beiden Punkte)
                t_color = (time_normalized[j] + time_normalized[j+1]) / 2
                rgba = cmap(t_color)
                color_str = f'rgb({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)})'
                
                fig.add_trace(go.Scatter3d(
                    x=sample_df['x'].iloc[j:j+2],
                    y=sample_df['y'].iloc[j:j+2],
                    z=sample_df['z'].iloc[j:j+2],
                    mode='lines',
                    line=dict(color=color_str, width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Punkte separat zeichnen mit Hover-Info
            fig.add_trace(go.Scatter3d(
                x=sample_df['x'],
                y=sample_df['y'],
                z=sample_df['z'],
                mode='markers',
                name=f'Sample {sample_id} (Label {label_val})',
                marker=dict(
                    size=5,
                    color=sample_df['time_bin'].tolist(),
                    colorscale='Viridis',
                    showscale=bool(idx == 0),  # Nur eine Colorbar - expliziter Python bool
                    colorbar=dict(title="Zeit", x=1.1)
                ),
                hovertemplate='<b>Sample %{text}</b><br>Zeit: %{marker.color}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}',
                text=[sample_id] * len(sample_df)
            ))
        
        title = f'{method.upper()} - Trajektorien über Zeit (Farbverlauf)'
    
    elif viz_type == 'scatter':
        # Scatter Plot mit Farbcodierung
        fig = px.scatter_3d(df, 
                           x='x', y='y', z='z',
                           color='label',
                           symbol='label',
                           hover_data=['sample_id', 'time_bin'],
                           color_continuous_scale='Viridis')
        title = f'{method.upper()} - Scatter Plot (farbcodiert nach Label)'
    
    elif viz_type == 'animation':
        # Animation über Zeit
        fig = px.scatter_3d(df, 
                           x='x', y='y', z='z',
                           animation_frame='time_bin',
                           animation_group='sample_id',
                           color='label',
                           hover_data=['sample_id'],
                           range_x=[df['x'].min(), df['x'].max()],
                           range_y=[df['y'].min(), df['y'].max()],
                           range_z=[df['z'].min(), df['z'].max()])
        title = f'{method.upper()} - Animation über Zeit'
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f'{method} Dim 1',
            yaxis_title=f'{method} Dim 2',
            zaxis_title=f'{method} Dim 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        height=700,
        showlegend=(viz_type != 'scatter')
    )
    
    # Statistiken
    stats = html.Div([
        html.P(f"📁 Datei: {filename}"),
        html.P(f"🔬 Methode: {method.upper()}"),
        html.P(f"📊 Anzahl Samples: {len(sample_ids)}"),
        html.P(f"🏷️  Labels: {sorted(df['label'].unique())}"),
        html.P(f"📈 Datenpunkte gesamt: {len(df):,}"),
        html.P(f"⏱️  Zeitbins pro Sample: {df.groupby('sample_id')['time_bin'].count().iloc[0]}"),
        html.Hr(),
        html.P(f"X-Bereich: [{df['x'].min():.2f}, {df['x'].max():.2f}]"),
        html.P(f"Y-Bereich: [{df['y'].min():.2f}, {df['y'].max():.2f}]"),
        html.P(f"Z-Bereich: [{df['z'].min():.2f}, {df['z'].max():.2f}]"),
    ])
    
    return fig, stats

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 Starte Dash-App auf http://127.0.0.1:8050")
    print("="*80 + "\n")
    app.run(debug=True, port=8050)

