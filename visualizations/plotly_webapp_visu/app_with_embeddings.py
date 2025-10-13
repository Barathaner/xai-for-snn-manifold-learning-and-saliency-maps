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
        
        for idx, sample_id in enumerate(sample_ids):
            sample_df = df[df['sample_id'] == sample_id].sort_values('time_bin')
            label_val = sample_df['label'].iloc[0]
            
            # Linie UND Punkte zusammen - Farbverlauf auf beiden
            fig.add_trace(go.Scatter3d(
                x=sample_df['x'],
                y=sample_df['y'],
                z=sample_df['z'],
                mode='lines+markers',
                name=f'Sample {sample_id} (Label {label_val})',
                line=dict(
                    color=sample_df['time_bin'].tolist(),  # Farbverlauf auf Linie!
                    colorscale='Viridis',
                    width=3
                ),
                marker=dict(
                    size=5,
                    color=sample_df['time_bin'].tolist(),
                    colorscale='Viridis',
                    showscale=bool(idx == 0),  # Nur eine Colorbar
                    colorbar=dict(title="Zeit", x=1.1),
                    line=dict(color='white', width=0.5)
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
        # Animation: Punkt läuft über komplette Linie mit Farbverlauf
        fig = go.Figure()
        
        all_time_bins = sorted(df['time_bin'].unique())
        
        # 1. Zeichne alle kompletten Linien mit Farbverlauf (statisch)
        for sample_id in sample_ids:
            sample_df = df[df['sample_id'] == sample_id].sort_values('time_bin')
            label_val = sample_df['label'].iloc[0]
            
            # Linie mit Farbverlauf
            fig.add_trace(go.Scatter3d(
                x=sample_df['x'],
                y=sample_df['y'],
                z=sample_df['z'],
                mode='lines',
                line=dict(
                    color=sample_df['time_bin'].tolist(),  # Farbverlauf!
                    colorscale='Viridis',
                    width=2
                ),
                showlegend=False,
                hoverinfo='skip',
                name=f'Line_{sample_id}'
            ))
        
        # 2. Füge initiale Punkt-Trace hinzu
        initial_points_x, initial_points_y, initial_points_z = [], [], []
        initial_colors = []
        initial_hover = []
        
        for sample_id in sample_ids:
            sample_point = df[(df['sample_id'] == sample_id) & (df['time_bin'] == all_time_bins[0])]
            if not sample_point.empty:
                initial_points_x.append(sample_point['x'].iloc[0])
                initial_points_y.append(sample_point['y'].iloc[0])
                initial_points_z.append(sample_point['z'].iloc[0])
                label_val = sample_point['label'].iloc[0]
                initial_colors.append(label_val)
                initial_hover.append(f'Sample {sample_id}<br>Zeit: {all_time_bins[0]}<br>Label: {label_val}')
        
        fig.add_trace(go.Scatter3d(
            x=initial_points_x,
            y=initial_points_y,
            z=initial_points_z,
            mode='markers',
            marker=dict(
                size=12,
                color=initial_colors,
                colorscale='Turbo',
                line=dict(color='white', width=2),
                showscale=False
            ),
            text=initial_hover,
            hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
            name='Aktuelle Position'
        ))
        
        # 3. Erstelle Animations-Frames - Jeder Frame enthält ALLE Traces
        frames = []
        
        for time_bin in all_time_bins:
            frame_data = []
            
            # Füge alle Linien hinzu (gleich bleibend)
            for sample_id in sample_ids:
                sample_df = df[df['sample_id'] == sample_id].sort_values('time_bin')
                
                frame_data.append(go.Scatter3d(
                    x=sample_df['x'],
                    y=sample_df['y'],
                    z=sample_df['z'],
                    mode='lines',
                    line=dict(
                        color=sample_df['time_bin'].tolist(),
                        colorscale='Viridis',
                        width=2
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Füge die Punkte für diesen Zeitschritt hinzu
            points_x, points_y, points_z = [], [], []
            colors = []
            hover_texts = []
            
            for sample_id in sample_ids:
                sample_point = df[(df['sample_id'] == sample_id) & (df['time_bin'] == time_bin)]
                if not sample_point.empty:
                    points_x.append(sample_point['x'].iloc[0])
                    points_y.append(sample_point['y'].iloc[0])
                    points_z.append(sample_point['z'].iloc[0])
                    label_val = sample_point['label'].iloc[0]
                    colors.append(label_val)
                    hover_texts.append(f'Sample {sample_id}<br>Zeit: {time_bin}<br>Label: {label_val}')
            
            frame_data.append(go.Scatter3d(
                x=points_x,
                y=points_y,
                z=points_z,
                mode='markers',
                marker=dict(
                    size=12,
                    color=colors,
                    colorscale='Turbo',
                    line=dict(color='white', width=2),
                    showscale=False
                ),
                text=hover_texts,
                hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            ))
            
            frames.append(go.Frame(data=frame_data, name=str(time_bin)))
        
        fig.frames = frames
        
        # Animation-Einstellungen
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': True,
                'buttons': [
                    {'label': '▶ Play', 'method': 'animate', 
                     'args': [None, {'frame': {'duration': 100, 'redraw': True},
                                    'fromcurrent': True, 'mode': 'immediate', 
                                    'transition': {'duration': 0}}]},
                    {'label': '⏸ Pause', 'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 
                                      'mode': 'immediate'}]}
                ],
                'x': 0.1, 'y': 1.15
            }],
            sliders=[{
                'steps': [{'args': [[f.name], {'frame': {'duration': 0, 'redraw': True}, 
                                              'mode': 'immediate', 'transition': {'duration': 0}}],
                          'label': f'Zeit: {f.name}', 'method': 'animate'} for f in frames],
                'active': 0,
                'y': -0.1,
                'len': 0.9,
                'x': 0.1
            }]
        )
        
        title = f'{method.upper()} - Animation: Punkt läuft über Trajektorie'
    
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

