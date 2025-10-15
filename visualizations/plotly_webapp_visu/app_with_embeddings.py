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
manifold_dir = Path(__file__).parent.parent.parent / "manifold_embeddings"

# Suche in beiden Verzeichnissen
csv_files = list(data_dir.glob("embeddings_*.csv"))
manifold_files = list(manifold_dir.glob("embeddings_*.csv")) if manifold_dir.exists() else []

all_csv_files = csv_files + manifold_files

# Falls keine Dateien vorhanden
if not all_csv_files:
    print("⚠️  Keine Embedding-Dateien gefunden!")
    print("   Bitte zuerst compute_and_save_embeddings.py oder process_activity_logs.py ausführen")
    available_files = []
else:
    # Erstelle Dict: Dateiname -> vollständiger Pfad
    file_dict = {f.name: f for f in all_csv_files}
    available_files = list(file_dict.keys())

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
        ], style={'width': '48%', 'display': 'inline-block'}, id='label-filter-container'),
    ], style={'marginBottom': 20}),
    
    html.Div([
        html.Div([
            html.Label("Layer:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='layer-dropdown',
                options=[
                    {'label': 'Alle', 'value': 'all'},
                    {'label': 'Hidden', 'value': 'hidden'},
                    {'label': 'Output', 'value': 'output'}
                ],
                value='all',
                style={'width': '100%'}
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'paddingRight': '2%'}),
        
        html.Div([
            html.Label("Epoche:", id='epoch-label', style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='epoch-slider',
                min=0, max=100, value=0, step=1,
                marks={i: str(i) for i in range(0, 101, 10)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'width': '48%', 'display': 'inline-block'}, id='epoch-filter-container'),
    ], style={'marginBottom': 20}),
    
    html.Div([
        html.Div([
            html.Label("Anzahl Samples:", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='n-samples-slider',
                min=1, max=1000, value=10, step=1,
                marks={i: str(i) for i in [1, 10, 25, 50, 100, 250, 500, 1000]},
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
                    {'label': ' Animation über Zeit', 'value': 'animation'},
                    {'label': ' Animation über Epochen', 'value': 'epoch_animation'}
                ],
                value='lines',
                inline=True
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),
    ], style={'marginBottom': 30}),
    
    # Info-Box für Epochen-Animation
    html.Div(id='epoch-animation-info', style={'marginBottom': 20}),
    
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
    [Output('epoch-slider', 'marks'),
     Output('epoch-slider', 'max'),
     Output('epoch-slider', 'value')],
    Input('file-dropdown', 'value')
)
def update_epoch_slider(filename):
    """Aktualisiert die Epochen-Slider-Optionen basierend auf der ausgewählten Datei."""
    if not filename:
        return {0: '0'}, 0, 0
    
    try:
        filepath = file_dict.get(filename)
        if filepath is None:
            return {0: '0'}, 0, 0
        
        df = pd.read_csv(filepath)
        
        # Prüfe ob 'epoch' Spalte existiert
        if 'epoch' in df.columns:
            epochs = sorted(df['epoch'].unique())
            if epochs:
                # Konvertiere numpy.int64 zu Python int
                epochs = [int(epoch) for epoch in epochs]
                max_epoch = int(max(epochs))
                # Erstelle Marks alle 5 Epochen oder bei wichtigen Epochen
                marks = {}
                for i in range(0, max_epoch + 1, max(1, max_epoch // 10)):
                    marks[i] = str(i)
                # Füge die tatsächlichen Epochen hinzu
                for epoch in epochs:
                    marks[epoch] = str(epoch)
                return marks, max_epoch, epochs[0]
            else:
                return {0: '0'}, 0, 0
        else:
            return {0: '0'}, 0, 0
        
    except:
        return {0: '0'}, 0, 0

@callback(
    Output('epoch-label', 'children'),
    Input('epoch-slider', 'value')
)
def update_epoch_label(epoch_value):
    """Aktualisiert das Epochen-Label mit dem aktuellen Wert."""
    return f"Epoche: {epoch_value}"

@callback(
    [Output('epoch-filter-container', 'style'),
     Output('label-filter-container', 'style'),
     Output('epoch-animation-info', 'children')],
    Input('viz-type', 'value')
)
def toggle_filters(viz_type):
    """Deaktiviert nur den Epochen-Filter bei Epochen-Animation."""
    if viz_type == 'epoch_animation':
        # Nur Epochen-Filter verstecken, Label-Filter bleibt sichtbar
        info_box = html.Div([
            html.P("ℹ️ Bei Epochen-Animation können Sie ein Label auswählen, um dessen Entwicklung über alle Epochen zu sehen.",
                   style={'backgroundColor': '#e8f4fd', 'padding': '10px', 'borderRadius': '5px', 
                          'border': '1px solid #bee5eb', 'color': '#0c5460', 'margin': 0})
        ])
        return {'width': '48%', 'display': 'none'}, {'width': '48%', 'display': 'inline-block'}, info_box
    else:
        # Beide Filter anzeigen bei anderen Visualisierungen
        return {'width': '48%', 'display': 'inline-block'}, {'width': '48%', 'display': 'inline-block'}, ""

@callback(
    [Output('3d-plot', 'figure'),
     Output('stats-output', 'children')],
    [Input('file-dropdown', 'value'),
     Input('label-dropdown', 'value'),
     Input('layer-dropdown', 'value'),
     Input('epoch-slider', 'value'),
     Input('n-samples-slider', 'value'),
     Input('viz-type', 'value')]
)
def update_graph(filename, label, layer, epoch, n_samples, viz_type):
    if not filename:
        return go.Figure(), "Keine Daten verfügbar"
    
    # Daten laden (aus file_dict, das beide Verzeichnisse enthält)
    filepath = file_dict.get(filename)
    if filepath is None:
        return go.Figure(), f"Datei nicht gefunden: {filename}"
    
    df = pd.read_csv(filepath)
    
    # Filter nach Label
    if label != 'all':
        df = df[df['label'] == label]
    
    # Filter nach Layer (falls Spalte vorhanden)
    if layer != 'all' and 'layer' in df.columns:
        df = df[df['layer'] == layer]
    
    # Filter nach Epoche (falls Spalte vorhanden) - NICHT bei Epochen-Animation
    if epoch != 0 and 'epoch' in df.columns and viz_type != 'epoch_animation':
        df = df[df['epoch'] == epoch]
    
    if df.empty:
        filter_info = []
        if label != 'all':
            filter_info.append(f"Label={label}")
        if layer != 'all':
            filter_info.append(f"Layer={layer}")
        if epoch != 0 and viz_type != 'epoch_animation':
            filter_info.append(f"Epoche={epoch}")
        return go.Figure(), f"Keine Daten für Filter: {', '.join(filter_info)}"
    
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
    
    elif viz_type == 'epoch_animation':
        # Epochen-Animation: Zeigt Entwicklung über Epochen
        if 'epoch' not in df.columns:
            return go.Figure(), "Epochen-Animation benötigt 'epoch' Spalte in den Daten"
        
        fig = go.Figure()
        
        all_epochs = sorted(df['epoch'].unique())
        if len(all_epochs) < 2:
            return go.Figure(), f"Epochen-Animation benötigt mindestens 2 Epochen. Gefunden: {all_epochs}"
        
        # Debug: Zeige verfügbare Epochen
        print(f"DEBUG: Verfügbare Epochen für Animation: {all_epochs}")
        print(f"DEBUG: Erste Epoche (initial_epoch): {min(all_epochs)}")
        print(f"DEBUG: Sample IDs: {sample_ids}")
        print(f"DEBUG: Labels in Daten: {sorted(df['label'].unique())}")
        
        # 1. Zeichne alle Trajektorien für alle Epochen (statisch, transparent)
        for epoch in all_epochs:
            epoch_df = df[df['epoch'] == epoch]
            for sample_id in sample_ids:
                sample_df = epoch_df[epoch_df['sample_id'] == sample_id].sort_values('time_bin')
                if not sample_df.empty:
                    label_val = sample_df['label'].iloc[0]
                    
                    # Transparente Linien für alle Epochen
                    fig.add_trace(go.Scatter3d(
                        x=sample_df['x'],
                        y=sample_df['y'],
                        z=sample_df['z'],
                        mode='lines',
                        line=dict(
                            color=sample_df['time_bin'].tolist(),
                            colorscale='Viridis',
                            width=1
                        ),
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo='skip',
                        name=f'Line_{sample_id}_epoch_{epoch}'
                    ))
        
        # 2. Füge initiale Trajektorien hinzu (erste Epoche, sichtbar)
        # Verwende die kleinste Epoche (normalerweise 0)
        initial_epoch = min(all_epochs)
        initial_epoch_df = df[df['epoch'] == initial_epoch]
        
        for idx, sample_id in enumerate(sample_ids):
            sample_df = initial_epoch_df[initial_epoch_df['sample_id'] == sample_id].sort_values('time_bin')
            if not sample_df.empty:
                label_val = sample_df['label'].iloc[0]
                
                fig.add_trace(go.Scatter3d(
                    x=sample_df['x'],
                    y=sample_df['y'],
                    z=sample_df['z'],
                    mode='lines+markers',
                    name=f'Sample {sample_id} (Label {label_val})',
                    line=dict(
                        color=sample_df['time_bin'].tolist(),
                        colorscale='Viridis',
                        width=4
                    ),
                    marker=dict(
                        size=6,
                        color=sample_df['time_bin'].tolist(),
                        colorscale='Viridis',
                        showscale=bool(idx == 0),
                        colorbar=dict(title="Zeit", x=1.1),
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate=f'<b>Sample {sample_id}</b><br>Epoche: {initial_epoch}<br>Zeit: %{{marker.color}}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}',
                    visible=True
                ))
        
        # 3. Erstelle Animations-Frames für Epochen (in aufsteigender Reihenfolge)
        frames = []
        
        for epoch in sorted(all_epochs):
            frame_data = []
            
            # Füge alle transparenten Linien hinzu (gleich bleibend)
            for ep in all_epochs:
                epoch_df = df[df['epoch'] == ep]
                for sample_id in sample_ids:
                    sample_df = epoch_df[epoch_df['sample_id'] == sample_id].sort_values('time_bin')
                    if not sample_df.empty:
                        frame_data.append(go.Scatter3d(
                            x=sample_df['x'],
                            y=sample_df['y'],
                            z=sample_df['z'],
                            mode='lines',
                            line=dict(
                                color=sample_df['time_bin'].tolist(),
                                colorscale='Viridis',
                                width=1
                            ),
                            opacity=0.3,
                            showlegend=False,
                            hoverinfo='skip'
                        ))
            
            # Füge die sichtbaren Trajektorien für diese Epoche hinzu
            epoch_df = df[df['epoch'] == epoch]
            for idx, sample_id in enumerate(sample_ids):
                sample_df = epoch_df[epoch_df['sample_id'] == sample_id].sort_values('time_bin')
                if not sample_df.empty:
                    label_val = sample_df['label'].iloc[0]
                    
                    frame_data.append(go.Scatter3d(
                        x=sample_df['x'],
                        y=sample_df['y'],
                        z=sample_df['z'],
                        mode='lines+markers',
                        line=dict(
                            color=sample_df['time_bin'].tolist(),
                            colorscale='Viridis',
                            width=4
                        ),
                        marker=dict(
                            size=6,
                            color=sample_df['time_bin'].tolist(),
                            colorscale='Viridis',
                            showscale=bool(idx == 0),
                            colorbar=dict(title="Zeit", x=1.1),
                            line=dict(color='white', width=1)
                        ),
                        hovertemplate=f'<b>Sample {sample_id}</b><br>Epoche: {epoch}<br>Zeit: %{{marker.color}}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}'
                    ))
            
            frames.append(go.Frame(data=frame_data, name=str(epoch)))
        
        fig.frames = frames
        
        # Animation-Einstellungen für Epochen
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': True,
                'buttons': [
                    {'label': '▶ Play Epochen', 'method': 'animate', 
                     'args': [None, {'frame': {'duration': 500, 'redraw': True},
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
                          'label': f'Epoche: {f.name}', 'method': 'animate'} for f in frames],
                'active': 0,
                'y': -0.1,
                'len': 0.9,
                'x': 0.1
            }]
        )
        
        title = f'{method.upper()} - Animation: Entwicklung über Epochen'
    
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
    stats_items = [
        html.P(f"📁 Datei: {filename}"),
        html.P(f"🔬 Methode: {method.upper()}"),
    ]
    
    # Füge Layer hinzu falls vorhanden
    if 'layer' in df.columns:
        layers = sorted(df['layer'].unique())
        stats_items.append(html.P(f"🧠 Layer: {', '.join(layers)}"))
    
    # Füge Epoche hinzu falls vorhanden
    if 'epoch' in df.columns:
        epochs = sorted(df['epoch'].unique())
        if viz_type == 'epoch_animation':
            stats_items.append(html.P(f"📅 Epochen für Animation: {epochs}"))
        elif len(epochs) == 1:
            stats_items.append(html.P(f"📅 Epoche: {epochs[0]}"))
        else:
            stats_items.append(html.P(f"📅 Epochen: {epochs}"))
    
    stats_items.extend([
        html.P(f"📊 Anzahl Samples: {len(sample_ids)}"),
        html.P(f"🏷️  Labels: {sorted(df['label'].unique())}"),
        html.P(f"📈 Datenpunkte gesamt: {len(df):,}"),
        html.P(f"⏱️  Zeitbins pro Sample: {df.groupby('sample_id')['time_bin'].count().iloc[0]}"),
        html.Hr(),
        html.P(f"X-Bereich: [{df['x'].min():.2f}, {df['x'].max():.2f}]"),
        html.P(f"Y-Bereich: [{df['y'].min():.2f}, {df['y'].max():.2f}]"),
        html.P(f"Z-Bereich: [{df['z'].min():.2f}, {df['z'].max():.2f}]"),
    ])
    
    # Debug-Info für Epochen-Animation
    if viz_type == 'epoch_animation':
        stats_items.append(html.Hr())
        stats_items.append(html.P(f"🔍 DEBUG: Animation startet bei Epoche {min(all_epochs)}"))
        stats_items.append(html.P(f"🔍 DEBUG: Alle Epochen: {all_epochs}"))
    
    stats = html.Div(stats_items)
    
    return fig, stats

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 Starte Dash-App auf http://127.0.0.1:8050")
    print("="*80 + "\n")
    app.run(debug=True, port=8050)

