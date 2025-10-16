#!/usr/bin/env python3
"""
Plotly t-SNE Visualisierung für Jupyter Notebook
Kopieren Sie diesen Code in eine neue Zelle in Ihrem Notebook
"""

# Zelle 1: Imports und Setup
import tonic
import tonic.transforms as transforms
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from torch.utils.data import Subset
import ipywidgets as widgets
from IPython.display import display

class Downsample1D:
    def __init__(self, spatial_factor=0.1):
        self.spatial_factor = spatial_factor

    def __call__(self, events):
        events = events.copy()
        events['x'] = (events['x'] * self.spatial_factor).astype(events['x'].dtype)
        return events

# Zelle 2: Daten laden
print("Lade SHD Datensatz...")
trainset = tonic.datasets.SHD(save_to="./data", train=True)

# Filter auf Labels 0-9
label_range = set(range(0, 10))
filtered_indices = [
    i for i in range(len(trainset)) if trainset[i][1] in label_range
]
trainset = Subset(trainset, filtered_indices)

# Transform Pipeline
trans = transforms.Compose([
    Downsample1D(spatial_factor=0.1),   
    tonic.transforms.ToFrame(sensor_size=(70, 1, 1), n_time_bins=80),
])

print(f"Dataset Größe: {len(trainset)} samples")

# Zelle 3: Plotly Visualisierung Funktion
def create_plotly_tsne_visualization(target_label=5, n_samples=2, perplexity=30, learning_rate=200, n_components=3):
    """
    Erstellt eine interaktive Plotly 3D-Visualisierung der t-SNE Trajektorien
    """
    
    # Daten sammeln
    trajectories = []
    
    count = 0
    for i in range(len(trainset)):
        events, label = trainset[i]
        if label != target_label:
            continue
        count += 1
        if count > n_samples:
            break

        frames = trans(events)
        vec = frames[:, 0, :]  # shape: (time_bins, neurons)

        # t-SNE Einbettung
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            random_state=42,
            verbose=0
        )
        emb = tsne.fit_transform(vec)
        
        # Trajektorie als DataFrame
        traj_df = pd.DataFrame({
            'x': emb[:, 0],
            'y': emb[:, 1], 
            'z': emb[:, 2] if n_components > 2 else [0] * len(emb),
            'time_bin': range(len(emb)),
            'sample_id': i,
            'label': label
        })
        trajectories.append(traj_df)
    
    # Plotly Figure erstellen
    fig = go.Figure()
    
    # Für jede Trajektorie
    for i, traj_df in enumerate(trajectories):
        
        # Linie mit Farbverlauf (Zeit)
        fig.add_trace(go.Scatter3d(
            x=traj_df['x'],
            y=traj_df['y'],
            z=traj_df['z'],
            mode='lines+markers',
            line=dict(
                color=traj_df['time_bin'],
                colorscale='Viridis',
                width=6,
                showscale=True if i == 0 else False,
                colorbar=dict(title="Zeit-Bin") if i == 0 else None
            ),
            marker=dict(
                size=4,
                color=traj_df['time_bin'],
                colorscale='Viridis',
                showscale=False
            ),
            name=f'Sample {i}',
            hovertemplate='<b>Sample %{customdata[0]}</b><br>' +
                         'Zeit-Bin: %{customdata[1]}<br>' +
                         'X: %{x:.2f}<br>' +
                         'Y: %{y:.2f}<br>' +
                         'Z: %{z:.2f}<extra></extra>',
            customdata=list(zip(traj_df['sample_id'], traj_df['time_bin']))
        ))
        
        # Start-Punkt (rot)
        fig.add_trace(go.Scatter3d(
            x=[traj_df['x'].iloc[0]],
            y=[traj_df['y'].iloc[0]],
            z=[traj_df['z'].iloc[0]],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='circle',
                line=dict(width=2, color='black')
            ),
            name=f'Start {i}' if i == 0 else None,
            showlegend=True if i == 0 else False,
            hovertemplate='<b>Start-Punkt</b><br>' +
                         'Sample: %{customdata[0]}<br>' +
                         'Zeit-Bin: %{customdata[1]}<extra></extra>',
            customdata=[[traj_df['sample_id'].iloc[0], traj_df['time_bin'].iloc[0]]]
        ))
        
        # End-Punkt (gelb)
        fig.add_trace(go.Scatter3d(
            x=[traj_df['x'].iloc[-1]],
            y=[traj_df['y'].iloc[-1]],
            z=[traj_df['z'].iloc[-1]],
            mode='markers',
            marker=dict(
                size=6,
                color='yellow',
                symbol='diamond',
                line=dict(width=1, color='black')
            ),
            name=f'Ende {i}' if i == 0 else None,
            showlegend=True if i == 0 else False,
            hovertemplate='<b>End-Punkt</b><br>' +
                         'Sample: %{customdata[0]}<br>' +
                         'Zeit-Bin: %{customdata[1]}<extra></extra>',
            customdata=[[traj_df['sample_id'].iloc[-1], traj_df['time_bin'].iloc[-1]]]
        ))
    
    # Layout anpassen
    fig.update_layout(
        title=f't-SNE Trajektorien - Label {target_label} (rot→gelb = Zeit)',
        scene=dict(
            xaxis_title='t-SNE Dimension 1',
            yaxis_title='t-SNE Dimension 2',
            zaxis_title='t-SNE Dimension 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

# Zelle 4: Interaktive Widgets
def interactive_plotly_tsne():
    """
    Erstellt interaktive Widgets für die Plotly t-SNE Visualisierung
    """
    
    # Widgets erstellen
    label_widget = widgets.IntSlider(
        value=5, min=0, max=9, step=1,
        description='Label:', style={'description_width': 'initial'}
    )
    
    samples_widget = widgets.IntSlider(
        value=2, min=1, max=10, step=1,
        description='Samples:', style={'description_width': 'initial'}
    )
    
    perplexity_widget = widgets.FloatSlider(
        value=30, min=5, max=50, step=1,
        description='Perplexity:', style={'description_width': 'initial'}
    )
    
    learning_rate_widget = widgets.FloatSlider(
        value=200, min=10, max=1000, step=10,
        description='Lernrate:', style={'description_width': 'initial'}
    )
    
    components_widget = widgets.IntSlider(
        value=3, min=2, max=3, step=1,
        description='Komponenten:', style={'description_width': 'initial'}
    )
    
    # Output Widget für die Visualisierung
    output_widget = widgets.Output()
    
    def update_plot(label, samples, perplexity, learning_rate, components):
        with output_widget:
            output_widget.clear_output(wait=True)
            fig = create_plotly_tsne_visualization(
                target_label=label,
                n_samples=samples,
                perplexity=perplexity,
                learning_rate=learning_rate,
                n_components=components
            )
            fig.show()
    
    # Interaktive Verbindung
    interactive_widget = widgets.interactive(
        update_plot,
        label=label_widget,
        samples=samples_widget,
        perplexity=perplexity_widget,
        learning_rate=learning_rate_widget,
        components=components_widget
    )
    
    # Widgets anzeigen
    display(widgets.VBox([
        widgets.HBox([label_widget, samples_widget]),
        widgets.HBox([perplexity_widget, learning_rate_widget, components_widget]),
        output_widget
    ]))
    
    return interactive_widget

# Zelle 5: Einfache Visualisierung (ohne Widgets)
def simple_plotly_tsne(target_label=5, n_samples=2, perplexity=30, learning_rate=200, n_components=3):
    """
    Einfache Plotly t-SNE Visualisierung ohne Widgets
    """
    fig = create_plotly_tsne_visualization(
        target_label=target_label,
        n_samples=n_samples,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_components=n_components
    )
    fig.show()
    return fig

print("✅ Plotly t-SNE Visualisierung bereit!")
print("\nVerwendung:")
print("1. Für interaktive Widgets: interactive_plotly_tsne()")
print("2. Für einfache Visualisierung: simple_plotly_tsne()")
print("\nBeispiel:")
print("simple_plotly_tsne(target_label=5, n_samples=3, perplexity=30)")
