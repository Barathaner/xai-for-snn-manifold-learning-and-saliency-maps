"""
Wiederverwendbare Preprocessing-Transformationen für SHD-Daten.
Entspricht dem Preprocessing aus dem Notebook für Manifold-Analysen.
"""

import numpy as np
import tonic.transforms as transforms


class Downsample1D:
    """Downsampling nur in X-Dimension (für 1D-Audio-Daten)"""
    def __init__(self, spatial_factor=0.1):
        self.spatial_factor = spatial_factor

    def __call__(self, events):
        events = events.copy()
        events['x'] = (events['x'] * self.spatial_factor).astype(events['x'].dtype)
        return events


class SqrtTransform:
    """Wurzel-Transformation auf Frames (zur Normalisierung)"""
    def __init__(self, eps=0):
        self.eps = eps  # um sqrt(0) Probleme zu vermeiden
    
    def __call__(self, frames):
        frames = frames.astype(np.float32)
        return np.sqrt(frames + self.eps)


def get_manifold_preprocessing(n_time_bins=80):
    """
    Gibt das Standard-Preprocessing für Manifold-Analysen zurück.
    
    Args:
        n_time_bins: Anzahl der Zeitbins (Standard: 80)
    
    Returns:
        tonic.transforms.Compose Objekt
    """
    return transforms.Compose([
        Downsample1D(spatial_factor=0.1),   
        transforms.ToFrame(sensor_size=(70, 1, 1), n_time_bins=n_time_bins),
        SqrtTransform()
    ])


def get_training_preprocessing(n_time_bins=250, spatial_factor=0.5):
    """
    Gibt das Standard-Preprocessing für Training zurück.
    
    Args:
        n_time_bins: Anzahl der Zeitbins (Standard: 250)
        spatial_factor: Downsampling-Faktor (Standard: 0.5)
    
    Returns:
        tonic.transforms.Compose Objekt
    """
    return transforms.Compose([
        transforms.Downsample(spatial_factor=spatial_factor),
        transforms.ToFrame(
            sensor_size=(int(700 * spatial_factor), 1, 1),
            n_time_bins=n_time_bins
        )
    ])

