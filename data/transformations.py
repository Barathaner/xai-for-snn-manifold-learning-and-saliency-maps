import numpy as np
from sklearn.cluster import DBSCAN

def denoise_with_dbscan_1d(events, eps=100, min_samples=20):
    """
    Entfernt Spikes, die nicht zu dichten zeitlichen Clustern gehören (1D-DBSCAN).
    
    Args:
        events: structured array mit 't' (Zeit) und 'x' (Sensorposition)
        eps: maximaler Zeitabstand innerhalb eines Clusters (in ms oder Zeiteinheit)
        min_samples: Mindestanzahl an Events innerhalb eines Zeitclusters
    
    Returns:
        Gefiltertes structured array mit Events innerhalb dichter Cluster
    """
    if len(events) == 0:
        return events  # leeres Array bleibt leer

    # Zeitpunkte in DBSCAN-Format bringen: (n_samples, 1)
    t = events["t"].reshape(-1, 1)

    # DBSCAN auf Zeitachse anwenden
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(t)
    labels = clustering.labels_

    # -1 steht für Rauschen → wir behalten nur Clusterpunkte
    keep_mask = labels != -1
    return events[keep_mask]


def denoise_knn_1d(events, window=100, min_neighbors=5):
    """
    Behält nur Events, die mindestens `min_neighbors` andere Events
    innerhalb eines Zeitfensters ±`window` haben.

    Args:
        events: structured array mit 't' (int) und 'x'
        window: Zeitfenstergröße (nach links und rechts)
        min_neighbors: Mindestanzahl an Nachbarevents, um Event zu behalten

    Returns:
        Gefiltertes Event-Array
    """
    times = events["t"]
    sorted_indices = np.argsort(times)
    sorted_times = times[sorted_indices]

    # Initialisiere Maske, die angibt, ob Event behalten wird
    keep_mask = np.zeros(len(events), dtype=bool)

    for i in range(len(sorted_times)):
        t = sorted_times[i]

        # Finde linke und rechte Grenzen im Zeitfenster
        lower = t - window
        upper = t + window

        # Anzahl Nachbarn im Fenster zählen
        left = np.searchsorted(sorted_times, lower, side='left')
        right = np.searchsorted(sorted_times, upper, side='right')
        neighbors = right - left - 1  # -1: sich selbst nicht mitzählen

        if neighbors >= min_neighbors:
            keep_mask[sorted_indices[i]] = True

    return events[keep_mask]
