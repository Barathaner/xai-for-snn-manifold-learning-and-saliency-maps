import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
import torch
import numpy as np
def plot_raster_from_frames(frames, title="Rasterplot nach Transform"):
    """
    Visualisiert einen Rasterplot aus transformierten Spike-Frames.

    frames: numpy array mit Shape [T, 1, 700] (Zeitbins, Kanäle, Sensorgröße)
    title: Titel für die Grafik
    """
    frames = frames.reshape(frames.shape[0], -1)  # Garantiert 2D
    spike_times, neuron_indices = np.nonzero(frames)

    plt.figure(figsize=(10, 5))
    plt.scatter(spike_times, neuron_indices, s=1, marker='|', color='black')
    plt.xlabel("Zeitbins")
    plt.ylabel("Neuron Index")
    plt.title(title)
    plt.show()
    
def plot_raster_from_events(events,title="Rasterplot eines Rohdaten-Spiking Events"):
    # Dynamisch maximale Werte bestimmen
    neurons = events["x"].max() + 1
    timesteps = events["t"].max() + 1

    # Spiketrain-Tensor erzeugen
    spike_tensor = torch.zeros((timesteps, neurons), dtype=torch.float32)

    # Spikes eintragen
    for x, t in zip(events["x"], events["t"]):
        spike_tensor[t, x] = 1

    # Plotten
    fig, ax = plt.subplots(figsize=(12, 6))
    splt.raster(spike_tensor, ax, s=1.5, c="black")
    ax.set_title(title)
    ax.set_xlabel("Zeit (ms)")
    ax.set_ylabel("Neuron")
    plt.show()