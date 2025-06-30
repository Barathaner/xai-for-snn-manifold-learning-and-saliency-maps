# data/dataloader.py

import numpy as np
from torch.utils.data import Dataset
import tonic


class DummySpikeDataset(Dataset):
    """Künstlicher Spike-Datensatz mit 2 Klassen basierend auf Eventzeit."""

    sensor_size = (700, 1, 1)  # (x, y, polarity), wie SHD

    def __init__(self, num_samples=100, num_events=300, seed=42):
        np.random.seed(seed)
        self.num_samples = num_samples
        self.num_events = num_events
        self.samples = []

        for i in range(num_samples):
            label = np.random.choice([0, 1])  # 0 = früh, 1 = spät
            if label == 0:
                times = np.random.randint(0, 5000, size=num_events)
            else:
                times = np.random.randint(5000, 10000, size=num_events)

            xs = np.random.randint(0, self.sensor_size[0], size=num_events)
            ps = np.random.randint(0, self.sensor_size[2], size=num_events)
            events = np.zeros(num_events, dtype=[("x", "i4"), ("y", "i4"), ("t", "i4"), ("p", "i4")])
            events["x"] = xs
            events["y"] = 0  # y = 0 (wie bei SHD)
            events["t"] = times
            events["p"] = 0

            self.samples.append((events, label))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx]
