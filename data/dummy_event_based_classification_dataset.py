import numpy as np
from torch.utils.data import Dataset


class DummySpikeDataset(Dataset):
    """Dummy-Spike-Datensatz mit Hintergrundrauschen und Clustern je Klasse."""

    sensor_size = (700, 1, 1)  # (x, y, p)

    def __init__(self, num_samples=100, num_events=300, seed=42):
        np.random.seed(seed)
        self.num_samples = num_samples
        self.num_events = num_events
        self.samples = []

        for _ in range(num_samples):
            label = np.random.choice([0, 1])  # 1 = früh, 2 = spät

            n_cluster_events = int(num_events * 5 / 6)  # 5 Teile Cluster
            n_noise_events = num_events - n_cluster_events  # 1 Teil Rauschen

            # Hintergrund-Rauschen gleichmäßig über gesamte Zeit
            noise_times = np.random.randint(0, 10000, size=n_noise_events)

            # Cluster in frühem oder spätem Zeitbereich
            if label == 0:
                cluster_center = 2000
            else:
                cluster_center = 8000

            cluster_times = np.random.normal(loc=cluster_center, scale=300, size=n_cluster_events)
            cluster_times = np.clip(cluster_times, 0, 9999).astype(int)

            # Zeiten kombinieren
            times = np.concatenate([noise_times, cluster_times])

            xs = np.random.randint(0, self.sensor_size[0], size=times.shape[0])
            events = np.zeros(len(times), dtype=[("t", "i8"), ("x", "i8"), ("p", "i8")])
            events["t"] = times
            events["x"] = xs
            events["p"] = np.ones(len(times))  # nur Polarität 0

            # WICHTIG: Nach Zeit sortieren!
            events = np.sort(events, order="t")

            self.samples.append((events, label))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx]
