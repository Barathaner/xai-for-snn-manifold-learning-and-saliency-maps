from tonic.transforms import ToFrame
from tonic import MemoryCachedDataset
from tonic.collation import PadTensors
from torch.utils.data import DataLoader
from data.dummy_event_based_classification_dataset import DummySpikeDataset

# Dummy-Datensatz erzeugen
dataset = DummySpikeDataset(num_samples=200, num_events=300)

# Transform
transform = ToFrame(sensor_size=DummySpikeDataset.sensor_size,
                    time_window=1_000,
                    n_time_bins=1000)

# In Memory laden + Dataloader
transformed_dataset = [(transform(e), l) for e, l in dataset]
cached_dataset = MemoryCachedDataset(transformed_dataset)

loader = DataLoader(cached_dataset, batch_size=32, collate_fn=PadTensors())


if __name__ == "__main__":
    # Test
    x, y = next(iter(loader))
    print(x.shape)  # z. B. [32, 1000, 1, 700]
    print(y)        # Labels: 0 (früh), 1 (spät)
