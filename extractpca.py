import tonic
import tonic.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import tonic
from tonic.transforms import ToFrame
from tonic import DiskCachedDataset
from tonic.collation import PadTensors
from torch.utils.data import Subset, Dataset, DataLoader


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        events, label = self.dataset[idx]
        return self.transform(events), label
import numpy as np

class Downsample1D:
    def __init__(self, spatial_factor=0.1):
        self.spatial_factor = spatial_factor

    def __call__(self, events):
        # events ist ein structured array -> unbedingt copy() behalten
        events = events.copy()
        # nur das Feld 'x' skalieren
        events['x'] = (events['x'] * self.spatial_factor).astype(events['x'].dtype)
        return events  # GANZES structured array zurückgeben!


class SqrtTransform:
    def __init__(self, eps=0):
        self.eps = eps  # um sqrt(0) Probleme zu vermeiden
    def __call__(self, frames):
        frames = frames.astype(np.float32)
        return np.sqrt(frames + self.eps)
    
label_range = set(range(0,10))



trainset = tonic.datasets.SHD(save_to="./data", train=True)
print("Original trainset size:", len(trainset))

filtered_indices = [
    i for i in range(len(trainset)) if trainset[i][1] in label_range
]
trainset = Subset(trainset, filtered_indices)
trans = tonic.transforms.Compose([
    Downsample1D(spatial_factor=0.1),   
    tonic.transforms.ToFrame(sensor_size=(70,1,1), n_time_bins=8),
    SqrtTransform()
    
])
events,label = trainset[0]
frames = trans(events)
vec = frames[:, 0, :]

# vec: shape (n_time_bins, n_neurons)
n_time_bins, n_neurons = vec.shape

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

_x = np.arange(n_neurons)
_y = np.arange(n_time_bins)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.flatten(), _yy.flatten()
z = np.zeros_like(x)
dz = vec.flatten()
dx = dy = 0.8

# Farben für die Zeit-Bins
colors = plt.cm.viridis(np.linspace(0, 1, n_time_bins))
bar_colors = np.repeat(colors, n_neurons, axis=0)

ax.bar3d(x, y, z, dx, dy, dz, color=bar_colors, shade=True)
ax.set_xlabel('Neuron-Index')
ax.set_ylabel('Zeit-Bin')
ax.set_zlabel('Wert')
ax.set_title('3D-Balkendiagramm: Neuronen vs. Zeit vs. Wert (farbcodiert)')
plt.tight_layout()
ax.view_init(elev=40, azim=210)  # elev=Höhe, azim=Azimutwinkel (Drehung um die y-Achse)
plt.show()
plt.savefig("3d_balkendiagramm222.png")