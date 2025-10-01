import tonic
from tonic.transforms import ToFrame

import data.dataloader as dataloader
# load dataset and filter only zero to nine number labels from SHD and then transform 
# binning of total spike counts ( in paper: total spike counts in 1-s bins)
transform = tonic.transforms.Compose([
    tonic.transforms.Downsample(sensor_size=tonic.datasets.SHD.sensor_size, target_size=(70,)),
    tonic.transforms.ToFrame(sensor_size=(70,), n_time_bins=100)
])
dataset = dataloader.load_filtered_shd_dataloader(
    label_range=range(0, 10),
    transform=transform,
    train=True,
    batch_size=1,
    shuffle=False)
print(dataset.dataset[1000])