import sys
import os
sys.path.append("/home/karl-/liquidstatemachines")
import tonic.transforms as transforms
from data.dataloader import *
from models.sffnn_batched import *
from training.trainer import *
from utils.metrics import *
#seed for reproduzierbarketi
import random
import numpy as np
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
# setup for cuda
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


# loading data and preprocessing
transform = transforms.Compose([
    transforms.Downsample(spatial_factor=0.5),
    transforms.ToFrame(
    sensor_size=(350,1,1),  # = (700,),
    n_time_bins=250)
    ])
train_dataloader=load_filtered_shd_dataloader(label_range=range(0, 10), transform=transform, train=True,batch_size=64)

test_dataloader=load_filtered_shd_dataloader(label_range=range(0, 10), transform=transform, train=False,batch_size=64)


#loading model
net = Net(num_inputs=350, num_hidden=1000, num_outputs=10, num_steps=250, beta=0.9).to(device)

#training parameters
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)

if __name__ == "__main__":
    #training loop
    num_epoch=3
    for epoch in range(1, num_epoch+1):
        print(f"\nEpoch {epoch}")
        train_one_epoch_batched(net=net,
                                dataloader=train_dataloader,
                                optimizer=optimizer,
                                loss_fn=loss,
                                device=device)
    #test val
    print_full_dataloader_accuracy_batched(net, test_dataloader)
