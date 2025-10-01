import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class RasterPlotCallback:
    def __init__(self, dataloader, device, out_dir="./plots", max_neurons_hidden=100):
        self.dataloader = dataloader
        self.device = device
        self.out_dir = out_dir
        self.max_neurons_hidden = max_neurons_hidden
        os.makedirs(self.out_dir, exist_ok=True)

    def _raster(self, spk_btj, batch_index=0, max_neurons=None, title="Raster", file_path=None):
        s = spk_btj[batch_index].numpy()
        t_idx, j_idx = (s > 0).nonzero()
        if max_neurons is not None:
            mask = j_idx < max_neurons
            t_idx, j_idx = t_idx[mask], j_idx[mask]
        plt.figure(figsize=(9, 4))
        plt.scatter(t_idx, j_idx, s=3)
        plt.xlabel("Zeit")
        plt.ylabel("Neuron")
        plt.title(title)
        plt.tight_layout()
        if file_path is not None:
            plt.savefig(file_path)
            plt.close()
        else:
            plt.show()

    def on_epoch_end(self, net, epoch: int):
        net.eval()
        with torch.no_grad():
            x, _ = next(iter(self.dataloader))
            if x.ndim == 4:
                x = x.squeeze(2)
            _ = net(x.to(self.device))

        spk1 = net.recordings.get("spk1")
        spk2 = net.recordings.get("spk2")
        if spk1 is None or spk2 is None:
            return

        out_hidden = os.path.join(self.out_dir, f"epoch_{epoch:03d}_hidden_raster.png")
        out_output = os.path.join(self.out_dir, f"epoch_{epoch:03d}_output_raster.png")

        self._raster(spk1, batch_index=0, max_neurons=self.max_neurons_hidden,
                     title=f"Hidden Raster – Epoche {epoch}", file_path=out_hidden)
        self._raster(spk2, batch_index=0, max_neurons=None,
                     title=f"Output Raster – Epoche {epoch}", file_path=out_output)


