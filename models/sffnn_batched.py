import snntorch as snn
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, num_steps, beta):
        super().__init__()

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)
        self.num_steps = num_steps
        # Container für Aufzeichnungen über Zeit (wird im forward gefüllt)
        self.recordings = {}

    def forward(self, x):
        # x shape: [B, T, num_inputs]
        B, T, _ = x.shape
        assert T == self.num_steps, f"Eingabezeitdimension ({T}) stimmt nicht mit num_steps ({self.num_steps}) überein."

        # initialisiere hidden states anhand Eingabeform
        # Initialize membrane potentials as zeros with the correct shape
        mem1 = torch.zeros(B, self.fc1.out_features, device=x.device)
        mem2 = torch.zeros(B, self.fc2.out_features, device=x.device)


        spk1_rec = []
        mem1_rec = []
        spk2_rec = []
        mem2_rec = []

        for step in range(T):
            x_t = x[:, step, :]              # Shape: [B, num_inputs]
            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk1_rec.append(spk1)
            mem1_rec.append(mem1)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        # [T, B, features] → [B, T, features]
        spk1_bt = torch.stack(spk1_rec, dim=0).permute(1, 0, 2)
        mem1_bt = torch.stack(mem1_rec, dim=0).permute(1, 0, 2)
        spk2_bt = torch.stack(spk2_rec, dim=0).permute(1, 0, 2)
        mem2_bt = torch.stack(mem2_rec, dim=0).permute(1, 0, 2)

        # Aufzeichnungen für spätere Auswertung (auf CPU, ohne Gradienten)
        self.recordings = {
            "spk1": spk1_bt.detach().cpu(),
            "mem1": mem1_bt.detach().cpu(),
            "spk2": spk2_bt.detach().cpu(),
            "mem2": mem2_bt.detach().cpu(),
        }

        return spk2_bt, mem2_bt
