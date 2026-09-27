"""Fully connected SNN classifier used for all benchmark experiments."""

import torch
import torch.nn as nn

from .neurons import AdEx, LIF


class SNNClassifier(nn.Module):
    """One-hidden-layer SNN with temporally accumulated output logits."""

    def __init__(self, input_size, hidden_size, num_classes, neuron_type, neuron_params):
        super().__init__()
        self.neuron_type = neuron_type
        self.fc1 = nn.Linear(input_size, hidden_size)

        if neuron_type == "adex":
            self.neuron = AdEx(**neuron_params)
        elif neuron_type == "lif":
            self.neuron = LIF(**neuron_params)
        else:
            raise ValueError("neuron_type must be 'adex' or 'lif'")

        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, num_steps):
        x = x.view(x.size(0), -1)
        batch_size = x.size(0)

        V = torch.full(
            (batch_size, self.fc1.out_features),
            self.neuron.EL,
            device=x.device,
            dtype=torch.float32,
        )
        if self.neuron_type == "adex":
            w = torch.zeros(
                (batch_size, self.fc1.out_features),
                device=x.device,
                dtype=torch.float32,
            )

        out_logits = torch.zeros(
            (batch_size, self.fc2.out_features),
            device=x.device,
            dtype=torch.float32,
        )
        total_spikes_per_sample = torch.zeros(
            batch_size, device=x.device, dtype=torch.float32
        )

        I_in = self.fc1(x)

        for _ in range(num_steps):
            if self.neuron_type == "adex":
                spike, V, w = self.neuron(I_in, V, w)
            else:
                spike, V = self.neuron(I_in, V)

            out_logits += self.fc2(spike)
            total_spikes_per_sample += spike.sum(dim=1)

        return out_logits, total_spikes_per_sample
