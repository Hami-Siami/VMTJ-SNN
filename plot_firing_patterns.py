"""Reproduce the software-level tonic- and phasic-bursting responses."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from snn_emulation.configs import PHASIC_BURST_PARAMS, TONIC_BURST_PARAMS
from snn_emulation.neurons import AdEx


def simulate(params, input_current=1.0, num_steps=70):
    neuron = AdEx(**params)
    V = torch.tensor([[neuron.EL]], dtype=torch.float32)
    w = torch.zeros_like(V)
    I_in = torch.full_like(V, input_current)

    spikes = []
    with torch.no_grad():
        for _ in range(num_steps):
            spike, V, w = neuron(I_in, V, w)
            spikes.append(float(spike.item()))
    return spikes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-current", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=70)
    parser.add_argument("--output", default="results/figure5_software_emulation.png")
    return parser.parse_args()


def main():
    args = parse_args()
    tonic = simulate(TONIC_BURST_PARAMS, args.input_current, args.num_steps)
    phasic = simulate(PHASIC_BURST_PARAMS, args.input_current, args.num_steps)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    for ax, spikes, title in zip(
        axes, [tonic, phasic], ["Tonic Bursting", "Phasic Bursting"]
    ):
        times = range(args.num_steps)
        ax.vlines(times, 0, spikes, linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Time step")
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Spike")
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
