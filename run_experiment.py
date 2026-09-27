"""Train and evaluate one SNN experiment."""

import argparse
from pathlib import Path

import numpy as np
import torch

from snn_emulation.configs import get_experiment_config
from snn_emulation.data import get_data_loaders
from snn_emulation.engine import (
    evaluate_model,
    load_checkpoint,
    save_checkpoint,
    train_model,
)
from snn_emulation.model import SNNClassifier
from snn_emulation.utils import choose_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["mnist", "fashion_mnist", "breast_cancer", "iris"],
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["tonic_burst", "phasic_burst", "lif"],
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate a supplied checkpoint without retraining.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_experiment_config(args.dataset, args.mode)
    set_seed(config["seed"])
    device = choose_device(args.device)
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_data_loaders(
        args.dataset, config["batch_size"], data_dir=args.data_dir
    )

    model = SNNClassifier(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_classes=config["num_classes"],
        neuron_type=config["neuron_type"],
        neuron_params=config["neuron_params"],
    ).to(device)

    output_dir = Path(args.output_dir)
    experiment_name = f"{args.dataset}_{args.mode}"
    checkpoint_path = output_dir / "checkpoints" / f"{experiment_name}.pt"

    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError("--eval-only requires --checkpoint")
        load_checkpoint(model, args.checkpoint, device)
        history = None
    else:
        history = train_model(model, train_loader, val_loader, config, device)
        save_checkpoint(model, config, checkpoint_path)

    metrics = evaluate_model(model, test_loader, config["num_steps"], device)
    spikes = metrics.pop("spikes_per_sample")

    result = {
        "dataset": args.dataset,
        "mode": args.mode,
        "config": config,
        "test": metrics,
    }
    if history is not None:
        result["history"] = history

    result_path = output_dir / "metrics" / f"{experiment_name}.json"
    spike_path = output_dir / "spikes" / f"{experiment_name}.npy"
    save_json(result, result_path)
    spike_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(spike_path, spikes)

    print("-" * 56)
    print(f"Test accuracy: {metrics['accuracy']:.2f}%")
    print(
        f"Spikes per sample: {metrics['mean_spikes']:.0f} "
        f"± {metrics['std_spikes']:.0f}"
    )
    print(f"Metrics: {result_path}")
    print(f"Per-sample spike counts: {spike_path}")
    if not args.eval_only:
        print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
