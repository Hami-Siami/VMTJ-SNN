"""Training and evaluation loops for the SNN experiments."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def train_model(model, train_loader, val_loader, config, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        betas=config["betas"],
    )

    history = []
    for epoch in range(config["num_epochs"]):
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            config["num_steps"],
            optimizer=optimizer,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            config["num_steps"],
            optimizer=None,
        )

        record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_mean_spikes": train_metrics["mean_spikes"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_mean_spikes": val_metrics["mean_spikes"],
        }
        history.append(record)
        print(
            f"Epoch [{epoch + 1}/{config['num_epochs']}] "
            f"Train Acc: {record['train_accuracy']:.2f}% "
            f"(Spikes/Sample: {record['train_mean_spikes']:.0f}) | "
            f"Val Acc: {record['val_accuracy']:.2f}% "
            f"(Spikes/Sample: {record['val_mean_spikes']:.0f})"
        )

    return history


def evaluate_model(model, data_loader, num_steps, device):
    criterion = nn.CrossEntropyLoss()
    metrics = _run_epoch(
        model, data_loader, criterion, device, num_steps, optimizer=None, keep_spikes=True
    )
    return metrics


def save_checkpoint(model, config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return checkpoint


def _run_epoch(model, data_loader, criterion, device, num_steps, optimizer=None, keep_spikes=False):
    training = optimizer is not None
    model.train(training)

    running_loss = 0.0
    correct = 0
    total = 0
    spike_batches = []

    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if training:
                optimizer.zero_grad()

            outputs, spikes_per_sample = model(inputs, num_steps=num_steps)
            loss = criterion(outputs, targets)

            if training:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predicted = outputs.argmax(dim=1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            spike_batches.append(spikes_per_sample.detach().cpu())

    all_spikes = torch.cat(spike_batches) if spike_batches else torch.empty(0)
    result = {
        "loss": running_loss / total if total else 0.0,
        "accuracy": 100.0 * correct / total if total else 0.0,
        "mean_spikes": all_spikes.mean().item() if len(all_spikes) else 0.0,
        "std_spikes": all_spikes.std().item() if len(all_spikes) > 1 else 0.0,
        "n_samples": int(total),
    }
    if keep_spikes:
        result["spikes_per_sample"] = all_spikes.numpy().astype(np.float32)
    return result
