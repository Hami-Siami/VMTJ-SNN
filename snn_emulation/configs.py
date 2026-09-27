"""Experiment settings matching the software experiments used for the reported results."""

from copy import deepcopy


SURROGATE = {"surrogate_type": "fast_sigmoid", "surrogate_param": 40.0}

TONIC_BURST_PARAMS = {
    "C": 0.6,
    "gL": 0.1,
    "EL": 0.0,
    "VT": 0.3,
    "DT": 0.5,
    "a": 0.25,
    "b": 0.25,
    "tau_w": 30.0,
    "v_reset": 1.7,
    "v_thresh": 2.0,
    "dt": 1.0,
    **SURROGATE,
}

PHASIC_BURST_PARAMS = {
    "C": 0.75,
    "gL": 0.1,
    "EL": -0.10,
    "VT": 0.2,
    "DT": 0.5,
    "a": 0.05,
    "b": 0.5,
    "tau_w": 90.0,
    "v_reset": 1.8,
    "v_thresh": 2.0,
    "dt": 1.0,
    **SURROGATE,
}

LIF_PARAMS = {
    "C": 0.25,
    "gL": 0.1,
    "EL": 0.0,
    "v_reset": 0.3,
    "v_thresh": 2.0,
    "dt": 1.0,
    **SURROGATE,
}


# Dataset-level settings used in the experiment notebook.
DATASET_CONFIGS = {
    "mnist": {
        "input_size": 784,
        "num_classes": 10,
        "hidden_size": 3000,
        "batch_size": 128,
        "num_steps": 40,
        "epochs": {"tonic_burst": 50, "phasic_burst": 50, "lif": 50},
        "learning_rate": {"tonic_burst": 5e-5, "phasic_burst": 5e-5, "lif": 5e-5},
    },
    "fashion_mnist": {
        "input_size": 784,
        "num_classes": 10,
        "hidden_size": 3000,
        "batch_size": 128,
        "num_steps": 40,
        "epochs": {"tonic_burst": 50, "phasic_burst": 50, "lif": 50},
        "learning_rate": {"tonic_burst": 5e-5, "phasic_burst": 5e-5, "lif": 5e-5},
    },
    "breast_cancer": {
        "input_size": 30,
        "num_classes": 2,
        "hidden_size": 256,
        "batch_size": 16,
        "num_steps": 40,
        "epochs": {"tonic_burst": 20, "phasic_burst": 20, "lif": 20},
        "learning_rate": {"tonic_burst": 1e-3, "phasic_burst": 1e-3, "lif": 1e-3},
    },
    "iris": {
        "input_size": 4,
        "num_classes": 3,
        "hidden_size": 256,
        "batch_size": 16,
        "num_steps": 40,
        "epochs": {"tonic_burst": 20, "phasic_burst": 15, "lif": 20},
        "learning_rate": {"tonic_burst": 1e-3, "phasic_burst": 1e-3, "lif": 5e-4},
    },
}


def get_experiment_config(dataset, mode):
    """Return a complete configuration dictionary for one experiment."""
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if mode not in {"tonic_burst", "phasic_burst", "lif"}:
        raise ValueError(f"Unknown mode: {mode}")

    base = deepcopy(DATASET_CONFIGS[dataset])
    if mode == "tonic_burst":
        neuron_type = "adex"
        neuron_params = deepcopy(TONIC_BURST_PARAMS)
    elif mode == "phasic_burst":
        neuron_type = "adex"
        neuron_params = deepcopy(PHASIC_BURST_PARAMS)
    else:
        neuron_type = "lif"
        neuron_params = deepcopy(LIF_PARAMS)

    return {
        "dataset": dataset,
        "mode": mode,
        "neuron_type": neuron_type,
        "neuron_params": neuron_params,
        "input_size": base["input_size"],
        "num_classes": base["num_classes"],
        "hidden_size": base["hidden_size"],
        "batch_size": base["batch_size"],
        "num_steps": base["num_steps"],
        "num_epochs": base["epochs"][mode],
        "learning_rate": base["learning_rate"][mode],
        "betas": (0.9, 0.999),
        "seed": 42,
    }
