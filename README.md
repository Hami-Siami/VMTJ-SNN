# Software SNN emulation for the V-MTJ/CMOS neuron study

This repository contains the software-level spiking neural network (SNN) experiments used to evaluate tonic-bursting and phasic-bursting neuron dynamics against a leaky integrate-and-fire (LIF) baseline.

The experiments use Adaptive Exponential Integrate-and-Fire (AdEx) neurons as continuous-time software proxies for the bursting behaviors and evaluate classification accuracy and hidden-layer spike activity on MNIST, Fashion-MNIST, Breast Cancer, and IRIS. Only the experiments reported in the manuscript are included; exploratory analyses are omitted.

## Repository structure

```text
.
├── run_experiment.py          # train/evaluate one configuration
├── run_all.py                 # run all dataset/model combinations
├── plot_firing_patterns.py    # tonic/phasic single-neuron responses
├── verify_statistics.py       # Welch-test check for Table 1 spike counts
├── snn_emulation/
│   ├── neurons.py             # AdEx and LIF neuron models
│   ├── model.py               # fully connected SNN classifier
│   ├── data.py                # dataset loading and preprocessing
│   ├── configs.py             # neuron and experiment parameters
│   ├── engine.py              # training and evaluation loops
│   └── utils.py               # reproducibility and I/O helpers
└── results/
    └── reported_table1.csv    # reported accuracy/spike-count values
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run an experiment

```bash
python run_experiment.py --dataset mnist --mode tonic_burst
python run_experiment.py --dataset mnist --mode phasic_burst
python run_experiment.py --dataset mnist --mode lif
```

Supported datasets are `mnist`, `fashion_mnist`, `breast_cancer`, and `iris`. Supported modes are `tonic_burst`, `phasic_burst`, and `lif`.

Each run saves the trained checkpoint, test metrics, and per-sample spike counts under `outputs/`. If the original trained checkpoints are available, they can be placed in `checkpoints/` and evaluated directly to reproduce the archived model results without retraining.

To evaluate an existing checkpoint without retraining:

```bash
python run_experiment.py --dataset mnist --mode tonic_burst \
  --eval-only --checkpoint path/to/checkpoint.pt
```

## Reproduce the software firing patterns

```bash
python plot_firing_patterns.py
```

The default input current is `1.0` and the simulation length is 70 time steps.

## Run the complete benchmark set

```bash
python run_all.py
```

MNIST and Fashion-MNIST use a 3,000-neuron hidden layer. The tabular Breast Cancer and IRIS experiments use a 256-neuron hidden layer, following the configurations used to generate the reported results. All experiments use 40 discrete time steps and the fast-sigmoid surrogate gradient with slope 40.

The reference values reported in the manuscript are provided in `results/reported_table1.csv`. Training is seeded for reproducibility; small numerical differences can still occur across PyTorch/CUDA versions and hardware.
