"""Run all dataset/neuron configurations used for the reported comparison."""

import subprocess
import sys


DATASETS = ["mnist", "fashion_mnist", "breast_cancer", "iris"]
MODES = ["lif", "tonic_burst", "phasic_burst"]


def main():
    for dataset in DATASETS:
        for mode in MODES:
            command = [
                sys.executable,
                "run_experiment.py",
                "--dataset",
                dataset,
                "--mode",
                mode,
            ]
            print("\n$ " + " ".join(command))
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
