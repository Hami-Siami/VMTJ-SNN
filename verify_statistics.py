"""Verify the reported Welch-test significance from Table 1 summary statistics."""

import csv
from pathlib import Path

from scipy.stats import ttest_ind_from_stats


TEST_SET_SIZES = {
    "MNIST": 10000,
    "Fashion-MNIST": 10000,
    "Breast Cancer": 115,
    "IRIS": 31,
}


def main():
    table_path = Path("results/reported_table1.csv")
    rows = list(csv.DictReader(table_path.open(encoding="utf-8")))

    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], {})[row["mode"]] = row

    for dataset, modes in by_dataset.items():
        lif = modes["lif"]
        n = TEST_SET_SIZES[dataset]
        for mode in ("tonic_burst", "phasic_burst"):
            burst = modes[mode]
            result = ttest_ind_from_stats(
                mean1=float(lif["mean_spikes"]),
                std1=float(lif["std_spikes"]),
                nobs1=n,
                mean2=float(burst["mean_spikes"]),
                std2=float(burst["std_spikes"]),
                nobs2=n,
                equal_var=False,
            )
            print(f"{dataset:16s} {mode:13s} p = {result.pvalue:.3e}")


if __name__ == "__main__":
    main()
