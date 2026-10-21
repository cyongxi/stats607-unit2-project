"""Summarize Lasso signed-support recovery simulation results."""

from pathlib import Path

import numpy as np
import pandas as pd


RAW_PATH = Path("results/raw/simulation_results.csv")
SUMMARY_PATH = Path("results/recovery_summary.csv")


def summarize_results(results):
    """Compute recovery probabilities and Monte Carlo standard errors."""
    required = {
        "scenario_id", "regime", "p", "k", "n",
        "theta", "replication_id", "success",
    }

    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if results.empty:
        raise ValueError("Simulation results must not be empty.")

    if not results["success"].isin([0, 1]).all():
        raise ValueError("Success values must be binary.")

    group_cols = [
        "scenario_id", "regime", "p", "k", "n", "theta"
    ]

    summary = (
        results.groupby(group_cols, as_index=False)
        .agg(
            n_replications=("success", "size"),
            n_success=("success", "sum"),
            recovery_probability=("success", "mean"),
        )
    )

    b = summary["n_replications"]
    prob = summary["recovery_probability"]

    summary["mcse"] = np.where(
        b > 1,
        np.sqrt(prob * (1 - prob) / (b - 1).clip(lower=1)),
        np.nan,
    )

    return summary


def main():
    """Read raw results and save scenario-level summaries."""
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw results not found: {RAW_PATH}. Run the simulation first."
        )

    results = pd.read_csv(RAW_PATH)
    summary = summarize_results(results)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"Summarized {len(summary)} scenarios.")
    print(f"Saved results to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
