"""Create diagnostic and publication-quality simulation figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_PATH = Path("results/recovery_summary.csv")
RAW_PATH = Path("results/raw/simulation_results.csv")
FIGURE_DIR = Path("results/figures")

REGIMES = ["linear", "sublinear", "fractional"]
REGIME_TITLES = {
    "linear": "Linear sparsity",
    "sublinear": "Sublinear sparsity",
    "fractional": "Fractional-power sparsity",
}


def plot_recovery_curves(summary):
    """Plot signed-support recovery probability versus theta."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, regime in zip(axes, REGIMES):
        subset = summary[summary["regime"] == regime]

        for p in sorted(subset["p"].unique()):
            curve = subset[subset["p"] == p].sort_values("theta")

            ax.errorbar(
                curve["theta"],
                curve["recovery_probability"],
                yerr=1.96 * curve["mcse"],
                marker="o",
                markersize=3,
                capsize=2,
                label=f"p = {p}",
            )

        ax.axvline(
            1.0, color="gray", linestyle="--",
            linewidth=1, alpha=0.7,
        )
        ax.set_title(REGIME_TITLES[regime])
        ax.set_xlabel("Rescaled sample size (theta)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.2)
        if not subset.empty:
            ax.legend()

    axes[0].set_ylabel("Exact signed-support recovery probability")

    fig.suptitle(
        "Lasso Signed-support Recovery: Wainwright (2009) Replication"
    )
    fig.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / "recovery_curves.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_solver_diagnostics(raw):
    """Plot distributions of Lasso solver iterations by sparsity regime."""
    fig, ax = plt.subplots(figsize=(8, 5))

    regimes = [r for r in REGIMES if r in raw["regime"].unique()]
    data = [
        raw.loc[raw["regime"] == regime, "n_iter"].to_numpy()
        for regime in regimes
    ]

    if not data:
        raise ValueError("No valid sparsity regimes found.")

    ax.boxplot(data, tick_labels=regimes)
    ax.set_xlabel("Sparsity regime")
    ax.set_ylabel("Lasso solver iterations")
    ax.set_title("Diagnostic: Lasso Solver Iterations")
    ax.grid(axis="y", alpha=0.2)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / "solver_diagnostics.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main():
    """Generate both figures from saved simulation results."""
    if not SUMMARY_PATH.exists() or not RAW_PATH.exists():
        raise FileNotFoundError(
            "Simulation results are missing. Run simulation and analysis first."
        )

    summary = pd.read_csv(SUMMARY_PATH)
    raw = pd.read_csv(RAW_PATH)

    recovery_path = plot_recovery_curves(summary)
    diagnostic_path = plot_solver_diagnostics(raw)

    print(f"Saved publication-quality plot: {recovery_path}")
    print(f"Saved diagnostic plot: {diagnostic_path}")


if __name__ == "__main__":
    main()
