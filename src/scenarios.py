
"""Define simulation scenarios for reproducing Wainwright (2009), Figure 2."""

import math
from itertools import product

import pandas as pd


DIMENSIONS = (128, 256, 512)
SPARSITY_REGIMES = ("linear", "sublinear", "fractional")
THETA_GRID = (
    0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1,
    1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4,
)

SPARSITY_ALPHA = 0.40
SPARSITY_DELTA = 0.75
NOISE_SIGMA = 0.50
BETA_MIN = 0.50
N_REPLICATIONS = 200


def calculate_sparsity(p, regime):
    """Calculate the integer sparsity level for a specified regime."""
    if p <= 1:
        raise ValueError("p must be greater than 1.")

    if regime == "linear":
        k = math.ceil(SPARSITY_ALPHA * p)
    elif regime == "sublinear":
        k = math.ceil(
            SPARSITY_ALPHA * p / math.log(SPARSITY_ALPHA * p)
        )
    elif regime == "fractional":
        k = math.ceil(SPARSITY_ALPHA * p**SPARSITY_DELTA)
    else:
        raise ValueError(f"Unknown sparsity regime: {regime}")

    if not 1 < k < p:
        raise ValueError(f"Invalid sparsity level k={k} for p={p}.")

    return k


def calculate_sample_size(p, k, theta):
    """Calculate sample size using the documented ceiling rule."""
    if not 1 < k < p:
        raise ValueError("Expected 1 < k < p.")
    if not 0 < theta <= 2.4:
        raise ValueError("theta must be in (0, 2.4].")

    return math.ceil(2 * theta * k * math.log(p - k))


def generate_scenarios():
    """Generate the full factorial design for Figure 2 replication."""
    scenarios = []

    for regime, p, theta in product(
        SPARSITY_REGIMES, DIMENSIONS, THETA_GRID
    ):
        k = calculate_sparsity(p, regime)
        n = calculate_sample_size(p, k, theta)

        scenarios.append(
            {
                "scenario_id": f"{regime}_p{p}_theta{theta:.1f}",
                "regime": regime,
                "p": p,
                "k": k,
                "theta": theta,
                "n": n,
                "sigma": NOISE_SIGMA,
                "beta_min": BETA_MIN,
                "n_replications": N_REPLICATIONS,
            }
        )

    return pd.DataFrame(scenarios)


if __name__ == "__main__":
    design = generate_scenarios()
    print(design.head())
    print(f"\nTotal scenarios: {len(design)}")

    output_path = "results/scenario_design.csv"
    design.to_csv(output_path, index=False)
    print(f"Saved design matrix to {output_path}")
