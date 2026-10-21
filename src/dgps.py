"""Data-generating mechanism for Wainwright (2009), Figure 2."""

import numpy as np


def generate_data(n, p, k, sigma, beta_min, rng):
    """Generate one sparse Gaussian linear regression dataset.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of predictors.
    k : int
        Number of nonzero coefficients.
    sigma : float
        Standard deviation of Gaussian noise.
    beta_min : float
        Absolute value of each nonzero coefficient.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    dict
        Generated X, y, and beta_true.
    """
    if not (isinstance(n, int) and n > 0):
        raise ValueError("n must be a positive integer.")

    if not (isinstance(p, int) and p > 1):
        raise ValueError("p must be an integer greater than 1.")

    if not (isinstance(k, int) and 1 < k < p):
        raise ValueError("k must satisfy 1 < k < p.")

    if sigma < 0 or beta_min <= 0:
        raise ValueError("Invalid noise or coefficient magnitude.")

    # 1. Generate the Gaussian design matrix.
    X = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    # 2. Randomly select the true support.
    support = rng.choice(p, size=k, replace=False)

    # 3. Assign random signs to nonzero coefficients.
    beta_true = np.zeros(p)
    signs = rng.choice([-1.0, 1.0], size=k)
    beta_true[support] = beta_min * signs

    # 4. Generate Gaussian observation noise.
    noise = rng.normal(loc=0.0, scale=sigma, size=n)

    # 5. Generate the response vector.
    y = X @ beta_true + noise

    return {
        "X": X,
        "y": y,
        "beta_true": beta_true,
    }
