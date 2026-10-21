"""Performance metrics for signed-support recovery."""

import numpy as np


def signed_support(beta, zero_tol=1e-8):
    """Return the signed support, treating tiny coefficients as zero."""
    beta = np.asarray(beta, dtype=float)

    if beta.ndim != 1:
        raise ValueError("beta must be a one-dimensional vector.")

    if zero_tol < 0:
        raise ValueError("zero_tol must be nonnegative.")

    signs = np.zeros(beta.shape, dtype=int)
    signs[beta > zero_tol] = 1
    signs[beta < -zero_tol] = -1

    return signs


def exact_recovery(beta_hat, beta_true, zero_tol=1e-8):
    """Check whether the estimated signed support is exactly correct."""
    beta_hat = np.asarray(beta_hat, dtype=float)
    beta_true = np.asarray(beta_true, dtype=float)

    if beta_hat.shape != beta_true.shape:
        raise ValueError("Estimated and true coefficients must match in shape.")

    estimated_signs = signed_support(beta_hat, zero_tol)
    true_signs = signed_support(beta_true, zero_tol=0.0)

    return bool(np.array_equal(estimated_signs, true_signs))
