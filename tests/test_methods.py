"""Tests for the Lasso estimation method."""

import math

import numpy as np
import pytest

from src.methods import calculate_lambda, fit_lasso


def test_lambda_formula():
    """Regularization parameter must match the paper's formula."""
    n, p, k, sigma = 91, 128, 52, 0.5

    expected = math.sqrt(
        2 * sigma**2 * math.log(k) * math.log(p - k) / n
    )

    actual = calculate_lambda(n, p, k, sigma)

    assert actual == pytest.approx(expected)


def test_lasso_output():
    """Lasso must return valid coefficient estimates."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 20))
    beta = np.zeros(20)
    beta[:3] = [1.0, -1.0, 0.8]
    y = X @ beta

    result = fit_lasso(X, y, k=3, sigma=0.1)

    assert result["beta_hat"].shape == (20,)
    assert np.all(np.isfinite(result["beta_hat"]))
    assert result["lambda_n"] > 0
    assert result["n_iter"] <= 20000


def test_lambda_rejects_invalid_input():
    """Invalid sparsity must raise ValueError."""
    with pytest.raises(ValueError):
        calculate_lambda(n=91, p=128, k=129, sigma=0.5)
