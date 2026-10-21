"""Tests for the Gaussian sparse regression data-generating mechanism."""

import numpy as np
import pytest

from src.dgps import generate_data


def test_dgp_dimensions_and_sparsity():
    """Generated data must have the correct dimensions and sparsity."""
    rng = np.random.default_rng(42)

    data = generate_data(
        n=91,
        p=128,
        k=52,
        sigma=0.5,
        beta_min=0.5,
        rng=rng,
    )

    assert data["X"].shape == (91, 128)
    assert data["y"].shape == (91,)
    assert data["beta_true"].shape == (128,)

    assert np.count_nonzero(data["beta_true"]) == 52

    nonzero_beta = data["beta_true"][data["beta_true"] != 0]
    assert np.all(np.isin(nonzero_beta, [-0.5, 0.5]))


def test_dgp_reproducibility():
    """The same seed must generate identical datasets."""
    kwargs = {
        "n": 91,
        "p": 128,
        "k": 52,
        "sigma": 0.5,
        "beta_min": 0.5,
    }

    first = generate_data(
        **kwargs,
        rng=np.random.default_rng(6072026),
    )

    second = generate_data(
        **kwargs,
        rng=np.random.default_rng(6072026),
    )

    for key in ("X", "y", "beta_true"):
        np.testing.assert_array_equal(first[key], second[key])


def test_dgp_rejects_invalid_sparsity():
    """A sparsity level greater than p must be rejected."""
    with pytest.raises(ValueError):
        generate_data(
            n=91,
            p=128,
            k=129,
            sigma=0.5,
            beta_min=0.5,
            rng=np.random.default_rng(42),
        )
