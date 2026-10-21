"""Tests for exact signed-support recovery metrics."""

import numpy as np
import pytest

from src.metrics import exact_recovery, signed_support


def test_exact_recovery_success():
    """Correct signed support must be marked successful."""
    beta_true = np.array([0.0, 0.5, -0.5, 0.0])
    beta_hat = np.array([1e-10, 0.4, -0.3, 0.0])

    assert exact_recovery(beta_hat, beta_true) is True


def test_exact_recovery_wrong_sign():
    """An incorrect coefficient sign must cause failure."""
    beta_true = np.array([0.0, 0.5, -0.5, 0.0])
    beta_hat = np.array([0.0, 0.4, 0.3, 0.0])

    assert exact_recovery(beta_hat, beta_true) is False


def test_exact_recovery_false_positive():
    """Selecting an inactive variable must cause failure."""
    beta_true = np.array([0.0, 0.5, -0.5, 0.0])
    beta_hat = np.array([0.1, 0.4, -0.3, 0.0])

    assert exact_recovery(beta_hat, beta_true) is False


def test_signed_support_rejects_negative_tolerance():
    """A negative zero tolerance is invalid."""
    with pytest.raises(ValueError):
        signed_support(np.array([0.0, 1.0]), zero_tol=-1e-8)
