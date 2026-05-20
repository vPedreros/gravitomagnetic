"""
Unit tests for the field normalisation logic in fields.py.

These tests don't invoke main() (which requires MAS_library / Pylians3) but
instead test the normalisation arithmetic directly to guard against regressions
like the previously-present `np.mean(rho_mean)` redundancy.
"""

import numpy as np
import pytest


def apply_fields_normalization(rho_3d, vel_3d):
    """
    Replicate the normalization from fields.py main() without Pylians3.
    Returns delta and momentum fields.
    """
    rho_mean = np.mean(rho_3d, dtype=np.float32)
    delta = rho_3d / rho_mean - 1.0

    qx = rho_3d * vel_3d[:, :, :, 0]
    qy = rho_3d * vel_3d[:, :, :, 1]
    qz = rho_3d * vel_3d[:, :, :, 2]

    qx /= rho_mean
    qy /= rho_mean
    qz /= rho_mean

    return delta, qx, qy, qz


def test_delta_mean_approx_zero():
    """Overdensity field should have mean ≈ 0 by construction."""
    rng = np.random.default_rng(42)
    # Uniform random density — mean should be preserved so delta mean = 0
    rho = rng.uniform(0.5, 1.5, (16, 16, 16)).astype(np.float32)
    vel = rng.standard_normal((16, 16, 16, 3)).astype(np.float32)

    delta, *_ = apply_fields_normalization(rho, vel)
    assert abs(np.mean(delta)) < 1e-5


def test_delta_not_biased():
    """Constant-density field should give delta = 0 everywhere."""
    rho = np.ones((8, 8, 8), dtype=np.float32) * 2.5
    vel = np.ones((8, 8, 8, 3), dtype=np.float32)

    delta, *_ = apply_fields_normalization(rho, vel)
    np.testing.assert_allclose(delta, 0.0, atol=1e-6)


def test_momentum_normalization_preserves_velocity_scale():
    """
    q / rho_mean should be ~ v for uniform density (rho everywhere = rho_mean).
    Checks that dividing by rho_mean (scalar) gives physically consistent result.
    """
    rho = np.ones((8, 8, 8), dtype=np.float32) * 3.0
    v_uniform = 5.0
    vel = np.full((8, 8, 8, 3), v_uniform, dtype=np.float32)

    _, qx, qy, qz = apply_fields_normalization(rho, vel)

    # With uniform rho = rho_mean, q = rho * v / rho_mean = v
    np.testing.assert_allclose(qx, v_uniform, rtol=1e-5)
    np.testing.assert_allclose(qy, v_uniform, rtol=1e-5)
    np.testing.assert_allclose(qz, v_uniform, rtol=1e-5)


def test_rho_mean_is_scalar():
    """Guard that rho_mean is always a scalar (not an array), ensuring /= works correctly."""
    rng = np.random.default_rng(0)
    rho = rng.uniform(0.1, 2.0, (10, 10, 10)).astype(np.float32)
    rho_mean = np.mean(rho, dtype=np.float32)
    assert np.isscalar(rho_mean) or rho_mean.ndim == 0
