"""Initial displacement fields for membrane simulations."""

from numbers import Real

import numpy as np


def create_initial_displacement(
    X: np.ndarray,
    Y: np.ndarray,
    amplitude: float,
    strike_x: float,
    strike_y: float,
    sigma: float,
    interior_mask: np.ndarray,
) -> np.ndarray:
    """Create a masked Gaussian displacement with zero initial velocity."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    interior_mask = np.asarray(interior_mask)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Grid coordinates must be two-dimensional arrays.")
    if X.shape != Y.shape or X.shape != interior_mask.shape:
        raise ValueError("Coordinates and interior mask must have identical shapes.")
    if interior_mask.dtype != np.bool_:
        raise TypeError("Interior mask must be boolean.")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("Grid coordinates must contain finite values.")
    if not np.any(interior_mask):
        raise ValueError("Interior mask contains no active points.")

    parameters = (amplitude, strike_x, strike_y, sigma)
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in parameters):
        raise TypeError("Initial condition parameters must be real numbers.")
    amplitude, strike_x, strike_y, sigma = map(float, parameters)
    if not np.isfinite(parameters).all():
        raise ValueError("Initial condition parameters must be finite.")
    if amplitude < 0:
        raise ValueError("Amplitude cannot be negative.")
    if sigma <= 0:
        raise ValueError("Gaussian spread sigma must be positive.")
    if not (
        float(np.min(X)) <= strike_x <= float(np.max(X))
        and float(np.min(Y)) <= strike_y <= float(np.max(Y))
    ):
        raise ValueError("Strike position must lie within the grid extent.")

    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        distance_squared = (X - strike_x) ** 2 + (Y - strike_y) ** 2
        displacement = amplitude * np.exp(-distance_squared / (2 * sigma**2))
    displacement[~interior_mask] = 0.0
    if not np.isfinite(displacement).all():
        raise FloatingPointError("Initial displacement contains invalid values.")
    return displacement
