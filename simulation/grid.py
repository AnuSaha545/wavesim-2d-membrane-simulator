"""Cartesian grid construction for the circular membrane."""

from numbers import Integral, Real

import numpy as np


def create_grid(
    nx: int,
    ny: int,
    length: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    """Create a square physical domain and circular membrane masks.

    The returned interior mask contains only points whose four Cartesian
    neighbours lie inside the circle. All other points are fixed at zero by
    the solver, giving a stair-stepped Cartesian approximation to the curved
    boundary.
    """
    if isinstance(nx, bool) or isinstance(ny, bool):
        raise TypeError("Grid dimensions must be integers.")
    if not isinstance(nx, Integral) or not isinstance(ny, Integral):
        raise TypeError("Grid dimensions must be integers.")
    if nx < 3 or ny < 3:
        raise ValueError("Grid dimensions must be at least 3.")
    if (
        isinstance(length, bool)
        or isinstance(radius, bool)
        or not isinstance(length, Real)
        or not isinstance(radius, Real)
    ):
        raise TypeError("Domain length and membrane radius must be real numbers.")

    length = float(length)
    radius = float(radius)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Domain length must be positive and finite.")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Membrane radius must be positive and finite.")
    if radius > length / 2:
        raise ValueError("Membrane radius cannot exceed half the domain length.")

    x = np.linspace(-length / 2, length / 2, int(nx))
    y = np.linspace(-length / 2, length / 2, int(ny))
    X, Y = np.meshgrid(x, y)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    domain_mask = np.hypot(X, Y) <= radius
    interior_mask = np.zeros((int(ny), int(nx)), dtype=bool)
    interior_mask[1:-1, 1:-1] = (
        domain_mask[1:-1, 1:-1]
        & domain_mask[1:-1, 2:]
        & domain_mask[1:-1, :-2]
        & domain_mask[2:, 1:-1]
        & domain_mask[:-2, 1:-1]
    )

    if not np.any(domain_mask):
        raise ValueError("Grid resolution is too low to represent the membrane.")
    if not np.any(interior_mask):
        raise ValueError("Grid resolution is too low for this membrane radius.")

    return X, Y, dx, dy, domain_mask, interior_mask
