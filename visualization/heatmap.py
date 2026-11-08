"""Two-dimensional displacement heatmap for the circular membrane."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def plot_heatmap(
    X: np.ndarray,
    Y: np.ndarray,
    displacement: np.ndarray,
    domain_mask: np.ndarray,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one finite displacement field, masking the exterior of the circle."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    domain_mask = np.asarray(domain_mask)

    if X.ndim != 2 or Y.ndim != 2 or displacement.ndim != 2:
        raise ValueError("Coordinates and displacement must be 2D arrays.")
    if X.shape != Y.shape or X.shape != displacement.shape:
        raise ValueError("Coordinates and displacement must have matching shapes.")
    if domain_mask.shape != X.shape:
        raise ValueError("Domain mask must match the coordinate arrays.")
    if domain_mask.dtype != np.bool_:
        raise TypeError("Domain mask must be boolean.")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("Coordinates must contain only finite values.")
    if not np.isfinite(displacement).all():
        raise ValueError("Displacement must contain only finite values.")
    if not np.any(domain_mask):
        raise ValueError("Domain mask contains no membrane points.")
    if np.any(displacement[~domain_mask] != 0.0):
        raise ValueError("Displacement outside the membrane must be zero.")

    limit = max(float(np.max(np.abs(displacement[domain_mask]))), 1e-12)
    norm = Normalize(vmin=-limit, vmax=limit)
    masked_displacement = np.ma.masked_where(~domain_mask, displacement)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.pcolormesh(
        X,
        Y,
        masked_displacement,
        cmap="seismic",
        norm=norm,
        shading="auto",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("WaveSim | Membrane Displacement")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Displacement (model units)")
    fig.tight_layout()
    return fig, ax