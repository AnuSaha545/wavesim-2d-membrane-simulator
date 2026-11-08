"""Three-dimensional membrane animation."""

from numbers import Integral, Real

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from simulation.solver import WaveSolver


def animate_membrane(
    X: np.ndarray,
    Y: np.ndarray,
    solver: WaveSolver,
    duration: float,
    dt: float,
    interval: int,
) -> FuncAnimation:
    """Create a 3D animation; retain the returned object until ``plt.show``."""
    values = (duration, dt)
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise TypeError("Duration and time step must be real numbers.")
    duration, dt = map(float, values)
    if not np.isfinite(values).all() or duration <= 0 or dt <= 0:
        raise ValueError("Duration and time step must be positive and finite.")
    if isinstance(interval, bool) or not isinstance(interval, Integral):
        raise TypeError("Animation interval must be a positive integer.")
    if interval <= 0:
        raise ValueError("Animation interval must be a positive integer.")

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or X.shape != Y.shape or X.shape != solver.u.shape:
        raise ValueError("Grid coordinates and solver displacement must match.")
    if solver.domain_mask.shape != X.shape or solver.domain_mask.dtype != np.bool_:
        raise ValueError("Solver domain mask must be boolean and match the grid.")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("Grid coordinates must contain only finite values.")
    if not np.isfinite(solver.u).all() or not np.isfinite(solver.time):
        raise ValueError("Solver state must contain only finite values.")
    if not np.isclose(solver.dt, dt, rtol=1e-12, atol=0.0):
        raise ValueError("Animation time step must match the solver time step.")
    dt = solver.dt
    frame_count = int(round(duration / dt))
    if frame_count < 1 or not np.isclose(frame_count * dt, duration):
        raise ValueError("Duration must be an integer multiple of the time step.")

    initial_max = float(np.max(np.abs(solver.u)))
    z_limit = max(initial_max * 1.2, 1e-6)
    x_min, x_max = float(np.min(X)), float(np.max(X))
    y_min, y_max = float(np.min(Y)), float(np.max(Y))
    norm = Normalize(vmin=-z_limit, vmax=z_limit)
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")
    color_map = ScalarMappable(norm=norm, cmap="seismic")
    color_map.set_array([])
    colorbar = fig.colorbar(color_map, ax=ax, shrink=0.7, pad=0.12)
    colorbar.set_label("Displacement (model units)", fontsize=12)

    surface = [None]

    def get_surface_data() -> np.ndarray:
        return np.where(solver.domain_mask, solver.u, np.nan)

    def draw_surface() -> object:
        nonlocal z_limit
        current_max = float(np.max(np.abs(solver.u)))
        if current_max > z_limit:
            z_limit = current_max * 1.1
            norm.vmin = -z_limit
            norm.vmax = z_limit
            ax.set_zlim(-z_limit, z_limit)
            colorbar.update_normal(color_map)
        if surface[0] is not None:
            surface[0].remove()
        surface[0] = ax.plot_surface(
            X,
            Y,
            get_surface_data(),
            cmap="seismic",
            norm=norm,
            edgecolor="none",
            antialiased=True,
            shade=False,
        )
        return surface[0]

    ax.set_xlabel("X Position", labelpad=10)
    ax.set_ylabel("Y Position", labelpad=10)
    ax.set_zlabel("Displacement (model units)", labelpad=10)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(-z_limit, z_limit)
    ax.set_box_aspect((x_max - x_min, y_max - y_min, 0.5 * (x_max - x_min)))
    ax.set_title(
        f"WaveSim | Circular Drum Membrane | Time: {solver.time:.3f} s",
        fontsize=15,
        fontweight="bold",
        pad=25,
    )

    def initialize() -> tuple[object, ...]:
        return (draw_surface(),)

    def update(frame: int) -> tuple[object, ...]:
        solver.step()
        surface_artist = draw_surface()
        ax.set_title(
            f"WaveSim | Circular Drum Membrane | Time: {solver.time:.3f} s "
            f"| Frame {frame + 1}/{frame_count}",
            fontsize=15,
            fontweight="bold",
            pad=25,
        )
        return (surface_artist,)

    animation = FuncAnimation(
        fig,
        update,
        init_func=initialize,
        frames=frame_count,
        interval=int(interval),
        blit=False,
        repeat=False,
        cache_frame_data=False,
    )
    fig.tight_layout()
    return animation
