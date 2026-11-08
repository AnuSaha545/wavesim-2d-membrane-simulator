"""Time-series visualizations for displacement and energy diagnostics."""

import numpy as np
import matplotlib.pyplot as plt


def validate_plot_data(
    time: np.ndarray,
    history: dict[str, object],
    required_keys: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Validate times and return numeric arrays for requested histories."""
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or len(time) < 2:
        raise ValueError("Time data must contain at least two values.")
    if not np.isfinite(time).all():
        raise ValueError("Time data contains invalid values.")
    if np.any(np.diff(time) <= 0):
        raise ValueError("Time values must be strictly increasing.")

    values_by_key: dict[str, np.ndarray] = {}
    for key in required_keys:
        if key not in history:
            raise KeyError(f"Missing measurement: {key}")
        values = np.asarray(history[key], dtype=float)
        if values.shape != time.shape:
            raise ValueError(f"Measurement '{key}' does not match time array.")
        if not np.isfinite(values).all():
            raise ValueError(f"Measurement '{key}' contains invalid values.")
        values_by_key[key] = values
    return time, values_by_key


def plot_amplitude(
    time: np.ndarray,
    history: dict[str, object],
) -> tuple[plt.Figure, plt.Axes]:
    """Plot maximum and RMS displacement against simulation time."""
    time, values = validate_plot_data(
        time,
        history,
        ["max_amplitude", "rms_displacement"],
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        time,
        values["max_amplitude"],
        label="Maximum Displacement",
        color="red",
        linewidth=2,
    )
    ax.plot(
        time,
        values["rms_displacement"],
        label="RMS Displacement",
        color="blue",
        linewidth=2,
    )
    ax.set_title("WaveSim | Displacement Analysis", fontsize=15, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (model units)")
    ax.set_xlim(time[0], time[-1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_energy(
    time: np.ndarray,
    history: dict[str, object],
) -> tuple[plt.Figure, plt.Axes]:
    """Plot kinetic, potential, and approximate total model-relative energy."""
    time, values = validate_plot_data(
        time,
        history,
        ["kinetic_energy", "potential_energy", "total_energy"],
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(time, values["kinetic_energy"], label="Kinetic Energy", linewidth=2)
    ax.plot(time, values["potential_energy"], label="Potential Energy", linewidth=2)
    ax.plot(
        time,
        values["total_energy"],
        label="Approximate Total Energy",
        linewidth=2.5,
        color="black",
    )
    ax.set_title(
        "WaveSim | Approximate Discrete Energy",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (relative model units)")
    ax.set_xlim(time[0], time[-1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax
