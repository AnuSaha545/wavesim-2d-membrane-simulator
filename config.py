"""Default model parameters and configuration validation."""

from numbers import Integral, Real

import numpy as np


# Grid configuration
NX = 101
NY = 101

DOMAIN_LENGTH = 2.0
MEMBRANE_RADIUS = 1.0

# Physical model parameters
WAVE_SPEED = 1.0
DAMPING = 0.35

# Initial disturbance parameters
AMPLITUDE = 0.5
STRIKE_X = 0.0
STRIKE_Y = 0.0
SIGMA = 0.15

# Numerical parameters
CFL_SAFETY = 0.9
SIMULATION_DURATION = 5.0

# Visualization parameters
ANIMATION_INTERVAL = 30


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def validate_config() -> None:
    """Validate all module-level simulation settings."""
    for name, value in (("NX", NX), ("NY", NY)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer.")
        if value < 3:
            raise ValueError(f"{name} must be at least 3.")

    length = _finite_real("Domain length", DOMAIN_LENGTH)
    radius = _finite_real("Membrane radius", MEMBRANE_RADIUS)
    speed = _finite_real("Wave speed", WAVE_SPEED)
    damping = _finite_real("Damping", DAMPING)
    amplitude = _finite_real("Amplitude", AMPLITUDE)
    strike_x = _finite_real("Strike X coordinate", STRIKE_X)
    strike_y = _finite_real("Strike Y coordinate", STRIKE_Y)
    sigma = _finite_real("Gaussian spread", SIGMA)
    cfl_safety = _finite_real("CFL safety factor", CFL_SAFETY)
    duration = _finite_real("Simulation duration", SIMULATION_DURATION)

    if length <= 0:
        raise ValueError("Domain length must be positive.")
    if radius <= 0:
        raise ValueError("Membrane radius must be positive.")
    if radius > length / 2:
        raise ValueError("Membrane radius exceeds computational domain.")
    if speed <= 0:
        raise ValueError("Wave speed must be positive.")
    if damping < 0:
        raise ValueError("Damping cannot be negative.")
    if amplitude < 0:
        raise ValueError("Amplitude cannot be negative.")
    if sigma <= 0:
        raise ValueError("Gaussian spread must be positive.")
    if np.hypot(strike_x, strike_y) > radius:
        raise ValueError("Strike position must lie inside the membrane.")
    if not 0 < cfl_safety <= 1:
        raise ValueError("CFL safety factor must be in (0, 1].")
    if duration <= 0:
        raise ValueError("Simulation duration must be positive.")

    if isinstance(ANIMATION_INTERVAL, bool) or not isinstance(
        ANIMATION_INTERVAL, Integral
    ):
        raise TypeError("Animation interval must be an integer.")
    if ANIMATION_INTERVAL <= 0:
        raise ValueError("Animation interval must be positive.")
