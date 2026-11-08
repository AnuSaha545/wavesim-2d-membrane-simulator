"""Explicit finite-difference solver for the damped wave equation."""

from numbers import Real

import numpy as np


def calculate_cfl_number(dx: float, dy: float, dt: float, speed: float) -> float:
    """Return the two-dimensional Cartesian wave-equation CFL number."""
    values = (dx, dy, dt, speed)
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise TypeError("Grid spacing, time step, and speed must be real numbers.")
    dx, dy, dt, speed = map(float, values)
    if not np.isfinite(values).all():
        raise ValueError("Grid spacing, time step, and speed must be finite.")
    if dx <= 0 or dy <= 0 or dt <= 0 or speed <= 0:
        raise ValueError("Grid spacing, time step, and speed must be positive.")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        cfl = speed * dt * np.hypot(1.0 / dx, 1.0 / dy)
    if not np.isfinite(cfl):
        raise ValueError("CFL number is not finite for the supplied parameters.")
    return float(cfl)


def calculate_time_step(
    dx: float,
    dy: float,
    duration: float,
    speed: float,
    cfl_safety: float = 1.0,
) -> tuple[float, int]:
    """Choose a stable step size that reaches ``duration`` exactly."""
    values = (dx, dy, duration, speed, cfl_safety)
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise TypeError("Time-step inputs must be real numbers.")
    dx, dy, duration, speed, cfl_safety = map(float, values)
    if not np.isfinite(values).all():
        raise ValueError("Time-step inputs must be finite.")
    if dx <= 0 or dy <= 0 or speed <= 0:
        raise ValueError("Grid spacing and wave speed must be positive.")
    if duration <= 0 or not 0 < cfl_safety <= 1:
        raise ValueError("Duration must be positive and CFL safety must be in (0, 1].")

    maximum_step = 1.0 / (speed * np.hypot(1.0 / dx, 1.0 / dy))
    if not np.isfinite(maximum_step) or maximum_step <= 0:
        raise ValueError("Could not calculate a finite positive CFL time step.")
    nominal_step = cfl_safety * maximum_step
    requested_steps = duration / nominal_step
    if (
        not np.isfinite(requested_steps)
        or requested_steps > np.iinfo(np.intp).max
    ):
        raise ValueError("Configured duration requires an unrepresentable step count.")

    steps = max(1, int(np.ceil(requested_steps)))
    dt = duration / steps
    if calculate_cfl_number(dx, dy, dt, speed) > 1.0 + 1e-12:
        raise ValueError("Rounded time step violates the CFL condition.")
    return float(dt), steps


class WaveSolver:
    """Advance a zero-Dirichlet membrane field on a Cartesian grid.

    The centered discretization of ``u_tt + 2*gamma*u_t = c**2*laplacian(u)``
    uses a fictitious previous state to represent zero initial velocity.
    """

    def __init__(
        self,
        initial_displacement: np.ndarray,
        dx: float,
        dy: float,
        dt: float,
        speed: float,
        damping: float,
        domain_mask: np.ndarray,
        interior_mask: np.ndarray,
    ) -> None:
        values = (dx, dy, dt, speed, damping)
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
            raise TypeError("Solver parameters must be real numbers.")
        dx, dy, dt, speed, damping = map(float, values)
        if not np.isfinite(values).all():
            raise ValueError("Solver parameters must be finite.")
        if dx <= 0 or dy <= 0 or dt <= 0:
            raise ValueError("Grid spacing and time step must be positive.")
        if speed <= 0:
            raise ValueError("Wave speed must be positive.")
        if damping < 0:
            raise ValueError("Damping cannot be negative.")

        initial_displacement = np.asarray(initial_displacement, dtype=float)
        domain_mask = np.asarray(domain_mask)
        interior_mask = np.asarray(interior_mask)
        if initial_displacement.ndim != 2:
            raise ValueError("Initial displacement must be a 2D array.")
        if initial_displacement.shape != domain_mask.shape:
            raise ValueError("Initial displacement and domain mask shapes differ.")
        if domain_mask.shape != interior_mask.shape:
            raise ValueError("Domain and interior mask shapes differ.")
        if domain_mask.dtype != np.bool_ or interior_mask.dtype != np.bool_:
            raise TypeError("Domain and interior masks must be boolean.")
        if not np.any(domain_mask):
            raise ValueError("Domain mask contains no active points.")
        if not np.any(interior_mask):
            raise ValueError("Interior mask contains no active points.")
        if np.any(interior_mask & ~domain_mask):
            raise ValueError("Interior mask must be contained within domain mask.")
        if not np.isfinite(initial_displacement).all():
            raise ValueError("Initial displacement contains invalid values.")

        self.cfl_number = calculate_cfl_number(dx, dy, dt, speed)
        if self.cfl_number > 1.0 + 1e-12:
            raise ValueError(f"Unstable time step: CFL number = {self.cfl_number:.6g}")
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.speed = speed
        self.damping = damping
        self.domain_mask = domain_mask.copy()
        self.interior_mask = interior_mask.copy()
        self._courant_x_squared = (speed * dt / dx) ** 2
        self._courant_y_squared = (speed * dt / dy) ** 2
        self._damping_step = damping * dt

        self.u = initial_displacement.copy()
        self.u[~self.interior_mask] = 0.0
        initial_spatial_increment = self._spatial_increment(self.u)
        self.u_prev = self.u + 0.5 * initial_spatial_increment
        self.u_prev[~self.interior_mask] = 0.0
        self.time = 0.0
        if not np.isfinite(self.u_prev).all():
            raise FloatingPointError("Initial solver state contains invalid values.")

    def _spatial_increment(self, u: np.ndarray) -> np.ndarray:
        """Return ``c**2 * dt**2 * laplacian(u)`` without large intermediates."""
        increment = np.zeros_like(u)
        increment[1:-1, 1:-1] = (
            self._courant_x_squared
            * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2])
            + self._courant_y_squared
            * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1])
        )
        increment[~self.interior_mask] = 0.0
        return increment

    def laplacian(self, u: np.ndarray) -> np.ndarray:
        """Return the centered Cartesian Laplacian on active interior points."""
        u = np.asarray(u, dtype=float)
        if u.shape != self.interior_mask.shape:
            raise ValueError("Displacement shape does not match solver grid.")
        if not np.isfinite(u).all():
            raise ValueError("Displacement contains invalid numerical values.")

        lap = np.zeros_like(u)
        lap[1:-1, 1:-1] = (
            (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / self.dx**2
            + (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / self.dy**2
        )
        lap[~self.interior_mask] = 0.0
        if not np.isfinite(lap).all():
            raise FloatingPointError("Laplacian contains invalid numerical values.")
        return lap

    def step(self) -> np.ndarray:
        """Advance the solution by one time step and return the new field."""
        if not np.isfinite(self.u).all() or not np.isfinite(self.u_prev).all():
            raise ValueError("Solver state contains invalid numerical values.")
        if np.any(self.u[~self.interior_mask] != 0.0) or np.any(
            self.u_prev[~self.interior_mask] != 0.0
        ):
            raise ValueError("Solver state violates the fixed boundary condition.")

        spatial_increment = self._spatial_increment(self.u)
        next_u = np.zeros_like(self.u)
        gamma_dt = self._damping_step
        next_u[self.interior_mask] = (
            2.0 * self.u[self.interior_mask]
            - (1.0 - gamma_dt) * self.u_prev[self.interior_mask]
            + spatial_increment[self.interior_mask]
        ) / (1.0 + gamma_dt)

        next_time = self.time + self.dt
        if not np.isfinite(next_u).all() or not np.isfinite(next_time):
            raise FloatingPointError("Solver produced invalid numerical values.")

        self.u_prev = self.u.copy()
        self.u = next_u
        self.time = next_time
        return self.u
