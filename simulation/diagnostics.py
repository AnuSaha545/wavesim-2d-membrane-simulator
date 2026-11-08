"""Model-relative displacement and discrete energy diagnostics."""

import numpy as np

from simulation.solver import WaveSolver


def calculate_diagnostics(solver: WaveSolver) -> dict[str, float]:
    """Measure displacement and approximate discrete mechanical energy.

    The energy uses a centered half-step velocity and nearest-neighbour
    displacement differences, including bonds from active points to fixed
    zero-valued points. It is a model-relative diagnostic, not energy in
    joules, and is not an exact conserved quantity for the damped discretization.
    """
    u = np.asarray(solver.u, dtype=float)
    u_prev = np.asarray(solver.u_prev, dtype=float)
    domain = np.asarray(solver.domain_mask)
    interior = np.asarray(solver.interior_mask)

    if u.ndim != 2 or u.shape != u_prev.shape:
        raise ValueError("Solver displacement arrays must be matching 2D arrays.")
    if u.shape != domain.shape or u.shape != interior.shape:
        raise ValueError("Solver arrays and masks must have matching shapes.")
    if domain.dtype != np.bool_ or interior.dtype != np.bool_:
        raise TypeError("Solver masks must be boolean.")
    if np.any(interior & ~domain):
        raise ValueError("Interior mask must be contained within domain mask.")
    if not np.any(domain) or not np.any(interior):
        raise ValueError("Simulation domain contains no active points.")
    if not np.isfinite(u).all() or not np.isfinite(u_prev).all():
        raise ValueError("Displacement contains invalid numerical values.")
    if np.any(u[~interior] != 0.0) or np.any(u_prev[~interior] != 0.0):
        raise ValueError("Solver state violates the fixed boundary condition.")

    parameters = np.asarray(
        (solver.dx, solver.dy, solver.dt, solver.speed, solver.damping),
        dtype=float,
    )
    if not np.isfinite(parameters).all():
        raise ValueError("Solver parameters must be finite.")
    dx, dy, dt, speed, damping = parameters
    if dx <= 0 or dy <= 0 or dt <= 0 or speed <= 0 or damping < 0:
        raise ValueError("Invalid solver parameters.")

    max_amplitude = float(np.max(np.abs(u[domain])))
    rms_displacement = float(np.sqrt(np.mean(u[domain] ** 2)))
    velocity = (u - u_prev) / dt
    midpoint_displacement = 0.5 * (u + u_prev)

    kinetic_energy = float(
        0.5 * np.sum(velocity[interior] ** 2) * dx * dy
    )

    horizontal_mask = interior[:, 1:] | interior[:, :-1]
    horizontal_difference = midpoint_displacement[:, 1:] - midpoint_displacement[:, :-1]
    horizontal_energy = float(
        np.sum(horizontal_difference[horizontal_mask] ** 2) * dy / dx
    )

    vertical_mask = interior[1:, :] | interior[:-1, :]
    vertical_difference = midpoint_displacement[1:, :] - midpoint_displacement[:-1, :]
    vertical_energy = float(
        np.sum(vertical_difference[vertical_mask] ** 2) * dx / dy
    )

    potential_energy = float(0.5 * speed**2 * (horizontal_energy + vertical_energy))
    total_energy = kinetic_energy + potential_energy
    values = np.asarray(
        (
            max_amplitude,
            rms_displacement,
            kinetic_energy,
            potential_energy,
            total_energy,
        )
    )
    if not np.isfinite(values).all():
        raise FloatingPointError("Energy diagnostics produced invalid values.")

    return {
        "max_amplitude": max_amplitude,
        "rms_displacement": rms_displacement,
        "kinetic_energy": kinetic_energy,
        "potential_energy": potential_energy,
        "total_energy": total_energy,
    }
