"""Grid-refinement comparison using aligned Cartesian sample points."""

import unittest

import numpy as np

from config import (
    AMPLITUDE,
    CFL_SAFETY,
    DAMPING,
    DOMAIN_LENGTH,
    MEMBRANE_RADIUS,
    SIGMA,
    STRIKE_X,
    STRIKE_Y,
    WAVE_SPEED,
)
from simulation.grid import create_grid
from simulation.initial_conditions import create_initial_displacement
from simulation.solver import WaveSolver, calculate_time_step


def run_simulation(resolution: int, duration: float = 1.0):
    """Run the same model through an exact requested final time."""
    if isinstance(resolution, bool) or not isinstance(resolution, int):
        raise TypeError("Resolution must be an integer.")
    if resolution < 3 or not np.isfinite(duration) or duration <= 0:
        raise ValueError("Resolution and duration must be positive.")

    X, Y, dx, dy, domain, interior = create_grid(
        resolution, resolution, DOMAIN_LENGTH, MEMBRANE_RADIUS
    )
    initial = create_initial_displacement(
        X, Y, AMPLITUDE, STRIKE_X, STRIKE_Y, SIGMA, interior
    )
    dt, steps = calculate_time_step(
        dx,
        dy,
        duration,
        WAVE_SPEED,
        CFL_SAFETY,
    )

    solver = WaveSolver(
        initial, dx, dy, dt, WAVE_SPEED, DAMPING, domain, interior
    )
    for _ in range(steps):
        solver.step()
    return X, Y, solver.u.copy(), interior


def calculate_rms_error(coarse, fine, coarse_mask, fine_mask) -> float:
    """Compare fields on coarse-grid points active in both solutions."""
    coarse = np.asarray(coarse, dtype=float)
    fine = np.asarray(fine, dtype=float)
    coarse_mask = np.asarray(coarse_mask)
    fine_mask = np.asarray(fine_mask)
    if coarse.ndim != 2 or fine.ndim != 2:
        raise ValueError("Displacement fields must be 2D arrays.")
    if coarse.shape != coarse_mask.shape or fine.shape != fine_mask.shape:
        raise ValueError("Each mask must match its displacement field.")
    if coarse_mask.dtype != np.bool_ or fine_mask.dtype != np.bool_:
        raise TypeError("Convergence masks must be boolean.")
    if not np.isfinite(coarse).all() or not np.isfinite(fine).all():
        raise ValueError("Displacement fields must be finite.")
    if any((fine_size - 1) % (coarse_size - 1) for coarse_size, fine_size in zip(coarse.shape, fine.shape)):
        raise ValueError("Grid dimensions must align under integer refinement.")
    refinement = (fine.shape[0] - 1) // (coarse.shape[0] - 1)
    if refinement < 1 or (fine.shape[1] - 1) // (coarse.shape[1] - 1) != refinement:
        raise ValueError("Fine grid must refine both axes by the same factor.")

    fine_sampled = fine[::refinement, ::refinement]
    fine_mask_sampled = fine_mask[::refinement, ::refinement]
    if fine_sampled.shape != coarse.shape:
        raise ValueError("Grid dimensions do not produce aligned samples.")
    common_mask = coarse_mask & fine_mask_sampled
    if not np.any(common_mask):
        raise ValueError("No common active membrane points found.")
    difference = coarse[common_mask] - fine_sampled[common_mask]
    return float(np.sqrt(np.mean(difference**2)))


def run_convergence_experiment(resolutions=(51, 101, 201), duration=1.0):
    """Return successive aligned RMS errors without inferring an order."""
    results = [run_simulation(resolution, duration) for resolution in resolutions]
    errors = [
        calculate_rms_error(
            results[index][2],
            results[index + 1][2],
            results[index][3],
            results[index + 1][3],
        )
        for index in range(len(results) - 1)
    ]
    return tuple(resolutions), errors


class ConvergenceTests(unittest.TestCase):
    def test_grid_refinement_reduces_sampled_error(self):
        resolutions, errors = run_convergence_experiment((51, 101, 201), 1.0)
        self.assertEqual(resolutions, (51, 101, 201))
        self.assertEqual(len(errors), 2)
        self.assertTrue(all(np.isfinite(errors)))
        self.assertGreater(errors[0], 0.0)
        self.assertLess(errors[1], errors[0])

    def test_error_comparison_rejects_misaligned_grids(self):
        coarse = np.zeros((5, 5))
        fine = np.zeros((8, 8))
        with self.assertRaises(ValueError):
            calculate_rms_error(coarse, fine, np.ones_like(coarse, dtype=bool), np.ones_like(fine, dtype=bool))


def main() -> None:
    resolutions, errors = run_convergence_experiment()
    print("\n===== WAVESIM GRID-REFINEMENT EXPERIMENT =====")
    for coarse, fine, error in zip(resolutions, resolutions[1:], errors):
        print(f"RMS difference ({coarse} vs {fine}): {error:.8f}")
    if len(errors) >= 2 and errors[-1] < errors[-2]:
        print("Observed error decreased over these refinements.")
    else:
        print("Observed error did not decrease monotonically.")
    print("This comparison does not establish a formal convergence order.")


if __name__ == "__main__":
    main()
