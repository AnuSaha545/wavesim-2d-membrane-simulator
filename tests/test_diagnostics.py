"""Tests for displacement and discrete-energy diagnostics."""

import unittest

import numpy as np

from simulation.diagnostics import calculate_diagnostics
from simulation.grid import create_grid
from simulation.initial_conditions import create_initial_displacement
from simulation.solver import WaveSolver


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        X, Y, dx, dy, domain, interior = create_grid(41, 41, 2.0, 1.0)
        initial = create_initial_displacement(
            X, Y, 0.5, 0.0, 0.0, 0.15, interior
        )
        dt = 0.9 / np.hypot(1 / dx, 1 / dy)
        self.solver = WaveSolver(
            initial, dx, dy, dt, 1.0, 0.35, domain, interior
        )
        self.domain = domain
        self.interior = interior

    def test_amplitude_and_rms_definitions(self):
        result = calculate_diagnostics(self.solver)
        expected_max = np.max(np.abs(self.solver.u[self.domain]))
        expected_rms = np.sqrt(np.mean(self.solver.u[self.domain] ** 2))
        self.assertAlmostEqual(result["max_amplitude"], expected_max)
        self.assertAlmostEqual(result["rms_displacement"], expected_rms)

    def test_energy_terms_are_finite_and_sum(self):
        self.solver.step()
        result = calculate_diagnostics(self.solver)
        for key in ("kinetic_energy", "potential_energy", "total_energy"):
            self.assertTrue(np.isfinite(result[key]))
            self.assertGreaterEqual(result[key], 0.0)
        self.assertAlmostEqual(
            result["total_energy"],
            result["kinetic_energy"] + result["potential_energy"],
        )

    def test_zero_displacement_has_zero_diagnostics(self):
        zero = np.zeros_like(self.solver.u)
        solver = WaveSolver(
            zero,
            self.solver.dx,
            self.solver.dy,
            self.solver.dt,
            self.solver.speed,
            self.solver.damping,
            self.solver.domain_mask,
            self.solver.interior_mask,
        )
        result = calculate_diagnostics(solver)
        for value in result.values():
            self.assertEqual(value, 0.0)

    def test_invalid_solver_state_is_rejected(self):
        self.solver.u[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "invalid"):
            calculate_diagnostics(self.solver)

    def test_nonzero_fixed_boundary_is_rejected(self):
        self.solver.u[0, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "boundary"):
            calculate_diagnostics(self.solver)


if __name__ == "__main__":
    unittest.main()
