"""Discoverable tests for the finite-difference wave solver."""

import unittest

import numpy as np

from simulation.grid import create_grid
from simulation.initial_conditions import create_initial_displacement
from simulation.solver import (
    WaveSolver,
    calculate_cfl_number,
    calculate_time_step,
)


def create_test_setup(nx: int = 41, ny: int = 41):
    X, Y, dx, dy, domain, interior = create_grid(nx, ny, 2.0, 1.0)
    initial = create_initial_displacement(X, Y, 0.5, 0.0, 0.0, 0.15, interior)
    dt = 0.9 / (np.hypot(1 / dx, 1 / dy))
    return X, Y, dx, dy, dt, domain, interior, initial


def create_solver(initial, dx, dy, dt, domain, interior, damping=0.35):
    return WaveSolver(initial, dx, dy, dt, 1.0, damping, domain, interior)


class WaveSolverTests(unittest.TestCase):
    def setUp(self):
        (
            self.X,
            self.Y,
            self.dx,
            self.dy,
            self.dt,
            self.domain,
            self.interior,
            self.initial,
        ) = create_test_setup()

    def test_zero_displacement_remains_zero(self):
        solver = create_solver(
            np.zeros_like(self.X),
            self.dx,
            self.dy,
            self.dt,
            self.domain,
            self.interior,
        )
        for _ in range(200):
            solver.step()
        np.testing.assert_array_equal(solver.u, 0.0)

    def test_wave_propagates_with_finite_values_and_fixed_boundary(self):
        solver = create_solver(
            self.initial,
            self.dx,
            self.dy,
            self.dt,
            self.domain,
            self.interior,
        )
        initial = solver.u.copy()
        for _ in range(30):
            solver.step()
        self.assertFalse(np.allclose(solver.u, initial))
        self.assertTrue(np.isfinite(solver.u).all())
        self.assertTrue(np.isfinite(solver.u_prev).all())
        np.testing.assert_array_equal(solver.u[~self.interior], 0.0)
        np.testing.assert_array_equal(solver.u[~self.domain], 0.0)
        self.assertAlmostEqual(solver.time, 30 * self.dt)

    def test_initialization_encodes_zero_initial_velocity(self):
        solver = create_solver(
            self.initial,
            self.dx,
            self.dy,
            self.dt,
            self.domain,
            self.interior,
            damping=0.8,
        )
        first_state = solver.u.copy()
        expected_acceleration_increment = solver._spatial_increment(first_state)
        solver.step()
        np.testing.assert_allclose(
            solver.u,
            first_state + 0.5 * expected_acceleration_increment,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_stronger_damping_reduces_late_rms_displacement(self):
        weak = create_solver(
            self.initial, self.dx, self.dy, self.dt, self.domain, self.interior, 0.1
        )
        strong = create_solver(
            self.initial, self.dx, self.dy, self.dt, self.domain, self.interior, 1.0
        )
        for _ in range(100):
            weak.step()
            strong.step()
        weak_rms = np.sqrt(np.mean(weak.u[self.domain] ** 2))
        strong_rms = np.sqrt(np.mean(strong.u[self.domain] ** 2))
        self.assertLess(strong_rms, weak_rms)

    def test_large_damping_step_remains_finite_under_centered_scheme(self):
        solver = create_solver(
            self.initial,
            self.dx,
            self.dy,
            self.dt,
            self.domain,
            self.interior,
            damping=100.0 / self.dt,
        )
        for _ in range(100):
            solver.step()
        self.assertTrue(np.isfinite(solver.u).all())

    def test_cfl_violation_is_rejected(self):
        unstable_dt = 1.1 / np.hypot(1 / self.dx, 1 / self.dy)
        with self.assertRaisesRegex(ValueError, "CFL"):
            create_solver(
                self.initial,
                self.dx,
                self.dy,
                unstable_dt,
                self.domain,
                self.interior,
            )

    def test_long_run_remains_finite(self):
        solver = create_solver(
            self.initial,
            self.dx,
            self.dy,
            self.dt,
            self.domain,
            self.interior,
        )
        for _ in range(1000):
            solver.step()
        self.assertTrue(np.isfinite(solver.u).all())
        self.assertTrue(np.isfinite(solver.u_prev).all())

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            create_solver(
                self.initial,
                0.0,
                self.dy,
                self.dt,
                self.domain,
                self.interior,
            )
        with self.assertRaises(ValueError):
            create_solver(
                self.initial,
                self.dx,
                self.dy,
                self.dt,
                self.domain,
                self.interior[:-1],
            )
        invalid = self.initial.copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            create_solver(
                invalid,
                self.dx,
                self.dy,
                self.dt,
                self.domain,
                self.interior,
            )

    def test_mutated_invalid_state_is_rejected(self):
        solver = create_solver(
            self.initial,
            self.dx,
            self.dy,
            self.dt,
            self.domain,
            self.interior,
        )
        solver.u[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "state"):
            solver.step()

    def test_cfl_number_matches_formula(self):
        expected = self.dt * np.sqrt(self.dx**-2 + self.dy**-2)
        self.assertAlmostEqual(
            calculate_cfl_number(self.dx, self.dy, self.dt, 1.0),
            expected,
        )

    def test_time_step_reaches_duration_and_respects_cfl(self):
        dt, steps = calculate_time_step(
            self.dx, self.dy, 2.0, 1.0, cfl_safety=0.9
        )
        self.assertAlmostEqual(dt * steps, 2.0)
        self.assertLessEqual(calculate_cfl_number(self.dx, self.dy, dt, 1.0), 0.9)


if __name__ == "__main__":
    unittest.main()
