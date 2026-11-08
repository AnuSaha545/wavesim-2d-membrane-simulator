"""Gaussian initial-condition validation and shape tests."""

import unittest

import numpy as np

from simulation.grid import create_grid
from simulation.initial_conditions import create_initial_displacement


class InitialConditionTests(unittest.TestCase):
    def setUp(self):
        self.X, self.Y, _, _, _, self.interior = create_grid(41, 41, 2.0, 1.0)

    def test_gaussian_is_centered_and_zero_on_fixed_points(self):
        displacement = create_initial_displacement(
            self.X, self.Y, 0.5, 0.0, 0.0, 0.15, self.interior
        )
        center = np.unravel_index(np.argmax(displacement), displacement.shape)
        self.assertEqual(self.X[center], 0.0)
        self.assertEqual(self.Y[center], 0.0)
        self.assertAlmostEqual(displacement[center], 0.5)
        np.testing.assert_array_equal(displacement[~self.interior], 0.0)

    def test_zero_amplitude_returns_zero_field(self):
        displacement = create_initial_displacement(
            self.X, self.Y, 0.0, 0.0, 0.0, 0.15, self.interior
        )
        np.testing.assert_array_equal(displacement, 0.0)

    def test_shifted_strike_has_peak_near_strike(self):
        displacement = create_initial_displacement(
            self.X, self.Y, 0.5, 0.2, -0.1, 0.15, self.interior
        )
        peak = np.unravel_index(np.argmax(displacement), displacement.shape)
        self.assertLessEqual(np.hypot(self.X[peak] - 0.2, self.Y[peak] + 0.1), np.hypot(0.05, 0.05))

    def test_invalid_parameters_and_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            create_initial_displacement(self.X, self.Y, -1.0, 0.0, 0.0, 0.1, self.interior)
        with self.assertRaises(ValueError):
            create_initial_displacement(self.X, self.Y, 1.0, 0.0, 0.0, 0.0, self.interior)
        with self.assertRaises(ValueError):
            create_initial_displacement(self.X, self.Y, 1.0, 1.1, 0.0, 0.1, self.interior)
        with self.assertRaises(ValueError):
            create_initial_displacement(self.X, self.Y[:-1], 1.0, 0.0, 0.0, 0.1, self.interior)
        with self.assertRaises(ValueError):
            create_initial_displacement(self.X, np.full_like(self.Y, np.nan), 1.0, 0.0, 0.0, 0.1, self.interior)
        with self.assertRaises(ValueError):
            create_initial_displacement(self.X, self.Y, np.inf, 0.0, 0.0, 0.1, self.interior)

    def test_non_boolean_mask_is_rejected(self):
        with self.assertRaises(TypeError):
            create_initial_displacement(
                self.X, self.Y, 1.0, 0.0, 0.0, 0.1, self.interior.astype(int)
            )


if __name__ == "__main__":
    unittest.main()
