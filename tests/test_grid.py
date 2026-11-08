"""Grid validation and circular-domain mask tests."""

import unittest

import numpy as np

from simulation.grid import create_grid


class GridTests(unittest.TestCase):
    def test_valid_grid_dimensions_spacing_and_masks(self):
        X, Y, dx, dy, domain, interior = create_grid(41, 31, 2.0, 0.8)
        self.assertEqual(X.shape, (31, 41))
        self.assertEqual(Y.shape, (31, 41))
        self.assertAlmostEqual(dx, 0.05)
        self.assertAlmostEqual(dy, 2.0 / 30)
        self.assertTrue(domain[15, 20])
        self.assertFalse(domain[0, 0])
        self.assertTrue(np.all(interior <= domain))
        self.assertTrue(np.any(interior))

    def test_interior_contains_only_points_with_four_domain_neighbours(self):
        _, _, _, _, domain, interior = create_grid(51, 51, 2.0, 1.0)
        expected = np.zeros_like(domain)
        expected[1:-1, 1:-1] = (
            domain[1:-1, 1:-1]
            & domain[1:-1, 2:]
            & domain[1:-1, :-2]
            & domain[2:, 1:-1]
            & domain[:-2, 1:-1]
        )
        np.testing.assert_array_equal(interior, expected)

    def test_invalid_dimensions_are_rejected(self):
        for nx, ny in ((2, 10), (10, 2)):
            with self.subTest(nx=nx, ny=ny):
                with self.assertRaises(ValueError):
                    create_grid(nx, ny, 2.0, 1.0)
        with self.assertRaises(TypeError):
            create_grid(True, 10, 2.0, 1.0)
        with self.assertRaises(TypeError):
            create_grid(10.0, 10, 2.0, 1.0)

    def test_invalid_domain_and_radius_are_rejected(self):
        for length, radius in ((0.0, 1.0), (np.inf, 1.0), (2.0, 0.0), (2.0, np.nan), (1.0, 1.0)):
            with self.subTest(length=length, radius=radius):
                with self.assertRaises(ValueError):
                    create_grid(11, 11, length, radius)
        with self.assertRaises(TypeError):
            create_grid(11, 11, "2", 1.0)

    def test_resolution_without_active_interior_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "resolution"):
            create_grid(3, 3, 2.0, 0.5)


if __name__ == "__main__":
    unittest.main()
