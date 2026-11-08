"""Headless tests for plots, heatmaps, and animation setup."""

import unittest

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from simulation.grid import create_grid
from simulation.initial_conditions import create_initial_displacement
from simulation.solver import WaveSolver
from visualization.animation import animate_membrane
from visualization.heatmap import plot_heatmap
from visualization.plots import plot_amplitude, plot_energy


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        self.X, self.Y, dx, dy, self.domain, interior = create_grid(
            21, 21, 2.0, 1.0
        )
        self.initial = create_initial_displacement(
            self.X, self.Y, 0.5, 0.0, 0.0, 0.15, interior
        )
        self.dt = 0.8 / np.hypot(1 / dx, 1 / dy)
        self.solver = WaveSolver(
            self.initial, dx, dy, self.dt, 1.0, 0.2, self.domain, interior
        )
        self.time = np.array([0.0, 0.1, 0.2])
        self.history = {
            "max_amplitude": [0.5, 0.4, 0.3],
            "rms_displacement": [0.1, 0.08, 0.06],
            "kinetic_energy": [0.0, 0.1, 0.08],
            "potential_energy": [0.2, 0.15, 0.1],
            "total_energy": [0.2, 0.25, 0.18],
        }

    def tearDown(self):
        plt.close("all")

    def test_amplitude_and_energy_figures_have_expected_lines_and_labels(self):
        amplitude_figure, amplitude_axis = plot_amplitude(self.time, self.history)
        energy_figure, energy_axis = plot_energy(self.time, self.history)
        amplitude_figure.canvas.draw()
        energy_figure.canvas.draw()
        self.assertEqual(len(amplitude_axis.lines), 2)
        self.assertEqual(amplitude_axis.get_xlabel(), "Time (s)")
        self.assertEqual(len(energy_axis.lines), 3)
        self.assertIn("relative model units", energy_axis.get_ylabel())
        plt.close(amplitude_figure)
        plt.close(energy_figure)

    def test_invalid_plot_data_is_rejected(self):
        with self.assertRaises(ValueError):
            plot_amplitude([0.0, 0.0, 1.0], self.history)
        invalid = dict(self.history)
        invalid["total_energy"] = [0.0, np.nan, 1.0]
        with self.assertRaises(ValueError):
            plot_energy(self.time, invalid)
        with self.assertRaises(KeyError):
            plot_energy(self.time, {"kinetic_energy": [0, 1, 2]})

    def test_heatmap_masks_exterior_and_has_matching_coordinate_scale(self):
        field = self.initial.copy()
        figure, axis = plot_heatmap(self.X, self.Y, field, self.domain)
        figure.canvas.draw()
        self.assertEqual(axis.get_aspect(), 1.0)
        self.assertEqual(axis.get_xlabel(), "X Position")
        image = axis.collections[0]
        self.assertTrue(np.ma.getmaskarray(image.get_array()).any())
        plt.close(figure)

    def test_heatmap_rejects_nonfinite_and_nonzero_exterior_data(self):
        invalid = self.initial.copy()
        invalid[0, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "outside"):
            plot_heatmap(self.X, self.Y, invalid, self.domain)
        invalid = self.initial.copy()
        invalid[10, 10] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            plot_heatmap(self.X, self.Y, invalid, self.domain)

    def test_animation_initializes_and_retains_figure(self):
        animation = animate_membrane(
            self.X,
            self.Y,
            self.solver,
            duration=5 * self.dt,
            dt=self.dt,
            interval=20,
        )
        self.assertIsInstance(animation, FuncAnimation)
        self.assertEqual(len(animation._fig.axes), 2)
        animation._step()
        self.assertAlmostEqual(self.solver.time, self.dt)

    def test_animation_rejects_invalid_timing_and_shape(self):
        with self.assertRaises(ValueError):
            animate_membrane(
                self.X, self.Y, self.solver, 1.0, self.dt, 20
            )
        with self.assertRaises(ValueError):
            animate_membrane(
                self.X[:-1], self.Y[:-1], self.solver, 5 * self.dt, self.dt, 20
            )


if __name__ == "__main__":
    unittest.main()
