"""Validation tests for module-level simulation configuration."""

import unittest
from unittest.mock import patch

import config


class ConfigurationTests(unittest.TestCase):
    def test_default_configuration_is_valid(self):
        config.validate_config()

    def test_invalid_grid_dimensions_are_rejected(self):
        with patch.multiple(config, NX=True):
            with self.assertRaises(TypeError):
                config.validate_config()
        with patch.multiple(config, NY=2):
            with self.assertRaises(ValueError):
                config.validate_config()

    def test_nonfinite_or_nonpositive_physics_are_rejected(self):
        cases = (
            ("DOMAIN_LENGTH", float("nan")),
            ("MEMBRANE_RADIUS", 0.0),
            ("WAVE_SPEED", float("inf")),
            ("DAMPING", -0.1),
            ("AMPLITUDE", -0.1),
            ("SIGMA", 0.0),
            ("SIMULATION_DURATION", float("nan")),
        )
        for name, value in cases:
            with self.subTest(name=name):
                with patch.multiple(config, **{name: value}):
                    with self.assertRaises(ValueError):
                        config.validate_config()

    def test_invalid_strike_and_cfl_factor_are_rejected(self):
        with patch.multiple(config, STRIKE_X=1.0, STRIKE_Y=1.0):
            with self.assertRaisesRegex(ValueError, "Strike"):
                config.validate_config()
        for value in (0.0, 1.01, float("inf")):
            with self.subTest(cfl_safety=value):
                with patch.multiple(config, CFL_SAFETY=value):
                    with self.assertRaises(ValueError):
                        config.validate_config()

    def test_animation_interval_must_be_positive_integer(self):
        with patch.multiple(config, ANIMATION_INTERVAL=0):
            with self.assertRaises(ValueError):
                config.validate_config()
        with patch.multiple(config, ANIMATION_INTERVAL=1.5):
            with self.assertRaises(TypeError):
                config.validate_config()


if __name__ == "__main__":
    unittest.main()
