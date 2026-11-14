import unittest

from run_nfw_profile_normalization_grid import (
    BASELINE_SETTING,
    BULGE_TO_DISK_MASS_TO_LIGHT_RATIO,
    DISK_MASS_TO_LIGHT_GRID,
    NORMALIZATION_GRID_SETTINGS,
)


class TestNfwProfileNormalizationGrid(unittest.TestCase):
    def test_grid_is_pre_specified_and_centered_on_baseline(self):
        self.assertEqual(DISK_MASS_TO_LIGHT_GRID, (
            0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
        ))
        self.assertEqual(BASELINE_SETTING, "disk_0.50")

    def test_bulge_ratio_is_tied_to_disk_at_every_grid_value(self):
        for _, disk_ratio, bulge_ratio in NORMALIZATION_GRID_SETTINGS:
            self.assertAlmostEqual(
                bulge_ratio / disk_ratio, BULGE_TO_DISK_MASS_TO_LIGHT_RATIO,
            )


if __name__ == "__main__":
    unittest.main()
