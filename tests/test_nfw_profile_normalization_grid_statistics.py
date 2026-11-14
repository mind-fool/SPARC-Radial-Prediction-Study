import unittest
from unittest.mock import patch

from nfw_profile_normalization_grid_statistics import (
    analyze_nfw_profile_normalization_grid,
)
from run_nfw_profile_normalization_grid import BASELINE_SETTING


class TestNormalizationGridStatistics(unittest.TestCase):
    def test_wrapper_uses_pre_specified_disk_point_five_baseline(self):
        with patch(
            "nfw_profile_normalization_grid_statistics.analyze_nfw_profile_mass_to_light",
        ) as analyze:
            analyze_nfw_profile_normalization_grid("input-run", output_dir="output-dir")
        analyze.assert_called_once_with(
            "input-run", output_dir="output-dir", baseline=BASELINE_SETTING,
        )


if __name__ == "__main__":
    unittest.main()
