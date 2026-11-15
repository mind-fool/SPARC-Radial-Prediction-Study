import unittest

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import main as pipeline_main
from run_nfw_profile_normalization_grid import (
    BASELINE_SETTING,
    BULGE_TO_DISK_MASS_TO_LIGHT_RATIO,
    DISK_MASS_TO_LIGHT_GRID,
    NORMALIZATION_GRID_SETTINGS,
)


class TestNfwProfileNormalizationGrid(unittest.TestCase):
    def test_main_option_twenty_routes_only_normalization_grid(self):
        with TemporaryDirectory() as directory:
            with patch.object(pipeline_main, "OUTPUT_DIR", Path(directory)), \
                 patch.object(pipeline_main, "prepare_input"), \
                 patch.object(pipeline_main, "run_nfw_profile_normalization_grid") as grid, \
                 patch.object(pipeline_main, "run_nfw_profile_mass_to_light") as settings, \
                 patch("builtins.input", side_effect=["20", "NGC0100,UGC05721", ""]), \
                 redirect_stdout(StringIO()):
                pipeline_main.main()
            grid.assert_called_once_with(
                output_dir=Path(directory) / "nfw_profile_normalization_grid",
                data_path=pipeline_main.DATA_PATH,
                galaxy_ids=["NGC0100", "UGC05721"],
                density_bounds=(1e-5, 10.0),
            )
            settings.assert_not_called()

    def test_main_option_twenty_accepts_low_density_floor(self):
        with TemporaryDirectory() as directory:
            with patch.object(pipeline_main, "OUTPUT_DIR", Path(directory)), \
                 patch.object(pipeline_main, "prepare_input"), \
                 patch.object(pipeline_main, "run_nfw_profile_normalization_grid") as grid, \
                 patch("builtins.input", side_effect=["20", "", "1e-9"]), \
                 redirect_stdout(StringIO()):
                pipeline_main.main()
            grid.assert_called_once_with(
                output_dir=Path(directory) / "nfw_profile_normalization_grid",
                data_path=pipeline_main.DATA_PATH,
                galaxy_ids=None,
                density_bounds=(1e-9, 10.0),
            )

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
