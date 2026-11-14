import unittest

import pandas as pd
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import main as pipeline_main
from run_nfw_profile_mass_to_light import shared_observation_index, validate_stellar_settings


class TestProfileMassToLightStudy(unittest.TestCase):
    def test_main_option_eighteen_routes_only_profile_mass_to_light_study(self):
        with TemporaryDirectory() as directory:
            with patch.object(pipeline_main, "OUTPUT_DIR", Path(directory)), \
                 patch.object(pipeline_main, "prepare_input"), \
                 patch.object(pipeline_main, "run_nfw_profile_mass_to_light") as study, \
                 patch.object(pipeline_main, "run_nfw_profile_grid") as profile_grid, \
                 patch("builtins.input", side_effect=["18", "NGC0100,UGC05721"]), \
                 redirect_stdout(StringIO()):
                pipeline_main.main()
            study.assert_called_once_with(
                output_dir=Path(directory) / "nfw_profile_mass_to_light",
                data_path=pipeline_main.DATA_PATH,
                galaxy_ids=["NGC0100", "UGC05721"],
            )
            profile_grid.assert_not_called()

    def test_shared_observation_index_keeps_only_common_id_radius_rows(self):
        datasets = {
            "baseline": pd.DataFrame({"ID": ["G1", "G1", "G2"], "R_kpc": [1.0, 2.0, 1.0]}),
            "higher_disk": pd.DataFrame({"ID": ["G1", "G2", "G3"], "R_kpc": [1.0, 1.0, 1.0]}),
            "lower_bulge": pd.DataFrame({"ID": ["G1", "G2"], "R_kpc": [1.0, 1.0]}),
        }
        shared = shared_observation_index(datasets)
        self.assertEqual(list(shared), [("G1", 1.0), ("G2", 1.0)])

    def test_stellar_settings_require_the_named_baseline(self):
        with self.assertRaisesRegex(ValueError, "Baseline setting"):
            validate_stellar_settings((("case", 0.5, 0.7),), "baseline")


if __name__ == "__main__":
    unittest.main()
