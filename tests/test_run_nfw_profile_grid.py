import unittest

import numpy as np
import pandas as pd
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import main as pipeline_main
from run_nfw_profile_grid import profile_summary


class TestRunNfwProfileGrid(unittest.TestCase):
    def test_main_option_fifteen_routes_only_profile_grid(self):
        with TemporaryDirectory() as directory:
            with patch.object(pipeline_main, "OUTPUT_DIR", Path(directory)), \
                 patch.object(pipeline_main, "prepare_input"), \
                 patch.object(pipeline_main, "run_nfw_profile_grid") as profile, \
                 patch.object(pipeline_main, "run_anchored_forecast") as forecast, \
                 patch("builtins.input", side_effect=["15", "NGC0100,NGC6195"]), \
                 redirect_stdout(StringIO()):
                pipeline_main.main()
            profile.assert_called_once_with(
                output_dir=Path(directory) / "nfw_profiles", data_path=pipeline_main.DATA_PATH,
                galaxy_ids=["NGC0100", "NGC6195"],
            )
            forecast.assert_not_called()

    def test_profile_summary_marks_an_interval_touching_upper_grid(self):
        rows = [
            {"galaxy_id": "G1", "fixed_scale_radius_kpc": 0.1, "training_chi_squared": 10.0, "fit_error": ""},
            {"galaxy_id": "G1", "fixed_scale_radius_kpc": 1.0, "training_chi_squared": 2.0, "fit_error": ""},
            {"galaxy_id": "G1", "fixed_scale_radius_kpc": 10.0, "training_chi_squared": 2.2, "fit_error": ""},
        ]
        profiled, summary = profile_summary(rows)
        np.testing.assert_allclose(
            profiled["delta_training_chi_squared"].to_numpy(), [8.0, 0.0, 0.2],
        )
        self.assertTrue(summary.iloc[0]["nominal_95_touches_upper_grid"])
        self.assertFalse(summary.iloc[0]["nominal_95_touches_lower_grid"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
