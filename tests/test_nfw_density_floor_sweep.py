import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import main as pipeline_main
from run_nfw_density_floor_sweep import classify_floor_profiles, validate_sweep


class TestNfwDensityFloorSweep(unittest.TestCase):
    def test_classification_identifies_floor_crossing_and_flags_failed_edge(self):
        rows = [
            {"galaxy_id": "A", "fixed_scale_radius_kpc": radius,
             "training_chi_squared": score, "density_bound_hit": bound, "fit_error": ""}
            for radius, score, bound in (
                (0.1, 8.0, ""), (1.0, 1.0, ""), (10.0, 2.0, ""),
                (100.0, 5.0, "lower"),
            )
        ]
        rows.extend([
            {"galaxy_id": "B", "fixed_scale_radius_kpc": radius,
             "training_chi_squared": score, "density_bound_hit": "", "fit_error": error}
            for radius, score, error in (
                (0.1, np.nan, "infeasible"), (1.0, 1.0, ""),
                (10.0, 2.0, ""), (100.0, 3.0, ""),
            )
        ])
        _, summary = classify_floor_profiles(rows, np.array([0.1, 1.0, 10.0, 100.0]))
        cases = summary.set_index("galaxy_id")
        self.assertEqual(cases.loc["A", "profile_status"], "closed")
        self.assertTrue(cases.loc["A", "first_upper_exclusion_at_density_floor"])
        self.assertEqual(cases.loc["B", "profile_status"], "open_both")
        self.assertTrue(cases.loc["B", "lower_grid_fit_failed"])
        self.assertTrue(cases.loc["B", "edge_status_ambiguous"])

    def test_rejects_sweep_without_original_baseline(self):
        with self.assertRaisesRegex(ValueError, "start at 1e-5"):
            validate_sweep((1e-6, 1e-7), (0.1, 1.0, 10.0), 10.0)

    def test_main_option_twenty_three_runs_floor_audit(self):
        with TemporaryDirectory() as directory:
            with patch.object(pipeline_main, "OUTPUT_DIR", Path(directory)), \
                 patch.object(pipeline_main, "prepare_input"), \
                 patch.object(pipeline_main, "run_nfw_density_floor_sweep") as sweep, \
                 patch("builtins.input", side_effect=["23", ""]), \
                 redirect_stdout(StringIO()):
                pipeline_main.main()
            sweep.assert_called_once_with(
                output_dir=Path(directory) / "nfw_density_floor_sweep",
                data_path=pipeline_main.DATA_PATH, galaxy_ids=None,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
