import unittest

import numpy as np
import pandas as pd
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import main as pipeline_main
from run_anchored_forecast import (
    classify_scores,
    comparison_rows,
    control_predictions,
    forecast_control_rows,
)


class TestAnchoredForecast(unittest.TestCase):
    def test_controls_do_not_use_held_out_velocities_to_predict(self):
        training = pd.DataFrame({
            "R_kpc": np.arange(1.0, 13.0),
            "Vobs_kms": np.arange(20.0, 32.0),
            "Vbar_squared_kms2": np.full(12, 100.0),
        })
        held_out = pd.DataFrame({
            "R_kpc": [13.0, 14.0],
            "Vobs_kms": [32.0, 33.0],
            "eV_kms": [2.0, 2.0],
            "Vbar_squared_kms2": [100.0, 120.0],
        })
        predictions = control_predictions(training, held_out)
        changed_targets = held_out.copy()
        changed_targets["Vobs_kms"] = [100.0, 200.0]
        for name, values in predictions.items():
            np.testing.assert_array_equal(
                values, control_predictions(training, changed_targets)[name],
            )
        np.testing.assert_array_equal(predictions["last_velocity_flat"], [31.0, 31.0])
        np.testing.assert_array_equal(predictions["last_three_mean_flat"], [30.0, 30.0])
        summaries, points = forecast_control_rows("G1", training, {"outer": held_out})
        flat = next(row for row in summaries if row["control"] == "last_velocity_flat")
        self.assertAlmostEqual(flat["chi_squared_per_point"], 0.625)
        self.assertAlmostEqual(flat["radial_leverage"], 14.0 / 12.0)
        self.assertEqual(len(points), 6)

    def test_score_classification_uses_expected_direction(self):
        self.assertEqual(classify_scores(1.0, 2.0), "burkert")
        self.assertEqual(classify_scores(2.0, 1.0), "nfw")
        self.assertEqual(classify_scores(np.nan, 1.0), "prediction_failed")

    def test_comparison_rows_keep_middle_and_outer_scores_separate(self):
        results = pd.DataFrame({
            "configuration": ["base", "base"], "family": ["radius_ceiling"] * 2,
            "galaxy_id": ["G1", "G1"], "model": ["burkert", "nfw"],
            "middle_chi_squared_per_point": [1.0, 2.0],
            "outer_chi_squared_per_point": [4.0, 3.0],
        })
        middle = comparison_rows(results, "middle")
        outer = comparison_rows(results, "outer")
        self.assertEqual(middle.iloc[0]["lower_score_model"], "burkert")
        self.assertEqual(outer.iloc[0]["lower_score_model"], "nfw")

    def test_main_option_eleven_routes_only_anchored_forecast(self):
        with TemporaryDirectory() as directory:
            with patch.object(pipeline_main, "OUTPUT_DIR", Path(directory)), \
                 patch.object(pipeline_main, "prepare_input"), \
                 patch.object(pipeline_main, "run_anchored_forecast") as forecast, \
                 patch.object(pipeline_main, "run_radial_prediction") as prediction, \
                 patch.object(pipeline_main, "run_boundary_sensitivity") as boundary, \
                 patch("builtins.input", side_effect=["11", "NGC0100,NGC6195"]), \
                 redirect_stdout(StringIO()):
                pipeline_main.main()
            forecast.assert_called_once_with(
                output_dir=Path(directory) / "anchored_forecast",
                data_path=pipeline_main.DATA_PATH,
                galaxy_ids=["NGC0100", "NGC6195"],
            )
            prediction.assert_not_called()
            boundary.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
