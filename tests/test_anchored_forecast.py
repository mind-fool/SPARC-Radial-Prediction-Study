import unittest

import numpy as np
import pandas as pd
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import main as pipeline_main
from run_anchored_forecast import classify_scores, comparison_rows


class TestAnchoredForecast(unittest.TestCase):
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
