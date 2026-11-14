import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

import main as pipeline_main
from run_radial_prediction import classify_prediction_scores, comparison_rows, ensure_input_data


class TestRadialPrediction(unittest.TestCase):
    def test_missing_csv_is_created_from_bundled_text(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "GALAXY_CSVDATA.csv"
            with patch("run_radial_prediction.convert_rtf_to_text", return_value="text"), \
                 patch("run_radial_prediction.convert_text_to_csv") as convert:
                convert.side_effect = lambda _input, output: Path(output).touch()
                self.assertEqual(ensure_input_data(destination), destination)
            convert.assert_called_once()

    def test_prediction_score_classification_uses_expected_direction(self):
        self.assertEqual(classify_prediction_scores(1.0, 2.0), "burkert")
        self.assertEqual(classify_prediction_scores(2.0, 1.0), "nfw")
        self.assertEqual(classify_prediction_scores(1.0, 1.0 + 1e-9), "near_tie")
        self.assertEqual(classify_prediction_scores(np.nan, 1.0), "prediction_failed")

    def test_comparison_rows_match_only_identical_configuration_predictions(self):
        results = pd.DataFrame({
            "configuration": ["a", "a", "b", "b"],
            "family": ["radius_ceiling"] * 4,
            "galaxy_id": ["G1", "G1", "G1", "G1"],
            "model": ["burkert", "nfw", "burkert", "nfw"],
            "held_out_chi_squared_per_point": [1.0, 2.0, 3.0, 2.0],
        })
        comparisons = comparison_rows(results).set_index("configuration")
        self.assertEqual(comparisons.loc["a", "lower_held_out_score_model"], "burkert")
        self.assertEqual(comparisons.loc["b", "lower_held_out_score_model"], "nfw")
        self.assertEqual(
            comparisons.loc["a", "delta_held_out_chi_squared_per_point_nfw_minus_burkert"],
            1.0,
        )

    def test_main_option_eight_routes_only_radial_prediction(self):
        with TemporaryDirectory() as directory:
            with patch.object(pipeline_main, "OUTPUT_DIR", Path(directory)), \
                 patch.object(pipeline_main, "prepare_input"), \
                 patch.object(pipeline_main, "run_radial_prediction") as prediction, \
                 patch.object(pipeline_main, "run_boundary_sensitivity") as boundary, \
                 patch.object(pipeline_main, "run_validation") as validation, \
                 patch("builtins.input", side_effect=["8", "NGC0100,NGC6195", "0.2"]), \
                 redirect_stdout(StringIO()):
                pipeline_main.main()
            prediction.assert_called_once_with(
                output_dir=Path(directory) / "radial_prediction",
                data_path=pipeline_main.DATA_PATH,
                galaxy_ids=["NGC0100", "NGC6195"], holdout_fraction=0.2,
            )
            boundary.assert_not_called()
            validation.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
