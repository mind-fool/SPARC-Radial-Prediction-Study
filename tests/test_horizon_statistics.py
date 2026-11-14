import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import main as pipeline_main
from horizon_statistics import common_galaxy_ids


class TestHorizonStatistics(unittest.TestCase):
    def test_main_option_ten_routes_selected_completed_runs(self):
        runs = [Path("run_40"), Path("run_30"), Path("run_20")]
        with patch.object(pipeline_main, "prepare_input"), \
             patch.object(pipeline_main, "completed_prediction_runs", return_value=runs), \
             patch.object(pipeline_main, "analyze_horizons") as analyze, \
             patch("builtins.input", side_effect=["10", "1,2,3"]), \
             redirect_stdout(StringIO()):
            pipeline_main.main()
        analyze.assert_called_once_with(runs)

    def test_common_galaxy_ids_requires_both_models_in_each_horizon(self):
        first = pd.DataFrame({
            "galaxy_id": ["A", "A", "B", "B", "C"],
            "model": ["burkert", "nfw", "burkert", "nfw", "burkert"],
        })
        second = pd.DataFrame({
            "galaxy_id": ["A", "A", "B", "B", "D", "D"],
            "model": ["burkert", "nfw", "burkert", "nfw", "burkert", "nfw"],
        })
        self.assertEqual(common_galaxy_ids([first, second]), ["A", "B"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
