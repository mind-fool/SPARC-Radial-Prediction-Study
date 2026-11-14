import unittest
import json

import pandas as pd
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import main as pipeline_main
from plot_nfw_profiles import representative_case_ids


class TestPlotNfwProfiles(unittest.TestCase):
    def test_main_option_seventeen_uses_matching_statistics_folder(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            profile_run = root / "nfw_profiles" / "inner60_example"
            profile_run.mkdir(parents=True)
            (profile_run / "manifest.json").write_text('{"status": "complete"}')
            (profile_run / "profile_grid.csv").write_text("galaxy_id\n")
            statistics = root / "nfw_profiles" / "statistics" / "summary_example"
            statistics.mkdir(parents=True)
            (statistics / "manifest.json").write_text(json.dumps({
                "status": "complete", "input_run": str(profile_run.resolve()),
            }))
            with patch.object(pipeline_main, "OUTPUT_DIR", root), \
                 patch.object(pipeline_main, "prepare_input"), \
                 patch.object(pipeline_main, "plot_nfw_profile_figures") as plot, \
                 patch("builtins.input", side_effect=["17", "1"]), \
                 redirect_stdout(StringIO()):
                pipeline_main.main()
            plot.assert_called_once_with(profile_run, statistics)

    def test_representative_case_ids_follow_documented_role_order(self):
        representatives = pd.DataFrame({
            "role": [
                "upper_open_lowest_training_chi_squared",
                "closed_profile_typical_scale",
                "lower_open_lowest_training_chi_squared",
            ],
            "galaxy_id": ["UPPER", "CLOSED", "LOWER"],
        })
        self.assertEqual(representative_case_ids(representatives), [
            ("closed_profile_typical_scale", "CLOSED"),
            ("upper_open_lowest_training_chi_squared", "UPPER"),
            ("lower_open_lowest_training_chi_squared", "LOWER"),
        ])


if __name__ == "__main__":
    unittest.main(verbosity=2)
