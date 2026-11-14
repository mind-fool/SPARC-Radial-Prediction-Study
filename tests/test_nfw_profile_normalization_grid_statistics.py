import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import main as pipeline_main
from nfw_profile_normalization_grid_statistics import (
    analyze_nfw_profile_normalization_grid,
)
from run_nfw_profile_normalization_grid import BASELINE_SETTING


class TestNormalizationGridStatistics(unittest.TestCase):
    def test_main_option_twenty_one_uses_selected_completed_grid_run(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "nfw_profile_normalization_grid" / "inner60_example"
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text('{"status": "complete"}')
            (run_dir / "profile_identifiability.csv").write_text("galaxy_id\n")
            with patch.object(pipeline_main, "OUTPUT_DIR", root), \
                 patch.object(pipeline_main, "prepare_input"), \
                 patch.object(pipeline_main, "analyze_nfw_profile_normalization_grid") as analyze, \
                 patch("builtins.input", side_effect=["21", "1"]), \
                 redirect_stdout(StringIO()):
                pipeline_main.main()
            analyze.assert_called_once_with(run_dir)

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
