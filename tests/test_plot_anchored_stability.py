import unittest
import json

import pandas as pd
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import main as pipeline_main
from plot_anchored_stability import radius_ceiling_rows


class TestPlotAnchoredStability(unittest.TestCase):
    def test_main_option_fourteen_uses_matching_summary_folders(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            forecast = root / "anchored_forecast" / "inner60_example"
            forecast.mkdir(parents=True)
            (forecast / "manifest.json").write_text('{"status": "complete"}')
            (forecast / "fit_forecasts.csv").write_text("configuration,model\n")
            for folder in (
                root / "anchored_forecast" / "statistics" / "summary_example",
                root / "anchored_forecast" / "configuration_stability" / "stability_example",
            ):
                folder.mkdir(parents=True)
                (folder / "manifest.json").write_text(json.dumps({
                    "status": "complete", "input_run": str(forecast.resolve()),
                }))
            statistics = root / "anchored_forecast" / "statistics" / "summary_example"
            stability = root / "anchored_forecast" / "configuration_stability" / "stability_example"
            with patch.object(pipeline_main, "OUTPUT_DIR", root), \
                 patch.object(pipeline_main, "prepare_input"), \
                 patch.object(pipeline_main, "plot_radius_ceiling_stability") as plot, \
                 patch("builtins.input", side_effect=["14", "1"]), \
                 redirect_stdout(StringIO()):
                pipeline_main.main()
            plot.assert_called_once_with(stability, statistics)

    def test_radius_ceiling_rows_excludes_other_bound_families_and_sorts(self):
        frame = pd.DataFrame({
            "configuration": [
                "radius_ceiling_1000_kpc", "density_floor_1e-5_msun_pc3",
                "radius_ceiling_10_kpc", "radius_ceiling_100_kpc",
            ],
            "family": ["radius_ceiling", "density_floor", "radius_ceiling", "radius_ceiling"],
            "region": ["outer", "outer", "outer", "outer"],
        })
        rows = radius_ceiling_rows(frame)
        self.assertEqual(rows["radius_ceiling_kpc"].tolist(), [10, 100, 1000])


if __name__ == "__main__":
    unittest.main(verbosity=2)
