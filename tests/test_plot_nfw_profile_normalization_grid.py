import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import main as pipeline_main

from plot_nfw_profile_normalization_grid import (
    normalization_grid_table,
    plot_normalization_grid_stability,
    with_baseline_reference,
)


def profile_changes():
    return pd.DataFrame({
        "setting": ["disk_0.60", "disk_0.40"],
        "galaxies": [60, 60],
        "profile_status_changes": [6, 4],
    })


def scale_changes():
    return pd.DataFrame({
        "setting": ["disk_0.60", "disk_0.40"],
        "factor_of_ten_or_more": [12, 9],
        "factor_of_ten_with_unchanged_status": [8, 6],
    })


class TestPlotNormalizationGrid(unittest.TestCase):
    def test_main_option_twenty_two_uses_matching_statistics(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            grid_run = root / "nfw_profile_normalization_grid" / "inner60_example"
            grid_run.mkdir(parents=True)
            (grid_run / "manifest.json").write_text('{"status": "complete"}')
            (grid_run / "profile_identifiability.csv").write_text("galaxy_id\n")
            statistics = root / "nfw_profile_normalization_grid" / "statistics" / "summary_example"
            statistics.mkdir(parents=True)
            (statistics / "manifest.json").write_text(
                '{"status": "complete", "input_run": "' + str(grid_run.resolve()) + '"}',
            )
            with patch.object(pipeline_main, "OUTPUT_DIR", root), \
                 patch.object(pipeline_main, "prepare_input"), \
                 patch.object(pipeline_main, "plot_normalization_grid_stability") as plot, \
                 patch("builtins.input", side_effect=["22", "1"]), \
                 redirect_stdout(StringIO()):
                pipeline_main.main()
            plot.assert_called_once_with(statistics)

    def test_table_sorts_settings_by_disk_normalization(self):
        table = normalization_grid_table(profile_changes(), scale_changes())
        self.assertEqual(table["setting"].tolist(), ["disk_0.40", "disk_0.60"])

    def test_baseline_reference_has_zero_changes(self):
        table = normalization_grid_table(profile_changes(), scale_changes())
        reference = with_baseline_reference(table)
        baseline = reference.loc[reference["disk_mass_to_light"] == 0.50].iloc[0]
        self.assertEqual(baseline["profile_status_changes"], 0)
        self.assertEqual(baseline["factor_of_ten_or_more"], 0)
        self.assertEqual(baseline["factor_of_ten_with_unchanged_status"], 0)

    def test_plot_writes_png_pdf_and_figure_data(self):
        with TemporaryDirectory() as directory:
            statistics = Path(directory) / "statistics"
            statistics.mkdir()
            profile_changes().to_csv(statistics / "profile_change_summary.csv", index=False)
            scale_changes().to_csv(statistics / "scale_radius_change_summary.csv", index=False)
            (statistics / "manifest.json").write_text('{"status": "complete"}\n')
            figures = plot_normalization_grid_stability(statistics)
            self.assertTrue((figures / "normalization_grid_sensitivity.png").is_file())
            self.assertTrue((figures / "normalization_grid_sensitivity.pdf").is_file())
            self.assertTrue((figures / "normalization_grid_figure_data.csv").is_file())


if __name__ == "__main__":
    unittest.main()
