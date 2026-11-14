import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from plot_nfw_profile_normalization_grid import (
    normalization_grid_table,
    plot_normalization_grid_stability,
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
    def test_table_sorts_settings_by_disk_normalization(self):
        table = normalization_grid_table(profile_changes(), scale_changes())
        self.assertEqual(table["setting"].tolist(), ["disk_0.40", "disk_0.60"])

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
