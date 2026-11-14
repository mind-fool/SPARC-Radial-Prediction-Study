import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import pandas as pd

from plot_rar import plot_rar


class TestPlotRar(unittest.TestCase):
    def test_plot_rar_saves_requested_figure(self):
        data = pd.DataFrame({
            "g_bar": [1e-12, 1e-11, 1e-10],
            "g_obs": [2e-12, 2e-11, 1.5e-10],
        })
        with TemporaryDirectory() as directory:
            output = Path(directory) / "rar.png"
            plot_rar(data, output_path=output, g0=1e-10, show=False)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
