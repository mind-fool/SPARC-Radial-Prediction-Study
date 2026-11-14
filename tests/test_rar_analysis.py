import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from rar_analysis import filter_data, generate_report, rar_model


class TestRarAnalysis(unittest.TestCase):
    def test_filter_data_applies_relative_velocity_error_cut(self):
        data = pd.DataFrame({
            "ID": ["G1", "G2"],
            "Vobs_kms": [100.0, 50.0],
            "eV_kms": [10.0, 15.0],
        })
        with TemporaryDirectory() as directory:
            output = Path(directory) / "filtered.csv"
            filtered = filter_data(df=data, output_path=output)
            self.assertEqual(filtered["ID"].tolist(), ["G1"])
            self.assertTrue(output.is_file())

    def test_rar_model_returns_finite_positive_prediction(self):
        prediction = rar_model(np.array([1e-12, 1e-10]), 1e-10)
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertTrue(np.all(prediction > 0))

    def test_generate_report_uses_given_g0_without_refitting(self):
        data = pd.DataFrame({"ID": ["G1", "G1", "G2"]})
        stream = StringIO()
        with redirect_stdout(stream):
            generate_report(data, g0=1.23e-10)
        report = stream.getvalue()
        self.assertIn("Number of galaxies: 2", report)
        self.assertIn("Number of total points: 3", report)
        self.assertIn("1.23e-10", report)


if __name__ == "__main__":
    unittest.main()
