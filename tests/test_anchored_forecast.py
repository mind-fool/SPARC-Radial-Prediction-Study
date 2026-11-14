import unittest

import numpy as np
import pandas as pd

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
