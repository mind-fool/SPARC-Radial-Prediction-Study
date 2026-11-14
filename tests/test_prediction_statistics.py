import unittest

import numpy as np
import pandas as pd

from prediction_statistics import (
    bootstrap_median_interval,
    paired_score_differences,
    paired_score_summary,
    rank_transition_summary,
)


class TestPredictionStatistics(unittest.TestCase):
    def results(self):
        return pd.DataFrame({
            "configuration": ["baseline"] * 8,
            "galaxy_id": ["G1", "G1", "G2", "G2", "G3", "G3", "G4", "G4"],
            "model": ["burkert", "nfw"] * 4,
            "training_reduced_chi_squared": [1, 2, 2, 1, 1, 2, 2, 1],
            "held_out_chi_squared_per_point": [2, 1, 1, 2, 3, 4, 4, 3],
        })

    def test_paired_differences_follow_nfw_minus_burkert_convention(self):
        values = paired_score_differences(self.results(), "baseline")
        np.testing.assert_array_equal(values.to_numpy(), [-1.0, 1.0, 1.0, -1.0])

    def test_bootstrap_interval_is_deterministic(self):
        first = bootstrap_median_interval([-2, -1, 1, 2], replicates=1000, seed=3)
        second = bootstrap_median_interval([-2, -1, 1, 2], replicates=1000, seed=3)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 0.0)

    def test_paired_summary_counts_model_preferences(self):
        summary = paired_score_summary([-1.0, 1.0, 1.0, -1.0], replicates=100, seed=1)
        self.assertEqual(summary["nfw_lower_score_galaxies"], 2)
        self.assertEqual(summary["burkert_lower_score_galaxies"], 2)
        self.assertEqual(summary["numerical_ties"], 0)

    def test_transition_summary_distinguishes_training_and_prediction_preferences(self):
        transitions, counts = rank_transition_summary(self.results(), "baseline")
        self.assertEqual(len(transitions), 4)
        table = counts.set_index(["training_preference", "held_out_preference"])["galaxies"]
        self.assertEqual(table.loc[("burkert", "nfw")], 1)
        self.assertEqual(table.loc[("nfw", "burkert")], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
