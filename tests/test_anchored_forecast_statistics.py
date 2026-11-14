import unittest

import pandas as pd

from anchored_forecast_statistics import (
    paired_score_differences,
    region_preference_transitions,
    training_to_forecast_transitions,
)


class TestAnchoredForecastStatistics(unittest.TestCase):
    def results(self):
        return pd.DataFrame({
            "configuration": ["baseline"] * 4,
            "galaxy_id": ["G1", "G1", "G2", "G2"],
            "model": ["burkert", "nfw"] * 2,
            "training_reduced_chi_squared": [1.0, 2.0, 2.0, 1.0],
            "middle_chi_squared_per_point": [2.0, 1.0, 1.0, 2.0],
            "outer_chi_squared_per_point": [3.0, 4.0, 2.0, 1.0],
        })

    def test_paired_differences_use_requested_region(self):
        values = paired_score_differences(self.results(), "baseline", "middle")
        self.assertEqual(values.to_dict(), {"G1": -1.0, "G2": 1.0})

    def test_training_to_forecast_transitions_keep_regions_separate(self):
        transitions, _ = training_to_forecast_transitions(self.results(), "baseline", "outer")
        by_galaxy = transitions.set_index("galaxy_id")
        self.assertEqual(by_galaxy.loc["G1", "training_preference"], "burkert")
        self.assertEqual(by_galaxy.loc["G1", "forecast_preference"], "burkert")
        self.assertEqual(by_galaxy.loc["G2", "forecast_preference"], "nfw")

    def test_middle_to_outer_transitions_use_same_configuration(self):
        comparisons = pd.DataFrame({
            "configuration": ["baseline"] * 4,
            "galaxy_id": ["G1", "G1", "G2", "G2"],
            "region": ["middle", "outer"] * 2,
            "lower_score_model": ["burkert", "nfw", "nfw", "nfw"],
        })
        transitions, counts = region_preference_transitions(comparisons, "baseline")
        by_galaxy = transitions.set_index("galaxy_id")
        self.assertEqual(by_galaxy.loc["G1", "middle_preference"], "burkert")
        self.assertEqual(by_galaxy.loc["G1", "outer_preference"], "nfw")
        self.assertEqual(counts["galaxies"].sum(), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
