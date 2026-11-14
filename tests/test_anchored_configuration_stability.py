import unittest

import pandas as pd

from anchored_configuration_stability import (
    matched_configuration_changes,
    preference_transition_summary,
    summarize_configuration_changes,
)


class TestAnchoredConfigurationStability(unittest.TestCase):
    def comparisons(self):
        return pd.DataFrame({
            "configuration": ["baseline", "baseline", "wide", "wide"] * 2,
            "family": ["radius_ceiling"] * 8,
            "galaxy_id": ["G1", "G2", "G1", "G2"] * 2,
            "region": ["middle"] * 4 + ["outer"] * 4,
            "lower_score_model": ["burkert", "nfw", "nfw", "nfw", "burkert", "nfw", "burkert", "nfw"],
            "delta_nfw_minus_burkert_chi_squared_per_point": [2.0, -1.0, -2.0, -0.5, 3.0, -4.0, 1.0, -2.0],
        })

    def test_changes_match_baseline_by_galaxy_and_region(self):
        matched = matched_configuration_changes(self.comparisons(), baseline="baseline")
        wide_middle = matched.loc[
            matched.configuration.eq("wide") & matched.region.eq("middle")
        ].set_index("galaxy_id")
        self.assertTrue(wide_middle.loc["G1", "preference_changed"])
        self.assertEqual(wide_middle.loc["G1", "configuration_minus_baseline_delta_score"], -4.0)
        self.assertFalse(wide_middle.loc["G2", "preference_changed"])

    def test_summary_counts_preference_changes(self):
        matched = matched_configuration_changes(self.comparisons(), baseline="baseline")
        summary = summarize_configuration_changes(matched, replicates=20, seed=1)
        wide_middle = summary.loc[
            summary.configuration.eq("wide") & summary.region.eq("middle")
        ].iloc[0]
        self.assertEqual(wide_middle["preference_changes"], 1)
        self.assertEqual(wide_middle["galaxies_with_paired_scores"], 2)

    def test_transition_summary_contains_baseline_to_configuration_labels(self):
        matched = matched_configuration_changes(self.comparisons(), baseline="baseline")
        transitions = preference_transition_summary(matched)
        row = transitions.loc[
            transitions.configuration.eq("wide")
            & transitions.region.eq("middle")
            & transitions.baseline_preference.eq("burkert")
            & transitions.configuration_preference.eq("nfw")
        ].iloc[0]
        self.assertEqual(row["galaxies"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
