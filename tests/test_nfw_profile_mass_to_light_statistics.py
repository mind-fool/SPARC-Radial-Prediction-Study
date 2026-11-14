import unittest

import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

from nfw_profile_mass_to_light_statistics import (
    analyze_nfw_profile_mass_to_light,
    matched_profile_changes,
    profile_change_summary,
    profile_status_transitions,
    status_count_summary,
)


def identifiability_table():
    rows = []
    for setting, statuses in {
        "baseline": {"G1": "closed", "G2": "open_upper"},
        "higher_disk": {"G1": "open_lower", "G2": "open_upper"},
        "lower_bulge": {"G1": "closed", "G2": "closed"},
    }.items():
        for number, (galaxy_id, status) in enumerate(statuses.items(), 1):
            rows.append({
                "setting": setting,
                "galaxy_id": galaxy_id,
                "profile_status": status,
                "minimum_training_chi_squared": float(number),
                "minimum_scale_radius_kpc": float(number),
                "nominal_95_lower_scale_radius_kpc": 0.01,
                "nominal_95_upper_scale_radius_kpc": 100.0,
            })
    return pd.DataFrame(rows)


class TestProfileMassToLightStatistics(unittest.TestCase):
    def test_analyzer_writes_matched_change_outputs(self):
        with TemporaryDirectory() as directory:
            input_dir = Path(directory) / "inner60_example"
            input_dir.mkdir()
            identifiability_table().to_csv(
                input_dir / "profile_identifiability.csv", index=False,
            )
            (input_dir / "manifest.json").write_text('{"status": "complete"}\n')
            output_dir = analyze_nfw_profile_mass_to_light(input_dir)
            self.assertTrue((output_dir / "galaxy_profile_changes.csv").is_file())
            self.assertTrue((output_dir / "profile_change_summary.csv").is_file())

    def test_status_count_summary_keeps_settings_separate(self):
        counts = status_count_summary(identifiability_table())
        higher = counts.loc[counts["setting"].eq("higher_disk")]
        self.assertEqual(higher.set_index("profile_status")["galaxies"].to_dict(), {
            "open_lower": 1, "open_upper": 1,
        })

    def test_matched_changes_compare_each_setting_to_baseline(self):
        matched = matched_profile_changes(identifiability_table())
        self.assertEqual(len(matched), 4)
        changed = matched.loc[matched["profile_status_changed"], ["setting", "galaxy_id"]]
        self.assertEqual(set(map(tuple, changed.to_numpy())), {
            ("higher_disk", "G1"), ("lower_bulge", "G2"),
        })

    def test_change_and_transition_summaries_have_expected_counts(self):
        matched = matched_profile_changes(identifiability_table())
        summary = profile_change_summary(matched).set_index("setting")
        self.assertEqual(summary.loc["higher_disk", "profile_status_changes"], 1)
        self.assertEqual(summary.loc["lower_bulge", "profile_status_unchanged"], 1)
        transitions = profile_status_transitions(matched)
        self.assertEqual(transitions["galaxies"].sum(), 4)


if __name__ == "__main__":
    unittest.main()
