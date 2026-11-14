import unittest

import pandas as pd

from nfw_profile_statistics import (
    profile_status,
    representative_profiles,
    training_selected_profiles,
)


class TestNfwProfileStatistics(unittest.TestCase):
    def summary(self):
        return pd.DataFrame({
            "galaxy_id": ["CLOSED", "UPPER", "LOWER", "BOTH"],
            "nominal_95_touches_lower_grid": [False, False, True, True],
            "nominal_95_touches_upper_grid": [False, True, False, True],
            "nominal_95_lower_scale_radius_kpc": [1.0, 1.0, 0.01, 0.01],
            "nominal_95_upper_scale_radius_kpc": [10.0, 10000.0, 2.0, 10000.0],
        })

    def test_profile_status_labels_each_edge_case(self):
        statuses = profile_status(self.summary()).set_index("galaxy_id")["profile_status"]
        self.assertEqual(statuses.to_dict(), {
            "CLOSED": "closed", "UPPER": "open_upper",
            "LOWER": "open_lower", "BOTH": "open_both",
        })

    def test_training_selection_uses_lowest_training_chi_squared_only(self):
        grid = pd.DataFrame({
            "galaxy_id": ["CLOSED", "CLOSED", "UPPER", "UPPER", "LOWER", "BOTH"],
            "fixed_scale_radius_kpc": [1.0, 10.0, 1.0, 10000.0, 0.01, 10000.0],
            "training_chi_squared": [5.0, 1.0, 4.0, 3.0, 2.0, 6.0],
            "fit_error": [""] * 6,
        })
        selected = training_selected_profiles(grid, self.summary()).set_index("galaxy_id")
        self.assertEqual(selected.loc["CLOSED", "fixed_scale_radius_kpc"], 10.0)
        self.assertEqual(selected.loc["UPPER", "fixed_scale_radius_kpc"], 10000.0)

    def test_representatives_are_selected_from_status_groups(self):
        selected = pd.DataFrame({
            "galaxy_id": ["C1", "C2", "U1", "U2"],
            "profile_status": ["closed", "closed", "open_upper", "open_upper"],
            "fixed_scale_radius_kpc": [1.0, 10.0, 100.0, 1000.0],
            "training_chi_squared": [2.0, 3.0, 5.0, 1.0],
        })
        representatives = representative_profiles(selected).set_index("role")
        self.assertEqual(representatives.loc["upper_open_lowest_training_chi_squared", "galaxy_id"], "U2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
