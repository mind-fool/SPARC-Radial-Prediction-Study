import unittest

import pandas as pd

from plot_nfw_profiles import representative_case_ids


class TestPlotNfwProfiles(unittest.TestCase):
    def test_representative_case_ids_follow_documented_role_order(self):
        representatives = pd.DataFrame({
            "role": [
                "upper_open_lowest_training_chi_squared",
                "closed_profile_typical_scale",
                "lower_open_lowest_training_chi_squared",
            ],
            "galaxy_id": ["UPPER", "CLOSED", "LOWER"],
        })
        self.assertEqual(representative_case_ids(representatives), [
            ("closed_profile_typical_scale", "CLOSED"),
            ("upper_open_lowest_training_chi_squared", "UPPER"),
            ("lower_open_lowest_training_chi_squared", "LOWER"),
        ])


if __name__ == "__main__":
    unittest.main(verbosity=2)
