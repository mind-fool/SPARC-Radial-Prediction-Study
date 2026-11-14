import unittest

import pandas as pd

from run_nfw_profile_mass_to_light import shared_observation_index


class TestProfileMassToLightStudy(unittest.TestCase):
    def test_shared_observation_index_keeps_only_common_id_radius_rows(self):
        datasets = {
            "baseline": pd.DataFrame({"ID": ["G1", "G1", "G2"], "R_kpc": [1.0, 2.0, 1.0]}),
            "higher_disk": pd.DataFrame({"ID": ["G1", "G2", "G3"], "R_kpc": [1.0, 1.0, 1.0]}),
            "lower_bulge": pd.DataFrame({"ID": ["G1", "G2"], "R_kpc": [1.0, 1.0]}),
        }
        shared = shared_observation_index(datasets)
        self.assertEqual(list(shared), [("G1", 1.0), ("G2", 1.0)])


if __name__ == "__main__":
    unittest.main()
