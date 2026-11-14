import unittest

import pandas as pd

from horizon_statistics import common_galaxy_ids


class TestHorizonStatistics(unittest.TestCase):
    def test_common_galaxy_ids_requires_both_models_in_each_horizon(self):
        first = pd.DataFrame({
            "galaxy_id": ["A", "A", "B", "B", "C"],
            "model": ["burkert", "nfw", "burkert", "nfw", "burkert"],
        })
        second = pd.DataFrame({
            "galaxy_id": ["A", "A", "B", "B", "D", "D"],
            "model": ["burkert", "nfw", "burkert", "nfw", "burkert", "nfw"],
        })
        self.assertEqual(common_galaxy_ids([first, second]), ["A", "B"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
