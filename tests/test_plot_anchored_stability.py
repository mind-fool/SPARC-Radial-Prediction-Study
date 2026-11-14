import unittest

import pandas as pd

from plot_anchored_stability import radius_ceiling_rows


class TestPlotAnchoredStability(unittest.TestCase):
    def test_radius_ceiling_rows_excludes_other_bound_families_and_sorts(self):
        frame = pd.DataFrame({
            "configuration": [
                "radius_ceiling_1000_kpc", "density_floor_1e-5_msun_pc3",
                "radius_ceiling_10_kpc", "radius_ceiling_100_kpc",
            ],
            "family": ["radius_ceiling", "density_floor", "radius_ceiling", "radius_ceiling"],
            "region": ["outer", "outer", "outer", "outer"],
        })
        rows = radius_ceiling_rows(frame)
        self.assertEqual(rows["radius_ceiling_kpc"].tolist(), [10, 100, 1000])


if __name__ == "__main__":
    unittest.main(verbosity=2)
