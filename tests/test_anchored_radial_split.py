import unittest

import numpy as np
import pandas as pd

from anchored_radial_split import (
    eligible_anchored_galaxy_ids,
    split_inner_middle_outer,
)


class TestAnchoredRadialSplit(unittest.TestCase):
    def galaxy(self, count, galaxy_id="synthetic"):
        radius = np.arange(1, count + 1, dtype=float)
        return pd.DataFrame({
            "ID": galaxy_id, "R_kpc": radius, "Vobs_kms": radius,
            "eV_kms": 2.0, "Vbar_squared_kms2": radius**2,
        })

    def test_twenty_points_split_into_twelve_four_four(self):
        training, middle, outer = split_inner_middle_outer(self.galaxy(20))
        self.assertEqual((len(training), len(middle), len(outer)), (12, 4, 4))
        self.assertLess(training["R_kpc"].max(), middle["R_kpc"].min())
        self.assertLess(middle["R_kpc"].max(), outer["R_kpc"].min())

    def test_nineteen_points_are_not_eligible(self):
        short = self.galaxy(19, "short")
        valid = self.galaxy(20, "valid")
        with self.assertRaisesRegex(ValueError, "Need at least 12 inner"):
            split_inner_middle_outer(short)
        self.assertEqual(eligible_anchored_galaxy_ids(pd.concat([short, valid])), ["valid"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
