import unittest

import numpy as np
import pandas as pd

from halo_models import nfw_halo_velocity
from radial_holdout import (
    eligible_galaxy_ids,
    score_held_out_prediction,
    split_inner_outer,
)


class TestRadialHoldout(unittest.TestCase):
    def galaxy(self, count=12, galaxy_id="synthetic"):
        radius = np.arange(1, count + 1, dtype=float)
        baryonic_squared = 100.0 * radius / (radius + 2.0)
        observed = np.sqrt(baryonic_squared + nfw_halo_velocity(radius, 0.03, 4.0) ** 2)
        return pd.DataFrame({
            "ID": galaxy_id, "R_kpc": radius, "Vobs_kms": observed,
            "eV_kms": 2.0, "Vbar_squared_kms2": baryonic_squared,
        })

    def test_primary_split_uses_eight_inner_and_four_outer_points(self):
        training, held_out = split_inner_outer(self.galaxy())
        self.assertEqual(len(training), 8)
        self.assertEqual(len(held_out), 4)
        self.assertLess(training["R_kpc"].max(), held_out["R_kpc"].min())
        np.testing.assert_array_equal(training["R_kpc"], np.arange(1, 9, dtype=float))
        np.testing.assert_array_equal(held_out["R_kpc"], np.arange(9, 13, dtype=float))

    def test_insufficient_data_are_not_eligible(self):
        short = self.galaxy(11, "short")
        valid = self.galaxy(12, "valid")
        with self.assertRaisesRegex(ValueError, "Need at least 8 inner"):
            split_inner_outer(short)
        self.assertEqual(eligible_galaxy_ids(pd.concat([short, valid])), ["valid"])

    def test_outer_score_uses_fixed_inner_fit_without_refitting(self):
        _, held_out = split_inner_outer(self.galaxy())
        score = score_held_out_prediction(
            {"rho_s_msun_pc3": 0.03, "r_s_kpc": 4.0}, held_out, model="nfw",
        )
        self.assertEqual(score["n_held_out"], 4)
        self.assertAlmostEqual(score["held_out_chi_squared"], 0.0, places=10)
        self.assertAlmostEqual(score["held_out_chi_squared_per_point"], 0.0, places=10)
        self.assertAlmostEqual(score["held_out_rmse_kms"], 0.0, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
