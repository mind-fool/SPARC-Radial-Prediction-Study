import unittest

import numpy as np
import pandas as pd

from run_nfw_profile_grid import profile_summary


class TestRunNfwProfileGrid(unittest.TestCase):
    def test_profile_summary_marks_an_interval_touching_upper_grid(self):
        rows = [
            {"galaxy_id": "G1", "fixed_scale_radius_kpc": 0.1, "training_chi_squared": 10.0, "fit_error": ""},
            {"galaxy_id": "G1", "fixed_scale_radius_kpc": 1.0, "training_chi_squared": 2.0, "fit_error": ""},
            {"galaxy_id": "G1", "fixed_scale_radius_kpc": 10.0, "training_chi_squared": 2.2, "fit_error": ""},
        ]
        profiled, summary = profile_summary(rows)
        np.testing.assert_allclose(
            profiled["delta_training_chi_squared"].to_numpy(), [8.0, 0.0, 0.2],
        )
        self.assertTrue(summary.iloc[0]["nominal_95_touches_upper_grid"])
        self.assertFalse(summary.iloc[0]["nominal_95_touches_lower_grid"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
