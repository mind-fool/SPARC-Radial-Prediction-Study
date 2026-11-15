import unittest

import numpy as np
import pandas as pd

from burkert_halo_fit import total_velocity_from_squared
from halo_models import nfw_halo_velocity
from nfw_profile_fit import fit_nfw_fixed_scale


class TestNfwProfileFit(unittest.TestCase):
    def synthetic_data(self, baryonic_squared):
        radius = np.linspace(0.5, 8.0, len(baryonic_squared))
        rho_s, r_s = 0.03, 4.0
        observed = total_velocity_from_squared(
            baryonic_squared, nfw_halo_velocity(radius, rho_s, r_s) ** 2,
        )
        return pd.DataFrame({
            "ID": "SYNTHETIC", "R_kpc": radius, "Vobs_kms": observed,
            "eV_kms": np.ones_like(radius), "Vbar_squared_kms2": baryonic_squared,
        })

    def test_recovers_density_at_known_fixed_scale_radius(self):
        data = self.synthetic_data(np.full(12, 400.0))
        result = fit_nfw_fixed_scale("SYNTHETIC", data, 4.0)
        self.assertAlmostEqual(result["density_msun_pc3"], 0.03, places=5)
        self.assertLess(result["chi_squared"], 1e-8)

    def test_handles_negative_baryonic_contribution_with_positive_total(self):
        baryonic_squared = np.full(12, 300.0)
        baryonic_squared[0] = -10.0
        data = self.synthetic_data(baryonic_squared)
        result = fit_nfw_fixed_scale("SYNTHETIC", data, 4.0)
        self.assertTrue(np.all(result["model_kms"] >= 0))
        self.assertAlmostEqual(result["density_msun_pc3"], 0.03, places=5)

    def test_recovers_density_just_above_lower_bound(self):
        radius = np.linspace(0.5, 8.0, 12)
        true_density = 1.1e-5
        scale_radius = 10_000.0
        baryonic_squared = np.full(radius.size, 400.0)
        observed = total_velocity_from_squared(
            baryonic_squared,
            nfw_halo_velocity(radius, true_density, scale_radius) ** 2,
        )
        data = pd.DataFrame({
            "ID": "SYNTHETIC", "R_kpc": radius, "Vobs_kms": observed,
            "eV_kms": np.ones_like(radius),
            "Vbar_squared_kms2": baryonic_squared,
        })
        result = fit_nfw_fixed_scale("SYNTHETIC", data, scale_radius)
        self.assertAlmostEqual(result["density_msun_pc3"], true_density, places=10)
        self.assertLess(result["chi_squared"], 1e-10)

    def test_recovers_density_just_below_upper_bound(self):
        radius = np.linspace(0.5, 8.0, 12)
        true_density = 9.5
        scale_radius = 0.1
        baryonic_squared = np.full(radius.size, 400.0)
        observed = total_velocity_from_squared(
            baryonic_squared,
            nfw_halo_velocity(radius, true_density, scale_radius) ** 2,
        )
        data = pd.DataFrame({
            "ID": "SYNTHETIC", "R_kpc": radius, "Vobs_kms": observed,
            "eV_kms": np.ones_like(radius),
            "Vbar_squared_kms2": baryonic_squared,
        })
        result = fit_nfw_fixed_scale("SYNTHETIC", data, scale_radius)
        self.assertAlmostEqual(result["density_msun_pc3"], true_density, places=6)
        self.assertLess(result["chi_squared"], 1e-10)

    def test_rejects_nonpositive_fixed_scale_radius(self):
        data = self.synthetic_data(np.full(12, 400.0))
        with self.assertRaisesRegex(ValueError, "positive"):
            fit_nfw_fixed_scale("SYNTHETIC", data, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
