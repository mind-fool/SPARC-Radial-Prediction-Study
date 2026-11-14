"""Summarize NFW profile stability across the tied stellar-normalization grid."""

from nfw_profile_mass_to_light_statistics import analyze_nfw_profile_mass_to_light
from run_nfw_profile_normalization_grid import BASELINE_SETTING


def analyze_nfw_profile_normalization_grid(input_run_dir, output_dir=None):
    """Summarize every grid setting against the pre-specified disk-0.50 baseline."""
    return analyze_nfw_profile_mass_to_light(
        input_run_dir, output_dir=output_dir, baseline=BASELINE_SETTING,
    )
