"""Profile NFW scale-radius identifiability across one stellar normalization grid."""

from pathlib import Path

from run_nfw_profile_mass_to_light import run_nfw_profile_mass_to_light


DISK_MASS_TO_LIGHT_GRID = tuple(round(0.30 + 0.05 * index, 2) for index in range(9))
BULGE_TO_DISK_MASS_TO_LIGHT_RATIO = 1.4
BASELINE_SETTING = "disk_0.50"
NORMALIZATION_GRID_SETTINGS = tuple(
    (
        f"disk_{disk_ratio:.2f}",
        disk_ratio,
        round(BULGE_TO_DISK_MASS_TO_LIGHT_RATIO * disk_ratio, 10),
    )
    for disk_ratio in DISK_MASS_TO_LIGHT_GRID
)
PURPOSE = (
    "One-parameter NFW profile-identifiability sensitivity to stellar mass-to-light "
    "normalization. Disk mass-to-light spans 0.30 to 0.70 in pre-specified 0.05 "
    "steps; bulge mass-to-light remains 1.4 times the disk value."
)


def run_nfw_profile_normalization_grid(output_dir=None, **kwargs):
    """Run the pre-specified disk normalization grid with bulge/disk ratio fixed."""
    project_dir = Path(__file__).resolve().parent
    return run_nfw_profile_mass_to_light(
        output_dir=(
            output_dir
            if output_dir is not None
            else project_dir / "outputs" / "nfw_profile_normalization_grid"
        ),
        stellar_settings=NORMALIZATION_GRID_SETTINGS,
        baseline_setting=BASELINE_SETTING,
        purpose=PURPOSE,
        source_files=("run_nfw_profile_normalization_grid.py",),
        run_label="normalization-grid NFW profile",
        **kwargs,
    )


if __name__ == "__main__":
    run_nfw_profile_normalization_grid()
