"""Check NFW profile identifiability under pre-specified stellar assumptions.

This study holds the shared SPARC observations, eligible galaxies, anchored radial
split, NFW scale-radius grid, density bounds, and nominal profile thresholds fixed.
Only the stellar mass-to-light setting changes.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy

from anchored_radial_split import eligible_anchored_galaxy_ids, split_inner_middle_outer
from data_preprocess import load_sparc_dataset
from nfw_profile_fit import DEFAULT_DENSITY_BOUNDS, fit_nfw_fixed_scale
from nfw_profile_statistics import profile_status
from radial_holdout import score_held_out_prediction
from run_mass_to_light_sweep import STELLAR_SETTINGS
from run_nfw_profile_grid import PROFILE_RADIUS_GRID_KPC, profile_summary
from run_radial_prediction import ensure_input_data


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PURPOSE = "NFW profile-identifiability robustness to pre-specified stellar mass-to-light assumptions."


def shared_observation_index(datasets):
    """Return the exact ID--radius observations valid under every stellar setting."""
    if not datasets:
        raise ValueError("At least one stellar-setting dataset is required.")
    shared = None
    for dataset in datasets.values():
        required = {"ID", "R_kpc"}
        missing = required.difference(dataset.columns)
        if missing:
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
        index = dataset.set_index(["ID", "R_kpc"], verify_integrity=True).index
        shared = index if shared is None else shared.intersection(index, sort=False)
    if shared is None or shared.empty:
        raise ValueError("No observations are shared across the stellar settings.")
    return shared


def validate_stellar_settings(stellar_settings, baseline_setting):
    """Validate named positive stellar settings and return an immutable sequence."""
    settings = tuple(stellar_settings)
    if not settings:
        raise ValueError("At least one stellar setting is required.")
    names = []
    for setting, disk_ratio, bulge_ratio in settings:
        if not isinstance(setting, str) or not setting:
            raise ValueError("Each stellar setting needs a nonempty string name.")
        ratios = np.asarray([disk_ratio, bulge_ratio], dtype=float)
        if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0):
            raise ValueError("Stellar mass-to-light ratios must be finite and positive.")
        names.append(setting)
    if len(set(names)) != len(names):
        raise ValueError("Stellar setting names must be unique.")
    if baseline_setting not in names:
        raise ValueError(f"Baseline setting {baseline_setting!r} is not present.")
    return settings


def _git_metadata():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_DIR, text=True,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", None
    return commit, dirty


def _profile_rows(setting, disk_ratio, bulge_ratio, dataset, galaxy_ids, grid,
                  density_bounds):
    """Fit one fixed-radius NFW profile grid for every selected galaxy."""
    rows = []
    for number, galaxy_id in enumerate(galaxy_ids, 1):
        galaxy = dataset.loc[dataset["ID"].eq(galaxy_id)]
        training, middle, outer = split_inner_middle_outer(galaxy)
        for fixed_radius in grid:
            row = {
                "setting": setting,
                "disk_mass_to_light": disk_ratio,
                "bulge_mass_to_light": bulge_ratio,
                "galaxy_id": galaxy_id,
                "fixed_scale_radius_kpc": fixed_radius,
                "n_training": len(training),
                "n_middle": len(middle),
                "n_outer": len(outer),
                "training_chi_squared": np.nan,
                "training_reduced_chi_squared": np.nan,
                "density_msun_pc3": np.nan,
                "density_bound_hit": "",
                "physical_boundary_hit": "",
                "fit_error": "",
                "middle_chi_squared_per_point": np.nan,
                "outer_chi_squared_per_point": np.nan,
                "middle_rmse_kms": np.nan,
                "outer_rmse_kms": np.nan,
                "middle_prediction_error": "",
                "outer_prediction_error": "",
            }
            try:
                fit = fit_nfw_fixed_scale(
                    galaxy_id, training, fixed_radius, density_bounds=density_bounds,
                )
                row.update(
                    training_chi_squared=fit["chi_squared"],
                    training_reduced_chi_squared=fit["reduced_chi_squared"],
                    density_msun_pc3=fit["density_msun_pc3"],
                    density_bound_hit=fit["density_bound_hit"],
                    physical_boundary_hit=fit["physical_boundary_hit"],
                )
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                row["fit_error"] = str(error) or type(error).__name__
                rows.append(row)
                continue
            for region, held_out in (("middle", middle), ("outer", outer)):
                try:
                    score = score_held_out_prediction(fit, held_out, model="nfw")
                    row[f"{region}_chi_squared_per_point"] = score[
                        "held_out_chi_squared_per_point"
                    ]
                    row[f"{region}_rmse_kms"] = score["held_out_rmse_kms"]
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    row[f"{region}_prediction_error"] = str(error) or type(error).__name__
            rows.append(row)
        if number % 10 == 0 or number == len(galaxy_ids):
            print(f"  {setting}: {number}/{len(galaxy_ids)} galaxies profiled", flush=True)
    return rows


def run_nfw_profile_mass_to_light(
    output_dir=None, *, data_path=None, galaxy_ids=None,
    scale_radius_grid_kpc=PROFILE_RADIUS_GRID_KPC,
    density_bounds=DEFAULT_DENSITY_BOUNDS,
    stellar_settings=STELLAR_SETTINGS,
    baseline_setting="baseline",
    purpose=DEFAULT_PURPOSE,
    source_files=(),
    run_label="mass-to-light NFW profile",
):
    """Run controlled NFW profile grids across the pre-specified stellar settings."""
    settings = validate_stellar_settings(stellar_settings, baseline_setting)
    grid = np.asarray(scale_radius_grid_kpc, dtype=float)
    if grid.ndim != 1 or grid.size < 3 or not np.all(np.isfinite(grid)) or np.any(grid <= 0):
        raise ValueError("scale_radius_grid_kpc needs at least three finite positive radii.")
    if not np.all(np.diff(grid) > 0):
        raise ValueError("scale_radius_grid_kpc must be strictly increasing.")
    data_path = ensure_input_data(
        Path(data_path) if data_path else PROJECT_DIR / "GALAXY_CSVDATA.csv",
    )
    datasets = {
        setting: load_sparc_dataset(
            data_path, disk_mass_to_light=disk_ratio,
            bulge_mass_to_light=bulge_ratio,
        )
        for setting, disk_ratio, bulge_ratio in settings
    }
    shared_points = shared_observation_index(datasets)
    datasets = {
        setting: dataset.set_index(["ID", "R_kpc"], verify_integrity=True)
        .loc[shared_points].reset_index()
        for setting, dataset in datasets.items()
    }
    eligible_by_setting = {
        setting: eligible_anchored_galaxy_ids(dataset)
        for setting, dataset in datasets.items()
    }
    baseline_eligible = eligible_by_setting[baseline_setting]
    missing = {
        setting: sorted(set(baseline_eligible).difference(ids))
        for setting, ids in eligible_by_setting.items()
    }
    missing = {setting: ids for setting, ids in missing.items() if ids}
    if missing:
        raise RuntimeError(
            "The anchored eligibility sample changed across stellar settings: "
            f"{missing}",
        )
    selected = baseline_eligible
    if galaxy_ids is not None:
        available = {name.casefold(): name for name in selected}
        selected = []
        for requested_id in galaxy_ids:
            canonical = available.get(requested_id.strip().casefold())
            if canonical is None:
                raise ValueError(
                    f"Galaxy {requested_id!r} is not eligible for the controlled profile study.",
                )
            if canonical not in selected:
                selected.append(canonical)
    if not selected:
        raise ValueError("No galaxies satisfy the anchored profile criteria.")

    root = Path(output_dir) if output_dir else PROJECT_DIR / "outputs" / "nfw_profile_mass_to_light"
    run_dir = root / datetime.now(timezone.utc).strftime("inner60_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    commit, dirty = _git_metadata()
    manifest = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": purpose,
        "controlled_quantities": (
            "Shared ID-radius observations, galaxy sample, anchored radial split, scale-radius "
            "grid, density bounds, and nominal Delta chi-squared thresholds are fixed. "
            "Only stellar mass-to-light assumptions change."
        ),
        "interpretation": (
            "Nominal Delta chi-squared profile intervals are diagnostics, not physical confidence "
            "intervals under correlated errors or unmodeled astrophysical systematics. Held-out "
            "points are not used to select the NFW scale radius."
        ),
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "input_sha256": sha256(data_path.read_bytes()).hexdigest(),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "stellar_settings": [
            {"setting": setting, "disk_mass_to_light": disk, "bulge_mass_to_light": bulge}
            for setting, disk, bulge in settings
        ],
        "shared_observation_count": len(shared_points),
        "training_fraction": 0.60,
        "middle_fraction": 0.20,
        "outer_fraction": 0.20,
        "density_bounds_msun_pc3": list(density_bounds),
        "scale_radius_grid_kpc": grid.tolist(),
        "galaxy_count": len(selected),
        "galaxy_ids": selected,
        "source_sha256": {
            name: sha256((PROJECT_DIR / name).read_bytes()).hexdigest()
            for name in (
                "run_nfw_profile_mass_to_light.py", "run_nfw_profile_grid.py",
                "nfw_profile_fit.py", "nfw_profile_statistics.py", "data_preprocess.py",
                "anchored_radial_split.py", "radial_holdout.py", "run_mass_to_light_sweep.py",
                *source_files,
            ) if (PROJECT_DIR / name).is_file()
        },
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Profiling {len(selected)} shared galaxies at {len(grid)} NFW radii under "
        f"{len(settings)} stellar settings.", flush=True,
    )
    all_rows, all_summaries, all_status = [], [], []
    for setting, disk_ratio, bulge_ratio in settings:
        print(f"Running {setting} (disk={disk_ratio:g}, bulge={bulge_ratio:g})...", flush=True)
        rows = _profile_rows(
            setting, disk_ratio, bulge_ratio, datasets[setting], selected, grid, density_bounds,
        )
        profiled_rows, summary = profile_summary(rows)
        if summary.empty:
            raise RuntimeError(f"No successful NFW profile fits for setting {setting!r}.")
        summary = summary.assign(
            setting=setting, disk_mass_to_light=disk_ratio, bulge_mass_to_light=bulge_ratio,
        )
        all_rows.append(profiled_rows)
        all_summaries.append(summary)
        all_status.append(profile_status(summary))

    profile_grid = pd.concat(all_rows, ignore_index=True)
    profile_summary_table = pd.concat(all_summaries, ignore_index=True)
    identifiability = pd.concat(all_status, ignore_index=True)
    profile_grid.to_csv(run_dir / "profile_grid.csv", index=False)
    profile_summary_table.to_csv(run_dir / "profile_summary.csv", index=False)
    identifiability.to_csv(run_dir / "profile_identifiability.csv", index=False)
    manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {run_label} results under {run_dir}")
    return run_dir


if __name__ == "__main__":
    run_nfw_profile_mass_to_light()
