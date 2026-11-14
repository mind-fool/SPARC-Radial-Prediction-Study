"""Run fixed-scale NFW likelihood profiles on the anchored radial split."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy

from anchored_radial_split import (
    MIN_MIDDLE_POINTS,
    MIN_OUTER_POINTS,
    MIN_TRAINING_POINTS,
    eligible_anchored_galaxy_ids,
    split_inner_middle_outer,
)
from data_preprocess import load_sparc_dataset
from nfw_profile_fit import DEFAULT_DENSITY_BOUNDS, fit_nfw_fixed_scale
from radial_holdout import score_held_out_prediction
from run_radial_prediction import ensure_input_data


PROJECT_DIR = Path(__file__).resolve().parent
PROFILE_RADIUS_GRID_KPC = tuple(np.geomspace(0.01, 10_000.0, num=41))
DELTA_CHI_SQUARED_68 = 1.0
DELTA_CHI_SQUARED_95 = 3.84


def profile_summary(rows):
    """Summarize nominal one-parameter profile intervals by galaxy."""
    results = pd.DataFrame(rows).copy()
    successful = results.loc[results["fit_error"].eq("")].copy()
    if successful.empty:
        return results, pd.DataFrame()
    minimum = successful.groupby("galaxy_id")["training_chi_squared"].transform("min")
    successful["delta_training_chi_squared"] = successful["training_chi_squared"] - minimum
    results = results.merge(
        successful[["galaxy_id", "fixed_scale_radius_kpc", "delta_training_chi_squared"]],
        on=["galaxy_id", "fixed_scale_radius_kpc"], how="left",
    )
    summaries = []
    for galaxy_id, group in successful.groupby("galaxy_id", sort=True):
        group = group.sort_values("fixed_scale_radius_kpc")
        best = group.loc[group["training_chi_squared"].idxmin()]
        row = {
            "galaxy_id": galaxy_id,
            "profile_points": len(group),
            "minimum_training_chi_squared": best["training_chi_squared"],
            "minimum_scale_radius_kpc": best["fixed_scale_radius_kpc"],
        }
        for level, label in ((DELTA_CHI_SQUARED_68, "68"), (DELTA_CHI_SQUARED_95, "95")):
            allowed = group.loc[group["delta_training_chi_squared"].le(level)]
            row[f"nominal_{label}_lower_scale_radius_kpc"] = allowed["fixed_scale_radius_kpc"].min()
            row[f"nominal_{label}_upper_scale_radius_kpc"] = allowed["fixed_scale_radius_kpc"].max()
            row[f"nominal_{label}_touches_lower_grid"] = bool(
                np.isclose(allowed["fixed_scale_radius_kpc"].min(), group["fixed_scale_radius_kpc"].min())
            )
            row[f"nominal_{label}_touches_upper_grid"] = bool(
                np.isclose(allowed["fixed_scale_radius_kpc"].max(), group["fixed_scale_radius_kpc"].max())
            )
        summaries.append(row)
    return results, pd.DataFrame(summaries)


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


def run_nfw_profile_grid(output_dir=None, *, data_path=None, galaxy_ids=None,
                         scale_radius_grid_kpc=PROFILE_RADIUS_GRID_KPC,
                         density_bounds=DEFAULT_DENSITY_BOUNDS):
    """Fit a pre-specified NFW radius grid using only anchored inner data."""
    grid = np.asarray(scale_radius_grid_kpc, dtype=float)
    if grid.ndim != 1 or grid.size < 3 or not np.all(np.isfinite(grid)) or np.any(grid <= 0):
        raise ValueError("scale_radius_grid_kpc needs at least three finite positive radii.")
    if not np.all(np.diff(grid) > 0):
        raise ValueError("scale_radius_grid_kpc must be strictly increasing.")
    data_path = ensure_input_data(
        Path(data_path) if data_path else PROJECT_DIR / "GALAXY_CSVDATA.csv",
    )
    dataset = load_sparc_dataset(data_path)
    eligible = eligible_anchored_galaxy_ids(dataset)
    if galaxy_ids is not None:
        available = {name.casefold(): name for name in eligible}
        selected = []
        for requested_id in galaxy_ids:
            canonical = available.get(requested_id.strip().casefold())
            if canonical is None:
                raise ValueError(f"Galaxy {requested_id!r} is not eligible for the NFW profile study.")
            if canonical not in selected:
                selected.append(canonical)
        eligible = selected
    if not eligible:
        raise ValueError("No galaxies satisfy the anchored profile criteria.")

    root = Path(output_dir) if output_dir else PROJECT_DIR / "outputs" / "nfw_profiles"
    run_dir = root / datetime.now(timezone.utc).strftime("inner60_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    commit, dirty = _git_metadata()
    manifest = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Fixed-scale NFW training profiles and anchored middle/outer forecasts.",
        "interpretation": (
            "The nominal Delta chi-squared intervals assume one profiled parameter and "
            "are diagnostics, not physical confidence intervals under correlated errors. "
            "Forecast points are never used to choose the scale radius."
        ),
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "input_sha256": sha256(data_path.read_bytes()).hexdigest(),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "training_fraction": 0.60,
        "middle_fraction": 0.20,
        "outer_fraction": 0.20,
        "minimum_training_points": MIN_TRAINING_POINTS,
        "minimum_middle_points": MIN_MIDDLE_POINTS,
        "minimum_outer_points": MIN_OUTER_POINTS,
        "density_bounds_msun_pc3": list(density_bounds),
        "scale_radius_grid_kpc": grid.tolist(),
        "delta_chi_squared_thresholds": {"nominal_68_percent": DELTA_CHI_SQUARED_68, "nominal_95_percent": DELTA_CHI_SQUARED_95},
        "galaxy_count": len(eligible),
        "galaxy_ids": eligible,
        "source_sha256": {
            name: sha256((PROJECT_DIR / name).read_bytes()).hexdigest()
            for name in (
                "run_nfw_profile_grid.py", "nfw_profile_fit.py", "anchored_radial_split.py",
                "radial_holdout.py", "halo_models.py", "burkert_halo_fit.py", "data_preprocess.py",
            ) if (PROJECT_DIR / name).is_file()
        },
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Profiling {len(eligible)} galaxies at {len(grid)} fixed NFW scale radii.", flush=True)
    rows, point_rows = [], []
    for number, galaxy_id in enumerate(eligible, 1):
        galaxy = dataset.loc[dataset["ID"].eq(galaxy_id)]
        training, middle, outer = split_inner_middle_outer(galaxy)
        for fixed_radius in grid:
            row = {
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
                    row[f"{region}_chi_squared_per_point"] = score["held_out_chi_squared_per_point"]
                    row[f"{region}_rmse_kms"] = score["held_out_rmse_kms"]
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    row[f"{region}_prediction_error"] = str(error) or type(error).__name__
                    continue
                for radius, observed, uncertainty, baryonic_squared, prediction, residual in zip(
                    held_out["R_kpc"], held_out["Vobs_kms"], held_out["eV_kms"],
                    held_out["Vbar_squared_kms2"], score["held_out_prediction_kms"],
                    score["held_out_residuals_kms"],
                ):
                    point_rows.append({
                        "galaxy_id": galaxy_id,
                        "fixed_scale_radius_kpc": fixed_radius,
                        "region": region,
                        "radius_kpc": radius,
                        "observed_kms": observed,
                        "error_kms": uncertainty,
                        "baryonic_squared_kms2": baryonic_squared,
                        "prediction_kms": prediction,
                        "residual_kms": residual,
                        "normalized_residual": residual / uncertainty,
                    })
            rows.append(row)
        if number % 10 == 0 or number == len(eligible):
            print(f"  {number}/{len(eligible)} galaxies profiled", flush=True)

    profiled_rows, summaries = profile_summary(rows)
    profiled_rows.to_csv(run_dir / "profile_grid.csv", index=False)
    pd.DataFrame(point_rows).to_csv(run_dir / "profile_predictions.csv", index=False)
    summaries.to_csv(run_dir / "profile_summary.csv", index=False)
    manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved NFW profile-grid results under {run_dir}")
    return run_dir

