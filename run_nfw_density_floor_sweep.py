"""Audit NFW profile closure against the numerical density lower bound."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy

from anchored_radial_split import eligible_anchored_galaxy_ids, split_inner_middle_outer
from data_preprocess import load_sparc_dataset
from nfw_profile_fit import DEFAULT_DENSITY_BOUNDS, fit_nfw_fixed_scale
from nfw_profile_statistics import profile_status
from plot_nfw_density_floor_sweep import plot_density_floor_sweep
from run_nfw_profile_grid import (
    DELTA_CHI_SQUARED_95,
    PROFILE_RADIUS_GRID_KPC,
    _git_metadata,
    profile_summary,
)
from run_radial_prediction import ensure_input_data


PROJECT_DIR = Path(__file__).resolve().parent
DENSITY_FLOORS_MSUN_PC3 = (1e-5, 1e-6, 1e-7, 1e-8, 1e-9)


def validate_sweep(floors, radius_grid, density_upper):
    """Require a descending sequence starting at the original density floor."""
    floors = np.asarray(floors, dtype=float)
    grid = np.asarray(radius_grid, dtype=float)
    if (floors.ndim != 1 or floors.size < 2 or not np.all(np.isfinite(floors))
            or np.any(floors <= 0) or np.any(floors >= density_upper)
            or not np.all(np.diff(floors) < 0)
            or not np.isclose(floors[0], DEFAULT_DENSITY_BOUNDS[0], rtol=0, atol=0)):
        raise ValueError("Density floors must start at 1e-5 and decrease below the fixed upper bound.")
    if (grid.ndim != 1 or grid.size < 3 or not np.all(np.isfinite(grid))
            or np.any(grid <= 0) or not np.all(np.diff(grid) > 0)):
        raise ValueError("Scale-radius grid must contain at least three increasing positive radii.")
    return floors, grid


def classify_floor_profiles(grid_rows, radius_grid):
    """Apply the existing nominal profile rule and separately flag failed edges."""
    detailed, summary = profile_summary(grid_rows)
    if summary.empty:
        raise ValueError("No successful fixed-radius fits were produced.")
    summary = profile_status(summary)
    failures = detailed.loc[detailed["fit_error"].fillna("").ne("")]
    for edge, radius in (("lower", radius_grid[0]), ("upper", radius_grid[-1])):
        failed_ids = set(failures.loc[
            np.isclose(failures["fixed_scale_radius_kpc"], radius), "galaxy_id"
        ])
        summary[f"{edge}_grid_fit_failed"] = summary["galaxy_id"].isin(failed_ids)
    summary["edge_status_ambiguous"] = (
        (summary["lower_grid_fit_failed"] & summary["nominal_95_touches_lower_grid"])
        | (summary["upper_grid_fit_failed"] & summary["nominal_95_touches_upper_grid"])
    )
    first_exclusion_at_floor = {}
    for galaxy_id, group in detailed.groupby("galaxy_id", sort=True):
        successful = group.loc[group["fit_error"].fillna("").eq("")].sort_values(
            "fixed_scale_radius_kpc"
        )
        best = successful.loc[successful["training_chi_squared"].idxmin()]
        above_best = successful.loc[
            successful["fixed_scale_radius_kpc"] > best["fixed_scale_radius_kpc"]
        ]
        excluded = above_best.loc[
            above_best["delta_training_chi_squared"] > DELTA_CHI_SQUARED_95
        ]
        first_exclusion_at_floor[galaxy_id] = (
            bool(excluded.iloc[0]["density_bound_hit"] == "lower")
            if not excluded.empty else False
        )
    summary["first_upper_exclusion_at_density_floor"] = summary["galaxy_id"].map(
        first_exclusion_at_floor
    )
    return detailed, summary


def run_nfw_density_floor_sweep(output_dir=None, *, data_path=None, galaxy_ids=None,
                                density_floors=DENSITY_FLOORS_MSUN_PC3,
                                radius_grid_kpc=PROFILE_RADIUS_GRID_KPC,
                                density_upper=DEFAULT_DENSITY_BOUNDS[1]):
    """Reprofile identical inner data and radius grid, changing only density floor."""
    floors, grid = validate_sweep(density_floors, radius_grid_kpc, density_upper)
    data_path = ensure_input_data(
        Path(data_path) if data_path else PROJECT_DIR / "GALAXY_CSVDATA.csv"
    )
    dataset = load_sparc_dataset(data_path)
    eligible = eligible_anchored_galaxy_ids(dataset)
    if galaxy_ids is not None:
        available = {name.casefold(): name for name in eligible}
        selected = []
        for requested in galaxy_ids:
            canonical = available.get(requested.strip().casefold())
            if canonical is None:
                raise ValueError(f"Galaxy {requested!r} is not eligible for the NFW profile study.")
            if canonical not in selected:
                selected.append(canonical)
        eligible = selected
    if not eligible:
        raise ValueError("No galaxies satisfy the anchored profile criteria.")

    root = Path(output_dir) if output_dir else PROJECT_DIR / "outputs" / "nfw_density_floor_sweep"
    run_dir = root / datetime.now(timezone.utc).strftime("inner60_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    commit, dirty = _git_metadata()
    manifest = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Exploratory numerical density-floor sensitivity of NFW scale-radius profiles.",
        "interpretation": (
            "Only the NFW density lower bound varies. Nominal Delta chi-squared profile "
            "statuses are numerical diagnostics, not physical confidence intervals. "
            "A boundary-induced status change does not identify a physical scale radius."
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
        "density_floors_msun_pc3": floors.tolist(),
        "density_upper_msun_pc3": density_upper,
        "scale_radius_grid_kpc": grid.tolist(),
        "delta_chi_squared_threshold": DELTA_CHI_SQUARED_95,
        "galaxy_ids": eligible,
        "source_sha256": {
            name: sha256((PROJECT_DIR / name).read_bytes()).hexdigest()
            for name in (
                "run_nfw_density_floor_sweep.py", "plot_nfw_density_floor_sweep.py",
                "run_nfw_profile_grid.py",
                "nfw_profile_fit.py", "nfw_profile_statistics.py", "anchored_radial_split.py",
                "data_preprocess.py", "halo_models.py", "burkert_halo_fit.py",
            )
        },
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    all_grid, all_status = [], []
    for floor in floors:
        rows = []
        print(f"Density floor {floor:.0e}: profiling {len(eligible)} galaxies.", flush=True)
        for number, galaxy_id in enumerate(eligible, 1):
            galaxy = dataset.loc[dataset["ID"].eq(galaxy_id)]
            training, _, _ = split_inner_middle_outer(galaxy)
            for radius in grid:
                row = {
                    "galaxy_id": galaxy_id,
                    "fixed_scale_radius_kpc": float(radius),
                    "training_chi_squared": np.nan,
                    "density_msun_pc3": np.nan,
                    "density_bound_hit": "",
                    "physical_boundary_hit": "",
                    "fit_error": "",
                }
                try:
                    fit = fit_nfw_fixed_scale(
                        galaxy_id, training, radius,
                        density_bounds=(float(floor), float(density_upper)),
                    )
                    row.update(
                        training_chi_squared=fit["chi_squared"],
                        density_msun_pc3=fit["density_msun_pc3"],
                        density_bound_hit=fit["density_bound_hit"],
                        physical_boundary_hit=fit["physical_boundary_hit"],
                    )
                except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                    row["fit_error"] = str(error) or type(error).__name__
                rows.append(row)
            if number % 10 == 0 or number == len(eligible):
                print(f"  {number}/{len(eligible)} galaxies", flush=True)
        detailed, summary = classify_floor_profiles(rows, grid)
        detailed.insert(0, "density_floor_msun_pc3", floor)
        summary.insert(0, "density_floor_msun_pc3", floor)
        all_grid.append(detailed)
        all_status.append(summary)

    grid_table = pd.concat(all_grid, ignore_index=True)
    status_table = pd.concat(all_status, ignore_index=True)
    baseline = status_table.loc[status_table["density_floor_msun_pc3"].eq(floors[0]), [
        "galaxy_id", "profile_status"
    ]].rename(columns={"profile_status": "baseline_profile_status"})
    status_table = status_table.merge(baseline, on="galaxy_id", validate="many_to_one")
    status_table["changed_from_baseline"] = (
        status_table["profile_status"] != status_table["baseline_profile_status"]
    )
    counts = status_table.groupby(
        ["density_floor_msun_pc3", "profile_status"], dropna=False
    ).size().rename("galaxies").reset_index()
    grid_table.to_csv(run_dir / "profile_grid.csv", index=False)
    status_table.to_csv(run_dir / "profile_status_by_floor.csv", index=False)
    counts.to_csv(run_dir / "profile_status_counts.csv", index=False)
    status_table.loc[status_table["changed_from_baseline"]].to_csv(
        run_dir / "status_changes_from_baseline.csv", index=False
    )
    plot_density_floor_sweep(run_dir)
    manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for floor in floors:
        subset = status_table.loc[status_table["density_floor_msun_pc3"].eq(floor)]
        print(
            f"{floor:.0e} | " + " | ".join(
                f"{name}: {int(subset['profile_status'].eq(name).sum())}"
                for name in ("closed", "open_upper", "open_lower", "open_both")
            ) + f" | changed: {int(subset['changed_from_baseline'].sum())}"
        )
        edge_failures = int((
            subset["lower_grid_fit_failed"] | subset["upper_grid_fit_failed"]
        ).sum())
        if edge_failures:
            print(f"  Galaxies with a failed radius-grid endpoint: {edge_failures}; see CSV flags.")
    print(f"Saved density-floor sensitivity results under {run_dir}")
    return run_dir


if __name__ == "__main__":
    run_nfw_density_floor_sweep()
