"""Summarize fixed-scale NFW profile-grid identifiability diagnostics."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _boolean_column(values):
    """Interpret Boolean values consistently after an in-memory or CSV round trip."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.casefold().eq("true") | values.eq(1)


def profile_status(summary, *, level="95"):
    """Label whether a nominal profile interval touches either grid edge."""
    lower = f"nominal_{level}_touches_lower_grid"
    upper = f"nominal_{level}_touches_upper_grid"
    required = {"galaxy_id", lower, upper}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Missing profile summary columns: {sorted(missing)}")
    result = summary.copy()
    lower_hit = _boolean_column(result[lower])
    upper_hit = _boolean_column(result[upper])
    result["profile_status"] = np.select(
        [lower_hit & upper_hit, lower_hit, upper_hit],
        ["open_both", "open_lower", "open_upper"],
        default="closed",
    )
    return result


def training_selected_profiles(profile_grid, profile_summary):
    """Choose each galaxy's radius using its training chi-squared only."""
    successful = profile_grid.loc[profile_grid["fit_error"].fillna("").eq("")].copy()
    ordered = successful.sort_values(
        ["galaxy_id", "training_chi_squared", "fixed_scale_radius_kpc"],
    )
    selected = ordered.groupby("galaxy_id", as_index=False).first()
    status = profile_status(profile_summary)[[
        "galaxy_id", "profile_status", "nominal_95_touches_lower_grid",
        "nominal_95_touches_upper_grid", "nominal_95_lower_scale_radius_kpc",
        "nominal_95_upper_scale_radius_kpc",
    ]]
    return selected.merge(status, on="galaxy_id", how="left", validate="one_to_one")


def representative_profiles(selected):
    """Choose deterministic cases for figures; never use held-out scores to choose them."""
    required = {"galaxy_id", "profile_status", "fixed_scale_radius_kpc", "training_chi_squared"}
    missing = required.difference(selected.columns)
    if missing:
        raise ValueError(f"Missing selected-profile columns: {sorted(missing)}")
    rows = []
    closed = selected.loc[selected["profile_status"].eq("closed")].copy()
    if not closed.empty:
        median_log_radius = np.median(np.log10(closed["fixed_scale_radius_kpc"]))
        representative = closed.assign(
            distance_from_closed_median=np.abs(
                np.log10(closed["fixed_scale_radius_kpc"]) - median_log_radius,
            ),
        ).sort_values(["distance_from_closed_median", "galaxy_id"]).iloc[0]
        rows.append({
            "role": "closed_profile_typical_scale",
            "selection_rule": "Closed profile nearest the closed-sample median log scale radius.",
            "galaxy_id": representative["galaxy_id"],
        })
    for status, role in (("open_upper", "upper_open_lowest_training_chi_squared"),
                         ("open_lower", "lower_open_lowest_training_chi_squared"),
                         ("open_both", "two_sided_open_lowest_training_chi_squared")):
        candidates = selected.loc[selected["profile_status"].eq(status)]
        if not candidates.empty:
            representative = candidates.sort_values(
                ["training_chi_squared", "galaxy_id"],
            ).iloc[0]
            rows.append({
                "role": role,
                "selection_rule": "Lowest inner-training chi-squared within this profile-status group.",
                "galaxy_id": representative["galaxy_id"],
            })
    return pd.DataFrame(rows)


def analyze_nfw_profile_grid(input_run_dir, output_dir=None):
    """Write reproducible profile-identifiability summaries for one completed run."""
    input_run_dir = Path(input_run_dir)
    grid_path = input_run_dir / "profile_grid.csv"
    summary_path = input_run_dir / "profile_summary.csv"
    manifest_path = input_run_dir / "manifest.json"
    if not all(path.is_file() for path in (grid_path, summary_path, manifest_path)):
        raise FileNotFoundError(
            "Input run needs profile_grid.csv, profile_summary.csv, and manifest.json.",
        )
    grid = pd.read_csv(grid_path)
    summary = pd.read_csv(summary_path)
    status = profile_status(summary)
    selected = training_selected_profiles(grid, summary)
    representatives = representative_profiles(selected)
    root = Path(output_dir) if output_dir else input_run_dir.parent / "statistics"
    run_dir = root / datetime.now(timezone.utc).strftime("summary_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)

    status.to_csv(run_dir / "profile_identifiability.csv", index=False)
    selected.to_csv(run_dir / "training_selected_profiles.csv", index=False)
    representatives.to_csv(run_dir / "representative_profiles.csv", index=False)
    profile_status_counts = status.value_counts("profile_status", sort=False).rename(
        "galaxies",
    ).reset_index()
    profile_status_counts.to_csv(run_dir / "profile_status_summary.csv", index=False)
    failures = grid.loc[grid["fit_error"].fillna("").ne("")].copy()
    failures.to_csv(run_dir / "profile_fit_failures.csv", index=False)

    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_run": str(input_run_dir.resolve()),
        "input_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "input_profile_grid_sha256": sha256(grid_path.read_bytes()).hexdigest(),
        "input_profile_summary_sha256": sha256(summary_path.read_bytes()).hexdigest(),
        "profile_interval_label": "nominal 95 percent Delta chi-squared profile interval",
        "interpretation": (
            "Profile status records whether the nominal training profile interval "
            "touches the pre-specified grid boundaries. It is not a physical confidence "
            "statement under correlated errors or unmodeled astrophysical systematics."
        ),
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Saved NFW profile-grid statistics under {run_dir}")
    return run_dir

