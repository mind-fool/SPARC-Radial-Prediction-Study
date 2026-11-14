"""Summarize stellar mass-to-light sensitivity of NFW profile classifications."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd


DEFAULT_BASELINE = "baseline"


def status_count_summary(identifiability):
    """Count nominal NFW profile statuses separately for each stellar setting."""
    required = {"setting", "galaxy_id", "profile_status"}
    missing = required.difference(identifiability.columns)
    if missing:
        raise ValueError(f"Missing identifiability columns: {sorted(missing)}")
    return identifiability.value_counts(
        ["setting", "profile_status"], sort=False,
    ).rename("galaxies").reset_index().sort_values(["setting", "profile_status"])


def matched_profile_changes(identifiability, *, baseline=DEFAULT_BASELINE):
    """Match every stellar setting to baseline by galaxy without changing the sample."""
    required = {
        "setting", "galaxy_id", "profile_status", "minimum_training_chi_squared",
        "minimum_scale_radius_kpc", "nominal_95_lower_scale_radius_kpc",
        "nominal_95_upper_scale_radius_kpc",
    }
    missing = required.difference(identifiability.columns)
    if missing:
        raise ValueError(f"Missing identifiability columns: {sorted(missing)}")
    duplicated = identifiability.duplicated(["setting", "galaxy_id"])
    if duplicated.any():
        raise ValueError("Each setting must contain at most one row per galaxy.")
    baseline_rows = identifiability.loc[identifiability["setting"].eq(baseline)].copy()
    if baseline_rows.empty:
        raise ValueError(f"Baseline setting {baseline!r} was not found.")
    baseline_rows = baseline_rows.rename(columns={
        "profile_status": "baseline_profile_status",
        "minimum_training_chi_squared": "baseline_minimum_training_chi_squared",
        "minimum_scale_radius_kpc": "baseline_minimum_scale_radius_kpc",
        "nominal_95_lower_scale_radius_kpc": "baseline_nominal_95_lower_scale_radius_kpc",
        "nominal_95_upper_scale_radius_kpc": "baseline_nominal_95_upper_scale_radius_kpc",
    })[
        [
            "galaxy_id", "baseline_profile_status",
            "baseline_minimum_training_chi_squared", "baseline_minimum_scale_radius_kpc",
            "baseline_nominal_95_lower_scale_radius_kpc",
            "baseline_nominal_95_upper_scale_radius_kpc",
        ]
    ]
    current = identifiability.loc[~identifiability["setting"].eq(baseline)].copy()
    matched = current.merge(baseline_rows, on="galaxy_id", how="inner", validate="many_to_one")
    expected = len(baseline_rows) * current["setting"].nunique()
    if len(matched) != expected:
        raise ValueError("Every stellar setting must contain the same baseline galaxies.")
    matched["profile_status_changed"] = (
        matched["profile_status"] != matched["baseline_profile_status"]
    )
    matched["minimum_scale_radius_ratio_to_baseline"] = (
        matched["minimum_scale_radius_kpc"] / matched["baseline_minimum_scale_radius_kpc"]
    )
    return matched.sort_values(["setting", "galaxy_id"]).reset_index(drop=True)


def profile_change_summary(matched):
    """Summarize classification stability and the direction of any status changes."""
    required = {"setting", "profile_status_changed"}
    missing = required.difference(matched.columns)
    if missing:
        raise ValueError(f"Missing matched-change columns: {sorted(missing)}")
    summary = matched.groupby("setting", sort=True).agg(
        galaxies=("galaxy_id", "size"),
        profile_status_changes=("profile_status_changed", "sum"),
    ).reset_index()
    summary["profile_status_unchanged"] = (
        summary["galaxies"] - summary["profile_status_changes"]
    )
    return summary


def profile_status_transitions(matched):
    """Count baseline-to-alternative nominal profile classifications."""
    return matched.value_counts(
        ["setting", "baseline_profile_status", "profile_status"], sort=False,
    ).rename("galaxies").reset_index().sort_values(
        ["setting", "baseline_profile_status", "profile_status"],
    )


def analyze_nfw_profile_mass_to_light(input_run_dir, output_dir=None, *,
                                      baseline=DEFAULT_BASELINE):
    """Write a reproducible report for one completed profile-robustness run."""
    input_run_dir = Path(input_run_dir)
    status_path = input_run_dir / "profile_identifiability.csv"
    manifest_path = input_run_dir / "manifest.json"
    if not status_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Input run needs profile_identifiability.csv and manifest.json.",
        )
    identifiability = pd.read_csv(status_path)
    status_counts = status_count_summary(identifiability)
    matched = matched_profile_changes(identifiability, baseline=baseline)
    change_summary = profile_change_summary(matched)
    transitions = profile_status_transitions(matched)

    root = Path(output_dir) if output_dir else input_run_dir.parent / "statistics"
    run_dir = root / datetime.now(timezone.utc).strftime("summary_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    status_counts.to_csv(run_dir / "profile_status_summary.csv", index=False)
    matched.to_csv(run_dir / "galaxy_profile_changes.csv", index=False)
    change_summary.to_csv(run_dir / "profile_change_summary.csv", index=False)
    transitions.to_csv(run_dir / "profile_status_transitions.csv", index=False)
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_run": str(input_run_dir.resolve()),
        "input_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "input_profile_identifiability_sha256": sha256(status_path.read_bytes()).hexdigest(),
        "baseline_setting": baseline,
        "interpretation": (
            "These are controlled stellar mass-to-light sensitivity diagnostics. "
            "Each alternative setting is matched to baseline on identical galaxy IDs and "
            "shared ID-radius observations. A profile-status change means that the nominal "
            "95 percent fixed-grid profile classification changed; it is not a physical "
            "confidence statement."
        ),
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Saved mass-to-light NFW profile statistics under {run_dir}")
    return run_dir
