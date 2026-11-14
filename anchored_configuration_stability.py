"""Quantify how anchored forecast results change across numerical bounds."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prediction_statistics import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, bootstrap_median_interval


DEFAULT_BASELINE = "radius_ceiling_100_kpc"
REGIONS = ("middle", "outer")


def matched_configuration_changes(comparisons, baseline=DEFAULT_BASELINE):
    """Match every configuration to the baseline by galaxy and forecast region."""
    required = {
        "configuration", "family", "galaxy_id", "region", "lower_score_model",
        "delta_nfw_minus_burkert_chi_squared_per_point",
    }
    missing = required.difference(comparisons.columns)
    if missing:
        raise ValueError(f"Missing comparison columns: {sorted(missing)}")
    baseline_rows = comparisons.loc[comparisons["configuration"].eq(baseline)].copy()
    if baseline_rows.empty:
        raise ValueError(f"Baseline configuration {baseline!r} was not found.")
    baseline_rows = baseline_rows.rename(columns={
        "lower_score_model": "baseline_preference",
        "delta_nfw_minus_burkert_chi_squared_per_point": "baseline_delta_score",
    })[["galaxy_id", "region", "baseline_preference", "baseline_delta_score"]]
    current = comparisons.rename(columns={
        "lower_score_model": "configuration_preference",
        "delta_nfw_minus_burkert_chi_squared_per_point": "configuration_delta_score",
    }).copy()
    matched = current.merge(baseline_rows, on=["galaxy_id", "region"], how="inner")
    matched["preference_changed"] = (
        matched["configuration_preference"] != matched["baseline_preference"]
    )
    matched["configuration_minus_baseline_delta_score"] = (
        matched["configuration_delta_score"] - matched["baseline_delta_score"]
    )
    return matched


def summarize_configuration_changes(matched, *, replicates=BOOTSTRAP_REPLICATES,
                                    seed=BOOTSTRAP_SEED):
    """Summarize preference and paired-score changes at the galaxy level."""
    rows = []
    grouped = matched.groupby(["configuration", "family", "region"], sort=True)
    for (configuration, family, region), subset in grouped:
        values = subset["configuration_minus_baseline_delta_score"]
        median, lower, upper = bootstrap_median_interval(
            values, replicates=replicates, seed=seed,
        )
        rows.append({
            "configuration": configuration,
            "family": family,
            "region": region,
            "galaxies_with_paired_scores": len(subset),
            "preference_changes": int(subset["preference_changed"].sum()),
            "preference_unchanged": int((~subset["preference_changed"]).sum()),
            "median_configuration_minus_baseline_delta_score": median,
            "bootstrap_95_lower": lower,
            "bootstrap_95_upper": upper,
        })
    return pd.DataFrame(rows)


def preference_transition_summary(matched):
    """Count every baseline-to-configuration preference transition."""
    return matched.value_counts(
        ["configuration", "family", "region", "baseline_preference", "configuration_preference"],
        sort=False,
    ).rename("galaxies").reset_index()


def analyze_configuration_stability(input_run_dir, output_dir=None, *,
                                    baseline=DEFAULT_BASELINE,
                                    replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    """Write reproducible numerical-bound stability diagnostics for one anchored run."""
    input_run_dir = Path(input_run_dir)
    comparisons_path = input_run_dir / "model_comparisons.csv"
    manifest_path = input_run_dir / "manifest.json"
    if not comparisons_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Input run needs model_comparisons.csv and manifest.json.")
    comparisons = pd.read_csv(comparisons_path)
    matched = matched_configuration_changes(comparisons, baseline=baseline)
    root = Path(output_dir) if output_dir else input_run_dir.parent / "configuration_stability"
    run_dir = root / datetime.now(timezone.utc).strftime("stability_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)

    matched.to_csv(run_dir / "galaxy_configuration_changes.csv", index=False)
    summarize_configuration_changes(
        matched, replicates=replicates, seed=seed,
    ).to_csv(run_dir / "configuration_change_summary.csv", index=False)
    preference_transition_summary(matched).to_csv(
        run_dir / "configuration_preference_transitions.csv", index=False,
    )
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_run": str(input_run_dir.resolve()),
        "input_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "input_model_comparisons_sha256": sha256(comparisons_path.read_bytes()).hexdigest(),
        "baseline_configuration": baseline,
        "resampling_unit": "galaxy",
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "interpretation": (
            "This is a numerical-sensitivity diagnostic. Each configuration is "
            "compared with the selected baseline on matching galaxies and regions. "
            "It does not select a preferred configuration and does not correct for "
            "multiple configuration comparisons."
        ),
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Saved anchored configuration-stability results under {run_dir}")
    return run_dir

