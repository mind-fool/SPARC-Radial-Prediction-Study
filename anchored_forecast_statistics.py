"""Galaxy-level summaries for a completed fixed-training anchored forecast."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from prediction_statistics import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    bootstrap_median_interval,
    paired_score_summary,
)


REGIONS = ("middle", "outer")


def paired_score_differences(results, configuration, region):
    """Return NFW minus Burkert scores for one forecast region and configuration."""
    score_column = f"{region}_chi_squared_per_point"
    subset = results.loc[results["configuration"].eq(configuration)]
    scores = subset.pivot(index="galaxy_id", columns="model", values=score_column)
    if not {"burkert", "nfw"}.issubset(scores.columns):
        raise ValueError("Each configuration needs one Burkert and one NFW score per galaxy.")
    return (scores["nfw"] - scores["burkert"]).dropna().astype(float)


def _preference(burkert, nfw):
    """Use the same numerical near-tie rule as the production comparison."""
    from run_anchored_forecast import classify_scores

    return classify_scores(burkert, nfw)


def training_to_forecast_transitions(results, configuration, region):
    """Compare the inner fit preference to one out-of-sample forecast preference."""
    score_column = f"{region}_chi_squared_per_point"
    subset = results.loc[results["configuration"].eq(configuration)]
    values = subset.pivot(
        index="galaxy_id", columns="model",
        values=["training_reduced_chi_squared", score_column],
    )
    rows = []
    for galaxy_id, row in values.iterrows():
        rows.append({
            "galaxy_id": galaxy_id,
            "training_preference": _preference(
                row[("training_reduced_chi_squared", "burkert")],
                row[("training_reduced_chi_squared", "nfw")],
            ),
            "forecast_preference": _preference(
                row[(score_column, "burkert")], row[(score_column, "nfw")],
            ),
        })
    transitions = pd.DataFrame(rows)
    counts = transitions.value_counts(
        ["training_preference", "forecast_preference"], sort=False,
    ).rename("galaxies").reset_index()
    return transitions, counts


def region_preference_transitions(comparisons, configuration):
    """Compare middle and far-outer preferences from the same fixed inner fit."""
    subset = comparisons.loc[comparisons["configuration"].eq(configuration)]
    values = subset.pivot(index="galaxy_id", columns="region", values="lower_score_model")
    if not {"middle", "outer"}.issubset(values.columns):
        raise ValueError("Each configuration needs middle and outer comparisons.")
    transitions = values.rename(
        columns={"middle": "middle_preference", "outer": "outer_preference"},
    ).reset_index()
    counts = transitions.value_counts(
        ["middle_preference", "outer_preference"], sort=False,
    ).rename("galaxies").reset_index()
    return transitions, counts


def bound_hit_summary(results, region):
    """Describe regional forecast scores by selected scale-radius upper-bound status."""
    score_column = f"{region}_chi_squared_per_point"
    rows = []
    for (configuration, model), group in results.groupby(["configuration", "model"], sort=True):
        for upper_hit, subset in group.groupby(group["scale_radius_bound_hit"].fillna("").eq("upper")):
            scores = pd.to_numeric(subset[score_column], errors="coerce").dropna()
            rows.append({
                "configuration": configuration,
                "model": model,
                "region": region,
                "scale_radius_upper_hit": bool(upper_hit),
                "galaxies": len(scores),
                "median_chi_squared_per_point": scores.median(),
                "mean_chi_squared_per_point": scores.mean(),
            })
    return pd.DataFrame(rows)


def control_comparison_rows(results, controls):
    """Pair each halo score with a control on the same galaxy and radial region."""
    keys = ["galaxy_id", "region", "control"]
    if controls.duplicated(keys).any():
        raise ValueError("Forecast controls must have one row per galaxy, region, and control.")
    rows = []
    for region in REGIONS:
        halo = results[[
            "configuration", "family", "galaxy_id", "model",
            f"{region}_chi_squared_per_point",
        ]].rename(columns={f"{region}_chi_squared_per_point": "halo_score"})
        control = controls.loc[controls["region"].eq(region), [
            "galaxy_id", "control", "chi_squared_per_point", "radial_leverage",
        ]].rename(columns={"chi_squared_per_point": "control_score"})
        matched = halo.merge(control, on="galaxy_id", how="left", validate="many_to_many",
                             indicator=True)
        if matched["_merge"].ne("both").any():
            raise ValueError(f"Missing {region} forecast controls for some halo fits.")
        matched = matched.drop(columns="_merge")
        matched["region"] = region
        matched["halo_minus_control_score"] = matched.halo_score - matched.control_score
        near_tie = np.isclose(matched.halo_score, matched.control_score,
                              rtol=1e-5, atol=1e-8)
        finite = np.isfinite(matched.halo_score) & np.isfinite(matched.control_score)
        matched["lower_score"] = np.select(
            [~finite, near_tie, matched.halo_score < matched.control_score],
            ["prediction_failed", "near_tie", "halo"], default="control",
        )
        rows.append(matched)
    return pd.concat(rows, ignore_index=True)


def control_comparison_summary(comparisons, *, replicates=BOOTSTRAP_REPLICATES,
                               seed=BOOTSTRAP_SEED):
    """Summarize paired halo-versus-control scores by galaxy, not by point."""
    rows = []
    grouping = ["configuration", "family", "region", "model", "control"]
    for keys, group in comparisons.groupby(grouping, sort=True):
        finite = group.loc[np.isfinite(group.halo_minus_control_score)]
        median, lower, upper = bootstrap_median_interval(
            finite.halo_minus_control_score, replicates=replicates, seed=seed,
        )
        counts = group.lower_score.value_counts()
        halo_lower = int(counts.get("halo", 0))
        control_lower = int(counts.get("control", 0))
        tested = halo_lower + control_lower
        rows.append(dict(zip(grouping, keys),
                         galaxies_with_paired_scores=len(finite),
                         halo_lower_score_galaxies=halo_lower,
                         control_lower_score_galaxies=control_lower,
                         near_ties=int(counts.get("near_tie", 0)),
                         prediction_failures=int(counts.get("prediction_failed", 0)),
                         median_halo_score=float(finite.halo_score.median()),
                         median_control_score=float(finite.control_score.median()),
                         median_halo_minus_control_score=median,
                         bootstrap_95_lower=lower,
                         bootstrap_95_upper=upper,
                         two_sided_sign_test_p_value=(
                             float(binomtest(control_lower, tested, p=0.5).pvalue)
                             if tested else np.nan
                         )))
    return pd.DataFrame(rows)


def analyze_anchored_forecast(input_run_dir, output_dir=None, *,
                              replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    """Write paired summaries for middle and far-outer forecasts separately."""
    input_run_dir = Path(input_run_dir)
    results_path = input_run_dir / "fit_forecasts.csv"
    comparisons_path = input_run_dir / "model_comparisons.csv"
    manifest_path = input_run_dir / "manifest.json"
    if not all(path.is_file() for path in (results_path, comparisons_path, manifest_path)):
        raise FileNotFoundError(
            "Input run needs fit_forecasts.csv, model_comparisons.csv, and manifest.json."
        )
    results = pd.read_csv(results_path)
    comparisons = pd.read_csv(comparisons_path)
    configurations = sorted(results["configuration"].unique())
    root = Path(output_dir) if output_dir else input_run_dir.parent / "statistics"
    run_dir = root / datetime.now(timezone.utc).strftime("summary_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)

    paired_rows, training_rows, training_count_rows = [], [], []
    region_rows, region_count_rows, bound_rows = [], [], []
    for configuration in configurations:
        family = results.loc[results["configuration"].eq(configuration), "family"].iloc[0]
        for region in REGIONS:
            paired_rows.append({
                "configuration": configuration,
                "family": family,
                "region": region,
                **paired_score_summary(
                    paired_score_differences(results, configuration, region),
                    replicates=replicates, seed=seed,
                ),
            })
            transitions, counts = training_to_forecast_transitions(
                results, configuration, region,
            )
            training_rows.extend(
                dict(row, configuration=configuration, family=family, region=region)
                for row in transitions.to_dict("records")
            )
            training_count_rows.extend(
                dict(row, configuration=configuration, family=family, region=region)
                for row in counts.to_dict("records")
            )

        transitions, counts = region_preference_transitions(comparisons, configuration)
        region_rows.extend(
            dict(row, configuration=configuration, family=family)
            for row in transitions.to_dict("records")
        )
        region_count_rows.extend(
            dict(row, configuration=configuration, family=family)
            for row in counts.to_dict("records")
        )

    for region in REGIONS:
        bound_rows.extend(bound_hit_summary(results, region).to_dict("records"))

    pd.DataFrame(paired_rows).to_csv(run_dir / "paired_score_summary.csv", index=False)
    pd.DataFrame(training_rows).to_csv(run_dir / "training_to_forecast_transitions.csv", index=False)
    pd.DataFrame(training_count_rows).to_csv(
        run_dir / "training_to_forecast_transition_summary.csv", index=False,
    )
    pd.DataFrame(region_rows).to_csv(run_dir / "middle_to_outer_transitions.csv", index=False)
    pd.DataFrame(region_count_rows).to_csv(
        run_dir / "middle_to_outer_transition_summary.csv", index=False,
    )
    pd.DataFrame(bound_rows).to_csv(run_dir / "bound_hit_summary.csv", index=False)

    controls_path = input_run_dir / "forecast_controls.csv"
    if controls_path.is_file():
        controls = pd.read_csv(controls_path)
        control_comparisons = control_comparison_rows(results, controls)
        control_comparisons.to_csv(
            run_dir / "forecast_control_comparisons.csv", index=False,
        )
        control_comparison_summary(
            control_comparisons, replicates=replicates, seed=seed,
        ).to_csv(run_dir / "forecast_control_summary.csv", index=False)

    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_run": str(input_run_dir.resolve()),
        "input_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "input_fit_forecasts_sha256": sha256(results_path.read_bytes()).hexdigest(),
        "input_model_comparisons_sha256": sha256(comparisons_path.read_bytes()).hexdigest(),
        "input_forecast_controls_sha256": (
            sha256(controls_path.read_bytes()).hexdigest()
            if controls_path.is_file() else None
        ),
        "resampling_unit": "galaxy",
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "interpretation": (
            "Middle and far-outer scores come from the same fixed inner fit. "
            "The interval and sign-test outputs are descriptive paired diagnostics; "
            "they do not model correlated velocity errors, selection effects, or "
            "multiple testing across configurations. Flat forecasts are exploratory "
            "controls and are not alternative physical models."
        ),
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Saved anchored-forecast statistics under {run_dir}")
    return run_dir

