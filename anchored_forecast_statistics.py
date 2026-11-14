"""Galaxy-level summaries for a completed fixed-training anchored forecast."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from prediction_statistics import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, paired_score_summary


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

    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_run": str(input_run_dir.resolve()),
        "input_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "input_fit_forecasts_sha256": sha256(results_path.read_bytes()).hexdigest(),
        "input_model_comparisons_sha256": sha256(comparisons_path.read_bytes()).hexdigest(),
        "resampling_unit": "galaxy",
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "interpretation": (
            "Middle and far-outer scores come from the same fixed inner fit. "
            "The interval and sign-test outputs are descriptive paired diagnostics; "
            "they do not model correlated velocity errors, selection effects, or "
            "multiple testing across configurations."
        ),
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Saved anchored-forecast statistics under {run_dir}")
    return run_dir

