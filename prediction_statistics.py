"""Galaxy-level summaries for a completed outer-radius prediction run."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261114
SCORE_COLUMN = "held_out_chi_squared_per_point"


def paired_score_differences(results, configuration, *, score_column=SCORE_COLUMN):
    """Return NFW minus Burkert held-out scores, one value per galaxy."""
    subset = results.loc[results["configuration"].eq(configuration)]
    scores = subset.pivot(index="galaxy_id", columns="model", values=score_column)
    if not {"burkert", "nfw"}.issubset(scores.columns):
        raise ValueError("Each configuration needs one Burkert and one NFW score per galaxy.")
    differences = (scores["nfw"] - scores["burkert"]).dropna()
    return differences.astype(float)


def bootstrap_median_interval(values, *, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    """Return a deterministic percentile interval by resampling galaxies."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, values.size, size=(replicates, values.size))
    medians = np.median(values[samples], axis=1)
    lower, upper = np.quantile(medians, [0.025, 0.975])
    return float(np.median(values)), float(lower), float(upper)


def paired_score_summary(differences, *, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    """Summarize paired predictive scores without treating velocity points as independent."""
    differences = pd.Series(differences, dtype=float).dropna()
    median, lower, upper = bootstrap_median_interval(
        differences, replicates=replicates, seed=seed,
    )
    nfw_lower = int((differences < 0).sum())
    burkert_lower = int((differences > 0).sum())
    non_ties = nfw_lower + burkert_lower
    p_value = (
        float(binomtest(nfw_lower, non_ties, p=0.5).pvalue)
        if non_ties else np.nan
    )
    return {
        "galaxies_with_paired_scores": len(differences),
        "median_nfw_minus_burkert_score": median,
        "bootstrap_95_lower": lower,
        "bootstrap_95_upper": upper,
        "nfw_lower_score_galaxies": nfw_lower,
        "burkert_lower_score_galaxies": burkert_lower,
        "numerical_ties": len(differences) - non_ties,
        "two_sided_sign_test_p_value": p_value,
    }


def _preference(burkert, nfw):
    if not np.isfinite(burkert) or not np.isfinite(nfw):
        return "unavailable"
    if np.isclose(burkert, nfw, rtol=1e-5, atol=1e-8):
        return "near_tie"
    return "burkert" if burkert < nfw else "nfw"


def rank_transition_summary(results, configuration):
    """Count changes between inner-fit and held-out model preference."""
    subset = results.loc[results["configuration"].eq(configuration)]
    values = subset.pivot(
        index="galaxy_id", columns="model",
        values=["training_reduced_chi_squared", SCORE_COLUMN],
    )
    rows = []
    for galaxy_id, row in values.iterrows():
        train = _preference(
            row[("training_reduced_chi_squared", "burkert")],
            row[("training_reduced_chi_squared", "nfw")],
        )
        held_out = _preference(
            row[(SCORE_COLUMN, "burkert")], row[(SCORE_COLUMN, "nfw")],
        )
        rows.append({"galaxy_id": galaxy_id, "training_preference": train,
                     "held_out_preference": held_out})
    transitions = pd.DataFrame(rows)
    counts = transitions.value_counts(
        ["training_preference", "held_out_preference"], sort=False,
    ).rename("galaxies").reset_index()
    return transitions, counts


def bound_hit_summary(results):
    """Describe predictive scores by selected scale-radius upper-bound status."""
    rows = []
    for (configuration, model), group in results.groupby(["configuration", "model"], sort=True):
        for upper_hit, subset in group.groupby(group["scale_radius_bound_hit"].eq("upper")):
            scores = pd.to_numeric(subset[SCORE_COLUMN], errors="coerce").dropna()
            rows.append({
                "configuration": configuration,
                "model": model,
                "scale_radius_upper_hit": bool(upper_hit),
                "galaxies": len(scores),
                "median_held_out_chi_squared_per_point": scores.median(),
                "mean_held_out_chi_squared_per_point": scores.mean(),
            })
    return pd.DataFrame(rows)


def analyze_prediction_run(input_run_dir, output_dir=None, *, replicates=BOOTSTRAP_REPLICATES,
                           seed=BOOTSTRAP_SEED):
    """Write a separately versioned, galaxy-level statistical summary."""
    input_run_dir = Path(input_run_dir)
    results_path = input_run_dir / "fit_predictions.csv"
    manifest_path = input_run_dir / "manifest.json"
    if not results_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Input run must contain fit_predictions.csv and manifest.json.")
    results = pd.read_csv(results_path)
    configurations = sorted(results["configuration"].unique())
    root = Path(output_dir) if output_dir else input_run_dir.parent / "statistics"
    run_dir = root / datetime.now(timezone.utc).strftime("summary_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)

    paired_rows, transition_rows, transition_count_rows = [], [], []
    for configuration in configurations:
        differences = paired_score_differences(results, configuration)
        family = results.loc[results["configuration"].eq(configuration), "family"].iloc[0]
        paired_rows.append({
            "configuration": configuration, "family": family,
            **paired_score_summary(differences, replicates=replicates, seed=seed),
        })
        transitions, counts = rank_transition_summary(results, configuration)
        transition_rows.extend(dict(row, configuration=configuration, family=family)
                               for row in transitions.to_dict("records"))
        transition_count_rows.extend(dict(row, configuration=configuration, family=family)
                                     for row in counts.to_dict("records"))

    pd.DataFrame(paired_rows).to_csv(run_dir / "paired_score_summary.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(run_dir / "rank_transitions.csv", index=False)
    pd.DataFrame(transition_count_rows).to_csv(
        run_dir / "rank_transition_summary.csv", index=False,
    )
    bound_hit_summary(results).to_csv(run_dir / "bound_hit_summary.csv", index=False)
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_run": str(input_run_dir.resolve()),
        "input_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "input_fit_predictions_sha256": sha256(results_path.read_bytes()).hexdigest(),
        "resampling_unit": "galaxy",
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "interpretation": (
            "Percentile intervals and sign-test values are descriptive diagnostics "
            "for this selected SPARC subset; they do not address correlated velocity "
            "errors, selection effects, or multiple testing across configurations."
        ),
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Saved galaxy-level prediction statistics under {run_dir}")
    return run_dir
