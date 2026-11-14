"""Compare radial-holdout horizons on their common galaxy subset."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prediction_statistics import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    paired_score_differences,
    paired_score_summary,
)


SCORE_COLUMN = "held_out_chi_squared_per_point"


def load_prediction_run(run_dir):
    """Load one completed prediction run with its recorded holdout fraction."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "fit_predictions.csv"
    if not manifest_path.is_file() or not results_path.is_file():
        raise FileNotFoundError(f"{run_dir} is not a completed prediction result folder.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Prediction run is not complete: {run_dir}")
    return manifest, pd.read_csv(results_path)


def common_galaxy_ids(frames):
    """Return galaxies present for both models in every configuration and horizon."""
    available = []
    for results in frames:
        counts = results.groupby("galaxy_id")["model"].nunique()
        available.append(set(counts.index[counts.eq(2)]))
    return sorted(set.intersection(*available))


def _winner_table(results, configuration, common_ids):
    subset = results.loc[
        results["configuration"].eq(configuration) & results["galaxy_id"].isin(common_ids)
    ]
    scores = subset.pivot(index="galaxy_id", columns="model", values=SCORE_COLUMN)
    winner = np.where(scores["burkert"] < scores["nfw"], "burkert", "nfw")
    return pd.Series(winner, index=scores.index, name="winner")


def analyze_horizons(run_dirs, output_dir=None, *, replicates=BOOTSTRAP_REPLICATES,
                     seed=BOOTSTRAP_SEED):
    """Summarize every shared configuration across pre-specified holdout horizons."""
    loaded = [load_prediction_run(path) for path in run_dirs]
    if len(loaded) < 2:
        raise ValueError("At least two completed prediction runs are required.")
    manifests, frames = zip(*loaded)
    fractions = [manifest["holdout_fraction"] for manifest in manifests]
    if len(set(fractions)) != len(fractions):
        raise ValueError("Each input run must use a distinct holdout fraction.")
    if len({manifest["input_sha256"] for manifest in manifests}) != 1:
        raise ValueError("All horizons must use the same input data file.")
    config_sets = [set(frame["configuration"].unique()) for frame in frames]
    if any(config_set != config_sets[0] for config_set in config_sets[1:]):
        raise ValueError("All horizons must evaluate the same bound configurations.")
    common_ids = common_galaxy_ids(frames)
    if not common_ids:
        raise ValueError("The selected horizons do not share any eligible galaxies.")
    configurations = sorted(config_sets[0])
    root = Path(output_dir) if output_dir else Path(run_dirs[0]).parent / "horizon_statistics"
    run_dir = root / datetime.now(timezone.utc).strftime("horizons_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)

    paired_rows, winner_rows = [], []
    labels = [f"horizon_{fraction:g}" for fraction in fractions]
    for configuration in configurations:
        for fraction, results in zip(fractions, frames):
            restricted = results.loc[results["galaxy_id"].isin(common_ids)]
            family = restricted.loc[
                restricted["configuration"].eq(configuration), "family"
            ].iloc[0]
            paired_rows.append({
                "configuration": configuration,
                "family": family,
                "holdout_fraction": fraction,
                **paired_score_summary(
                    paired_score_differences(restricted, configuration),
                    replicates=replicates, seed=seed,
                ),
            })
        winners = pd.concat(
            [
                _winner_table(results, configuration, common_ids).rename(label)
                for results, label in zip(frames, labels)
            ],
            axis=1,
        )
        winners.insert(0, "configuration", configuration)
        winner_rows.append(winners.reset_index(names="galaxy_id"))

    paired = pd.DataFrame(paired_rows)
    winners = pd.concat(winner_rows, ignore_index=True)
    paired.to_csv(run_dir / "common_sample_paired_score_summary.csv", index=False)
    winners.to_csv(run_dir / "common_sample_winners.csv", index=False)
    transition_counts = (
        winners.groupby(["configuration", *labels]).size()
        .rename("galaxies").reset_index()
    )
    transition_counts.to_csv(run_dir / "common_sample_winner_transitions.csv", index=False)
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_runs": [str(Path(path).resolve()) for path in run_dirs],
        "input_manifests_sha256": [
            sha256((Path(path) / "manifest.json").read_bytes()).hexdigest() for path in run_dirs
        ],
        "holdout_fractions": fractions,
        "common_galaxy_count": len(common_ids),
        "common_galaxy_ids": common_ids,
        "resampling_unit": "galaxy",
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "interpretation": (
            "All horizon comparisons use the common eligible subset. The horizons "
            "are nested rather than independent, and the multiple configurations "
            "are sensitivity analyses rather than separately confirmed discoveries."
        ),
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Saved common-sample horizon statistics under {run_dir}")
    return run_dir
