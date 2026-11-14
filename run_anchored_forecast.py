"""Forecast middle and far-outer rotation-curve points from one inner fit."""

from contextlib import redirect_stdout
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
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
from halo_fit_selection import fit_best_halo
from radial_holdout import score_held_out_prediction
from run_boundary_sensitivity import boundary_configurations
from run_radial_prediction import ensure_input_data


PROJECT_DIR = Path(__file__).resolve().parent


def classify_scores(burkert_score, nfw_score):
    """Classify matched prediction scores with a numerical near-tie tolerance."""
    if not np.isfinite(burkert_score) or not np.isfinite(nfw_score):
        return "prediction_failed"
    if np.isclose(burkert_score, nfw_score, rtol=1e-5, atol=1e-8):
        return "near_tie"
    return "burkert" if burkert_score < nfw_score else "nfw"


def fit_and_forecast(galaxy_id, training, regions, model, configuration):
    """Fit inner data once and score that fit in each later radial region."""
    density_name, radius_name = (
        ("rho0", "r0") if model == "burkert" else ("rho_s", "r_s")
    )
    fields = {
        "training_chi_squared": np.nan,
        "training_reduced_chi_squared": np.nan,
        "density_msun_pc3": np.nan,
        "scale_radius_kpc": np.nan,
        "density_bound_hit": "",
        "scale_radius_bound_hit": "",
        "fit_parameterization": "",
        "physical_boundary_hit": "",
        "fit_strategy": "",
        "selection_trial_count": np.nan,
        "selection_success_count": np.nan,
        "selection_failed_count": np.nan,
        "selection_selected_trial": np.nan,
        "selection_agreement_count": np.nan,
        "selection_trials_json": "",
        "fit_error": "",
    }
    for region in regions:
        fields.update({
            f"{region}_prediction_error": "",
            f"{region}_chi_squared": np.nan,
            f"{region}_chi_squared_per_point": np.nan,
            f"{region}_rmse_kms": np.nan,
        })
    try:
        with redirect_stdout(StringIO()):
            fit = fit_best_halo(
                galaxy_id, df=training, model=model, plot=False,
                min_scale_radius_kpc=configuration["min_scale_radius_kpc"],
                max_scale_radius_kpc=configuration["max_scale_radius_kpc"],
                density_bounds=configuration["density_bounds"],
            )
        if not np.isfinite(fit["reduced_chi_squared"]):
            raise ValueError("Training reduced chi-squared is not finite.")
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        fields["fit_error"] = str(error) or type(error).__name__
        return fields, []

    fields.update(
        training_chi_squared=fit["chi_squared"],
        training_reduced_chi_squared=fit["reduced_chi_squared"],
        density_msun_pc3=fit[f"{density_name}_msun_pc3"],
        scale_radius_kpc=fit[f"{radius_name}_kpc"],
        density_bound_hit=fit["bound_hits"].get(density_name, ""),
        scale_radius_bound_hit=fit["bound_hits"].get(radius_name, ""),
        fit_parameterization=fit["fit_parameterization"],
        physical_boundary_hit=fit["physical_boundary_hit"],
        fit_strategy=fit["fit_strategy"],
        selection_trial_count=fit["selection_trial_count"],
        selection_success_count=fit["selection_success_count"],
        selection_failed_count=fit["selection_failed_count"],
        selection_selected_trial=fit["selection_selected_trial"],
        selection_agreement_count=fit["selection_agreement_count"],
        selection_trials_json=pd.DataFrame(fit["selection_trial_results"]).to_json(
            orient="records", double_precision=15,
        ),
    )
    point_rows = []
    for region, held_out in regions.items():
        try:
            score = score_held_out_prediction(fit, held_out, model=model)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            fields[f"{region}_prediction_error"] = str(error) or type(error).__name__
            continue
        fields.update({
            f"{region}_chi_squared": score["held_out_chi_squared"],
            f"{region}_chi_squared_per_point": score["held_out_chi_squared_per_point"],
            f"{region}_rmse_kms": score["held_out_rmse_kms"],
        })
        for radius, observed, error, baryonic_squared, prediction, residual in zip(
            held_out["R_kpc"], held_out["Vobs_kms"], held_out["eV_kms"],
            held_out["Vbar_squared_kms2"], score["held_out_prediction_kms"],
            score["held_out_residuals_kms"],
        ):
            point_rows.append({
                "region": region,
                "radius_kpc": radius,
                "observed_kms": observed,
                "error_kms": error,
                "baryonic_squared_kms2": baryonic_squared,
                "prediction_kms": prediction,
                "residual_kms": residual,
                "normalized_residual": residual / error,
            })
    return fields, point_rows


def comparison_rows(results, region):
    """Compare models only for matching bounds and one prediction region."""
    score_column = f"{region}_chi_squared_per_point"
    rows = []
    for (configuration, galaxy_id), group in results.groupby(
        ["configuration", "galaxy_id"], sort=False,
    ):
        by_model = group.set_index("model")
        burkert_score = by_model.loc["burkert", score_column]
        nfw_score = by_model.loc["nfw", score_column]
        rows.append({
            "configuration": configuration,
            "family": by_model.loc["burkert", "family"],
            "galaxy_id": galaxy_id,
            "region": region,
            "burkert_chi_squared_per_point": burkert_score,
            "nfw_chi_squared_per_point": nfw_score,
            "delta_nfw_minus_burkert_chi_squared_per_point": nfw_score - burkert_score,
            "lower_score_model": classify_scores(burkert_score, nfw_score),
        })
    return pd.DataFrame(rows)


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


def run_anchored_forecast(output_dir=None, *, data_path=None, configurations=None,
                          galaxy_ids=None):
    """Run a fixed-inner-fit forecast on middle and far-outer radial regions."""
    data_path = ensure_input_data(
        Path(data_path) if data_path else PROJECT_DIR / "GALAXY_CSVDATA.csv",
    )
    configs = boundary_configurations() if configurations is None else list(configurations)
    if not configs:
        raise ValueError("At least one numerical-bound configuration is required.")
    dataset = load_sparc_dataset(data_path)
    eligible = eligible_anchored_galaxy_ids(dataset)
    if galaxy_ids is not None:
        available = {name.casefold(): name for name in eligible}
        selected = []
        for requested_id in galaxy_ids:
            canonical = available.get(requested_id.strip().casefold())
            if canonical is None:
                raise ValueError(f"Galaxy {requested_id!r} is not eligible for the anchored forecast.")
            if canonical not in selected:
                selected.append(canonical)
        eligible = selected
    if not eligible:
        raise ValueError("No galaxies satisfy the anchored forecast criteria.")

    root = Path(output_dir) if output_dir else PROJECT_DIR / "outputs" / "anchored_forecast"
    run_dir = root / datetime.now(timezone.utc).strftime("inner60_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    commit, dirty = _git_metadata()
    manifest = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Fit inner 60% once and forecast the middle and far-outer 20% regions.",
        "interpretation": (
            "A model is never refit after the inner region. The middle and outer "
            "points come from the same published curve and are not independent surveys."
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
        "galaxy_count": len(eligible),
        "galaxy_ids": eligible,
        "configurations": configs,
        "source_sha256": {
            name: sha256((PROJECT_DIR / name).read_bytes()).hexdigest()
            for name in (
                "run_anchored_forecast.py", "anchored_radial_split.py", "radial_holdout.py",
                "burkert_halo_fit.py", "halo_fit_selection.py", "optimizer_starts.py",
                "halo_models.py", "data_preprocess.py", "run_boundary_sensitivity.py",
            ) if (PROJECT_DIR / name).is_file()
        },
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Forecasting {len(eligible)} galaxies in {len(configs)} configurations "
        f"({len(eligible) * len(configs) * 2} model fits; ten starts per fit).",
        flush=True,
    )
    result_rows, point_rows = [], []
    for config_number, configuration in enumerate(configs, 1):
        print(f"{config_number}/{len(configs)} {configuration['label']}", flush=True)
        for number, galaxy_id in enumerate(eligible, 1):
            galaxy = dataset.loc[dataset["ID"].eq(galaxy_id)]
            training, middle, outer = split_inner_middle_outer(galaxy)
            regions = {"middle": middle, "outer": outer}
            for model in ("burkert", "nfw"):
                fields, points = fit_and_forecast(
                    galaxy_id, training, regions, model, configuration,
                )
                metadata = {
                    "configuration": configuration["label"],
                    "family": configuration["family"],
                    "galaxy_id": galaxy_id,
                    "model": model,
                    "density_lower_bound_msun_pc3": configuration["density_bounds"][0],
                    "density_upper_bound_msun_pc3": configuration["density_bounds"][1],
                    "scale_radius_lower_bound_kpc": configuration["min_scale_radius_kpc"],
                    "scale_radius_upper_bound_kpc": configuration["max_scale_radius_kpc"],
                    "n_training": len(training),
                    "n_middle": len(middle),
                    "n_outer": len(outer),
                    "training_radius_max_kpc": training["R_kpc"].max(),
                    "middle_radius_min_kpc": middle["R_kpc"].min(),
                    "middle_radius_max_kpc": middle["R_kpc"].max(),
                    "outer_radius_min_kpc": outer["R_kpc"].min(),
                    "outer_radius_max_kpc": outer["R_kpc"].max(),
                }
                result_rows.append({**metadata, **fields})
                point_rows.extend({**metadata, **point} for point in points)
            if number % 20 == 0 or number == len(eligible):
                print(f"  {number}/{len(eligible)} galaxies fit", flush=True)
        pd.DataFrame(result_rows).to_csv(run_dir / "fit_forecasts.csv", index=False)
        pd.DataFrame(point_rows).to_csv(run_dir / "forecast_points.csv", index=False)

    results = pd.DataFrame(result_rows)
    comparisons = pd.concat(
        [comparison_rows(results, region) for region in ("middle", "outer")],
        ignore_index=True,
    )
    comparisons.to_csv(run_dir / "model_comparisons.csv", index=False)
    summary_rows = []
    for region in ("middle", "outer"):
        score = f"{region}_chi_squared_per_point"
        error = f"{region}_prediction_error"
        grouped = results.assign(prediction_failed=results[error].ne(""))
        summary = grouped.groupby(["configuration", "family", "model"], as_index=False).agg(
            galaxies=("galaxy_id", "size"),
            fit_failures=("fit_error", lambda values: values.ne("").sum()),
            prediction_failures=("prediction_failed", "sum"),
            median_chi_squared_per_point=(score, "median"),
            median_rmse_kms=(f"{region}_rmse_kms", "median"),
        )
        summary.insert(2, "region", region)
        summary_rows.append(summary)
    pd.concat(summary_rows, ignore_index=True).to_csv(
        run_dir / "forecast_summary.csv", index=False,
    )
    manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved anchored forecast results under {run_dir}")
    return run_dir


if __name__ == "__main__":
    run_anchored_forecast()
