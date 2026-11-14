"""Predict held-out outer SPARC rotation-curve measurements from inner fits.

Each Burkert or NFW fit uses only the inner velocity measurements of one
galaxy. Outer observed velocities are reserved for scoring, while the published
baryonic decomposition at those radii remains available as a model input.
Every pre-specified numerical-bound configuration is evaluated separately; no
configuration is selected using held-out velocities.
"""

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

from data_preprocess import load_sparc_dataset
from halo_fit_selection import fit_best_halo
from radial_holdout import (
    MIN_HELD_OUT_POINTS,
    MIN_TRAIN_POINTS,
    PRIMARY_HOLDOUT_FRACTION,
    eligible_galaxy_ids,
    score_held_out_prediction,
    split_inner_outer,
)
from report_converter import convert_rtf_to_text, convert_text_to_csv
from run_boundary_sensitivity import boundary_configurations


PROJECT_DIR = Path(__file__).resolve().parent


def ensure_input_data(data_path):
    """Create the CSV from the bundled SPARC report when a checkout is fresh."""
    data_path = Path(data_path)
    if data_path.is_file():
        return data_path
    text_path = PROJECT_DIR / "Galaxy_textfile.txt"
    if not text_path.is_file():
        rtf_path = PROJECT_DIR / "Galaxy_textfile.rtf"
        if not rtf_path.is_file():
            raise FileNotFoundError(
                f"Missing input data {data_path} and bundled report {rtf_path}."
            )
        if convert_rtf_to_text(rtf_path, text_path) is None:
            raise RuntimeError("Could not convert the bundled SPARC RTF report.")
    convert_text_to_csv(text_path, data_path)
    if not data_path.is_file():
        raise RuntimeError(f"Could not create input data: {data_path}")
    return data_path


def classify_prediction_scores(burkert_score, nfw_score):
    """Classify matched held-out scores using a numerical near-tie tolerance."""
    if not np.isfinite(burkert_score) or not np.isfinite(nfw_score):
        return "prediction_failed"
    if np.isclose(nfw_score, burkert_score, rtol=1e-5, atol=1e-8):
        return "near_tie"
    return "burkert" if burkert_score < nfw_score else "nfw"


def fit_and_predict(galaxy_id, training, held_out, model, configuration):
    """Fit inner measurements and score fixed parameters on outer measurements."""
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
        "prediction_error": "",
        "held_out_chi_squared": np.nan,
        "held_out_chi_squared_per_point": np.nan,
        "held_out_rmse_kms": np.nan,
    }
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
    try:
        score = score_held_out_prediction(fit, held_out, model=model)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        fields["prediction_error"] = str(error) or type(error).__name__
        return fields, []

    fields.update({
        key: score[key] for key in (
            "held_out_chi_squared", "held_out_chi_squared_per_point", "held_out_rmse_kms",
        )
    })
    point_rows = []
    for radius, observed, error, baryonic_squared, prediction, residual in zip(
        held_out["R_kpc"], held_out["Vobs_kms"], held_out["eV_kms"],
        held_out["Vbar_squared_kms2"], score["held_out_prediction_kms"],
        score["held_out_residuals_kms"],
    ):
        point_rows.append({
            "radius_kpc": radius,
            "observed_kms": observed,
            "error_kms": error,
            "baryonic_squared_kms2": baryonic_squared,
            "prediction_kms": prediction,
            "residual_kms": residual,
            "normalized_residual": residual / error,
        })
    return fields, point_rows


def comparison_rows(results):
    """Compare held-out performance only between models with identical bounds."""
    rows = []
    for (configuration, galaxy_id), group in results.groupby(
        ["configuration", "galaxy_id"], sort=False,
    ):
        by_model = group.set_index("model")
        burkert = by_model.loc["burkert"]
        nfw = by_model.loc["nfw"]
        burkert_score = burkert["held_out_chi_squared_per_point"]
        nfw_score = nfw["held_out_chi_squared_per_point"]
        rows.append({
            "configuration": configuration,
            "family": burkert["family"],
            "galaxy_id": galaxy_id,
            "burkert_held_out_chi_squared_per_point": burkert_score,
            "nfw_held_out_chi_squared_per_point": nfw_score,
            "delta_held_out_chi_squared_per_point_nfw_minus_burkert": (
                nfw_score - burkert_score
            ),
            "lower_held_out_score_model": classify_prediction_scores(
                burkert_score, nfw_score,
            ),
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


def _source_hashes():
    names = (
        "run_radial_prediction.py", "radial_holdout.py", "burkert_halo_fit.py",
        "halo_fit_selection.py", "optimizer_starts.py", "halo_models.py",
        "data_preprocess.py", "run_boundary_sensitivity.py",
    )
    return {
        name: sha256((PROJECT_DIR / name).read_bytes()).hexdigest()
        for name in names if (PROJECT_DIR / name).is_file()
    }


def run_radial_prediction(
    output_dir=None, *, data_path=None, configurations=None, galaxy_ids=None,
):
    """Run the pre-specified inner-fit/outer-prediction study."""
    data_path = Path(data_path) if data_path else PROJECT_DIR / "GALAXY_CSVDATA.csv"
    data_path = ensure_input_data(data_path)
    configs = boundary_configurations() if configurations is None else list(configurations)
    if not configs:
        raise ValueError("At least one numerical-bound configuration is required.")
    dataset = load_sparc_dataset(data_path)
    eligible = eligible_galaxy_ids(dataset)
    if not eligible:
        raise ValueError("No galaxies satisfy the primary radial holdout criteria.")
    if galaxy_ids is not None:
        available = {name.casefold(): name for name in eligible}
        selected = []
        for requested_id in galaxy_ids:
            canonical = available.get(requested_id.strip().casefold())
            if canonical is None:
                raise ValueError(
                    f"Galaxy {requested_id!r} does not satisfy the primary radial holdout criteria."
                )
            if canonical not in selected:
                selected.append(canonical)
        eligible = selected
    if not eligible:
        raise ValueError("Choose at least one eligible galaxy.")

    root = Path(output_dir) if output_dir else PROJECT_DIR / "outputs" / "radial_prediction"
    run_dir = root / datetime.now(timezone.utc).strftime("outer_holdout_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    commit, dirty = _git_metadata()
    manifest = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Predict held-out outer rotation-curve velocities from inner-radius fits.",
        "interpretation": (
            "Each bound configuration is evaluated separately. No fit configuration "
            "is selected using held-out velocities. Held-out points are from the same "
            "published rotation curve and are not an independent observing campaign."
        ),
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "input_sha256": sha256(data_path.read_bytes()).hexdigest(),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "holdout_fraction": PRIMARY_HOLDOUT_FRACTION,
        "minimum_training_points": MIN_TRAIN_POINTS,
        "minimum_held_out_points": MIN_HELD_OUT_POINTS,
        "galaxy_count": len(eligible),
        "galaxy_ids": eligible,
        "configurations": configs,
        "source_sha256": _source_hashes(),
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    total_fits = len(eligible) * len(configs) * 2
    print(
        f"Predicting outer radii for {len(eligible)} galaxies in {len(configs)} "
        f"configurations ({total_fits} model fits; ten starts per fit).",
        flush=True,
    )
    result_rows, point_rows = [], []
    for config_number, configuration in enumerate(configs, 1):
        print(f"{config_number}/{len(configs)} {configuration['label']}", flush=True)
        for number, galaxy_id in enumerate(eligible, 1):
            galaxy = dataset.loc[dataset["ID"].eq(galaxy_id)]
            training, held_out = split_inner_outer(galaxy)
            for model in ("burkert", "nfw"):
                fields, points = fit_and_predict(
                    galaxy_id, training, held_out, model, configuration,
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
                    "n_held_out": len(held_out),
                    "training_radius_max_kpc": training["R_kpc"].max(),
                    "held_out_radius_min_kpc": held_out["R_kpc"].min(),
                    "held_out_radius_max_kpc": held_out["R_kpc"].max(),
                }
                result_rows.append({**metadata, **fields})
                point_rows.extend({**metadata, **point} for point in points)
            if number % 25 == 0 or number == len(eligible):
                print(f"  {number}/{len(eligible)} galaxies fit", flush=True)
        pd.DataFrame(result_rows).to_csv(run_dir / "fit_predictions.csv", index=False)
        pd.DataFrame(point_rows).to_csv(run_dir / "held_out_points.csv", index=False)

    results = pd.DataFrame(result_rows)
    comparisons = comparison_rows(results)
    comparisons.to_csv(run_dir / "model_comparisons.csv", index=False)
    summary = (
        results.assign(
            fit_failed=results["fit_error"].ne(""),
            prediction_failed=results["prediction_error"].ne(""),
            density_lower_hit=results["density_bound_hit"].eq("lower"),
            density_upper_hit=results["density_bound_hit"].eq("upper"),
            radius_lower_hit=results["scale_radius_bound_hit"].eq("lower"),
            radius_upper_hit=results["scale_radius_bound_hit"].eq("upper"),
        )
        .groupby(["configuration", "family", "model"], as_index=False)
        .agg(
            galaxies=("galaxy_id", "size"),
            fit_failures=("fit_failed", "sum"),
            prediction_failures=("prediction_failed", "sum"),
            density_lower_hits=("density_lower_hit", "sum"),
            density_upper_hits=("density_upper_hit", "sum"),
            radius_lower_hits=("radius_lower_hit", "sum"),
            radius_upper_hits=("radius_upper_hit", "sum"),
            median_held_out_chi_squared_per_point=("held_out_chi_squared_per_point", "median"),
            median_held_out_rmse_kms=("held_out_rmse_kms", "median"),
        )
    )
    summary.to_csv(run_dir / "configuration_summary.csv", index=False)
    comparison_summary = (
        comparisons.groupby("configuration")["lower_held_out_score_model"]
        .value_counts().unstack(fill_value=0).reset_index()
    )
    comparison_summary.to_csv(run_dir / "comparison_summary.csv", index=False)
    manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved radial-prediction results under {run_dir}")
    return run_dir


if __name__ == "__main__":
    run_radial_prediction()
