"""Leak-free inner-radius fitting and outer-radius prediction utilities."""

from math import ceil

import numpy as np

from burkert_halo_fit import HALO_MODELS, total_velocity_from_squared


PRIMARY_HOLDOUT_FRACTION = 0.30
MIN_TRAIN_POINTS = 8
MIN_HELD_OUT_POINTS = 4


def split_inner_outer(
    galaxy, *, holdout_fraction=PRIMARY_HOLDOUT_FRACTION,
    min_train_points=MIN_TRAIN_POINTS, min_held_out_points=MIN_HELD_OUT_POINTS,
):
    """Split a galaxy's ordered measurements into inner training and outer tests.

    The held-out velocities are never supplied to the fitting function. The
    baryonic decomposition is retained at held-out radii because it is an
    observed model input, not a target velocity.
    """
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be strictly between zero and one.")
    if min_train_points < 3 or min_held_out_points < 1:
        raise ValueError("At least three training points and one held-out point are required.")
    ordered = galaxy.sort_values("R_kpc").copy()
    required = ("R_kpc", "Vobs_kms", "eV_kms", "Vbar_squared_kms2")
    missing = [column for column in required if column not in ordered]
    if missing:
        raise ValueError(f"Galaxy data are missing required columns: {', '.join(missing)}.")
    if not np.all(np.isfinite(ordered.loc[:, required].to_numpy(dtype=float))):
        raise ValueError("Radial holdout requires finite fit and prediction inputs.")
    if not ordered["R_kpc"].is_unique:
        raise ValueError("Radial holdout requires unique measured radii within a galaxy.")

    n_total = len(ordered)
    n_held_out = max(min_held_out_points, ceil(n_total * holdout_fraction))
    n_train = n_total - n_held_out
    if n_train < min_train_points:
        raise ValueError(
            f"Need at least {min_train_points} inner and {min_held_out_points} outer points; "
            f"this galaxy has {n_total} measurements."
        )
    training = ordered.iloc[:n_train].copy()
    held_out = ordered.iloc[n_train:].copy()
    if training["R_kpc"].max() >= held_out["R_kpc"].min():
        raise ValueError("Training and held-out measurements must occupy distinct radial ranges.")
    return training, held_out


def eligible_galaxy_ids(dataset, **split_kwargs):
    """Return IDs that meet the pre-specified radial split requirements."""
    eligible = []
    for galaxy_id, galaxy in dataset.groupby("ID", sort=True):
        try:
            split_inner_outer(galaxy, **split_kwargs)
        except ValueError:
            continue
        eligible.append(galaxy_id)
    return eligible


def predict_held_out_velocity(fit, held_out, *, model):
    """Predict held-out velocities from a fit obtained only on inner radii."""
    if model not in HALO_MODELS:
        raise ValueError(f"Unknown halo model: {model!r}")
    halo_speed, _, density_name, radius_name = HALO_MODELS[model]
    try:
        density = float(fit[f"{density_name}_msun_pc3"])
        scale_radius = float(fit[f"{radius_name}_kpc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Fit does not contain finite parameters for the selected model.") from exc
    if not np.isfinite(density) or not np.isfinite(scale_radius):
        raise ValueError("Fit does not contain finite parameters for the selected model.")
    radius = held_out["R_kpc"].to_numpy(dtype=float)
    baryonic_squared = held_out["Vbar_squared_kms2"].to_numpy(dtype=float)
    halo_squared = halo_speed(radius, density, scale_radius) ** 2
    return total_velocity_from_squared(baryonic_squared, halo_squared)


def score_held_out_prediction(fit, held_out, *, model):
    """Return held-out prediction diagnostics without refitting parameters.

    ``chi_squared_per_point`` is not a reduced chi-squared: no parameters are
    fitted to the held-out velocities, so dividing by N-2 would be incorrect.
    """
    prediction = predict_held_out_velocity(fit, held_out, model=model)
    observed = held_out["Vobs_kms"].to_numpy(dtype=float)
    error = held_out["eV_kms"].to_numpy(dtype=float)
    if np.any(error <= 0) or not np.all(np.isfinite(error)):
        raise ValueError("Held-out velocity uncertainties must be finite and positive.")
    residuals = observed - prediction
    chi_squared = float(np.sum((residuals / error) ** 2))
    return {
        "n_held_out": len(held_out),
        "held_out_chi_squared": chi_squared,
        "held_out_chi_squared_per_point": chi_squared / len(held_out),
        "held_out_rmse_kms": float(np.sqrt(np.mean(residuals**2))),
        "held_out_residuals_kms": residuals,
        "held_out_prediction_kms": prediction,
    }
