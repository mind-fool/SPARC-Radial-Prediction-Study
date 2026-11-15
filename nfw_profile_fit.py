"""One-dimensional NFW density fits at a fixed scale radius."""

import numpy as np
from scipy.optimize import brentq

from burkert_halo_fit import total_velocity_from_squared
from halo_models import nfw_halo_velocity


DEFAULT_DENSITY_BOUNDS = (1e-5, 10.0)


def _valid_arrays(galaxy_id, df):
    """Extract the rotation-curve arrays required for a fixed-radius profile fit."""
    galaxy = df.loc[df["ID"].eq(galaxy_id)].sort_values("R_kpc")
    if galaxy.empty:
        raise ValueError(f"Galaxy {galaxy_id!r} was not found.")
    radius = galaxy["R_kpc"].to_numpy(dtype=float)
    observed = galaxy["Vobs_kms"].to_numpy(dtype=float)
    error = galaxy["eV_kms"].to_numpy(dtype=float)
    baryonic_squared = galaxy["Vbar_squared_kms2"].to_numpy(dtype=float)
    valid = (
        np.isfinite(radius) & np.isfinite(observed) & np.isfinite(error)
        & np.isfinite(baryonic_squared) & (radius > 0) & (error > 0)
    )
    arrays = tuple(values[valid] for values in (radius, observed, error, baryonic_squared))
    if arrays[0].size < 3:
        raise ValueError(f"Galaxy {galaxy_id!r} has too few valid observations.")
    if np.any(arrays[1] <= 0):
        raise ValueError("Observed circular speeds must be positive for the density profile fit.")
    return arrays


def feasible_density_lower_bound(radius, baryonic_squared, fixed_scale_radius_kpc,
                                 density_bounds):
    """Return the density floor required for nonnegative total squared speed."""
    density_min, density_max = (float(value) for value in density_bounds)
    if not (np.isfinite(density_min) and np.isfinite(density_max)
            and 0 < density_min < density_max):
        raise ValueError("density_bounds must satisfy 0 < lower < upper.")
    if not np.isfinite(fixed_scale_radius_kpc) or fixed_scale_radius_kpc <= 0:
        raise ValueError("fixed_scale_radius_kpc must be finite and positive.")
    negative = baryonic_squared < 0
    if not np.any(negative):
        return density_min
    unit_halo_squared = nfw_halo_velocity(
        radius[negative], 1.0, fixed_scale_radius_kpc,
    ) ** 2
    if np.any(unit_halo_squared <= 0) or not np.all(np.isfinite(unit_halo_squared)):
        raise ValueError("Cannot determine a feasible density at the fixed scale radius.")
    required = np.max(-baryonic_squared[negative] / unit_halo_squared)
    # Avoid fitting exactly at a zero total-speed boundary.
    return max(density_min, required * (1.0 + 1e-12))


def fit_nfw_fixed_scale(galaxy_id, df, fixed_scale_radius_kpc, *,
                        density_bounds=DEFAULT_DENSITY_BOUNDS):
    """Fit NFW density at a specified scale radius using only the supplied rows.

    At fixed radius, squared halo speed is linear in density. For positive
    observed speeds, chi-squared is strictly convex in feasible density, so
    a bracketed derivative root or a bound gives its global minimum. The
    returned chi-squared is a training statistic with one fitted parameter.
    """
    radius, observed, error, baryonic_squared = _valid_arrays(galaxy_id, df)
    density_min, density_max = (float(value) for value in density_bounds)
    feasible_min = feasible_density_lower_bound(
        radius, baryonic_squared, fixed_scale_radius_kpc, density_bounds,
    )
    if feasible_min >= density_max:
        raise ValueError("No positive total-speed solution within the density bounds.")

    unit_halo_squared = nfw_halo_velocity(radius, 1.0, fixed_scale_radius_kpc) ** 2

    def evaluate(density):
        halo_squared = density * unit_halo_squared
        model = total_velocity_from_squared(baryonic_squared, halo_squared)
        residuals = observed - model
        return float(np.sum((residuals / error) ** 2)), model, residuals

    def derivative(log_density):
        total_squared = baryonic_squared + np.exp(log_density) * unit_halo_squared
        if np.any(total_squared <= 0):
            return -np.inf
        return float(np.sum(
            unit_halo_squared / error**2
            * (1.0 - observed / np.sqrt(total_squared))
        ))

    lower_log = np.log(feasible_min)
    upper_log = np.log(density_max)
    if derivative(lower_log) >= 0:
        density = feasible_min
    elif derivative(upper_log) <= 0:
        density = density_max
    else:
        density = float(np.exp(brentq(
            derivative, lower_log, upper_log, xtol=1e-12,
        )))
    chi_squared, model, residuals = evaluate(density)
    halo_squared = density * unit_halo_squared
    physical_boundary_hit = bool(np.any(
        (baryonic_squared < 0)
        & np.isclose(halo_squared, -baryonic_squared, rtol=1e-4, atol=0.0)
    ))
    if np.isclose(density, feasible_min, rtol=1e-5, atol=0.0):
        density_bound_hit = "feasible_lower" if feasible_min > density_min else "lower"
    elif np.isclose(density, density_max, rtol=1e-5, atol=0.0):
        density_bound_hit = "upper"
    else:
        density_bound_hit = ""
    return {
        "galaxy_id": galaxy_id,
        "rho_s_msun_pc3": density,
        "r_s_kpc": float(fixed_scale_radius_kpc),
        "fixed_scale_radius_kpc": float(fixed_scale_radius_kpc),
        "density_msun_pc3": density,
        "chi_squared": chi_squared,
        "reduced_chi_squared": chi_squared / (radius.size - 1),
        "density_lower_bound_msun_pc3": density_min,
        "density_upper_bound_msun_pc3": density_max,
        "feasible_density_lower_bound_msun_pc3": feasible_min,
        "density_bound_hit": density_bound_hit,
        "physical_boundary_hit": physical_boundary_hit,
        "density_optimizer": "monotone_derivative_brentq",
        "radius_kpc": radius,
        "observed_kms": observed,
        "error_kms": error,
        "baryonic_squared_kms2": baryonic_squared,
        "model_kms": model,
        "residuals_kms": residuals,
    }

