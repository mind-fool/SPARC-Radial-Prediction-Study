"""One-dimensional NFW density fits at a fixed scale radius."""

import numpy as np
from scipy.optimize import minimize_scalar

from burkert_halo_fit import total_velocity_from_squared
from halo_models import nfw_halo_velocity


DEFAULT_DENSITY_BOUNDS = (1e-5, 10.0)
DENSITY_GRID_POINTS = 81


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
                        density_bounds=DEFAULT_DENSITY_BOUNDS,
                        density_grid_points=DENSITY_GRID_POINTS):
    """Fit NFW density at a specified scale radius using only the supplied rows.

    The profile search is deterministic: it evaluates a logarithmic density grid,
    then refines the lowest grid interval in log density. The returned chi-squared
    is a training statistic, with one fitted parameter (density).
    """
    if density_grid_points < 5:
        raise ValueError("density_grid_points must be at least five.")
    radius, observed, error, baryonic_squared = _valid_arrays(galaxy_id, df)
    density_min, density_max = (float(value) for value in density_bounds)
    feasible_min = feasible_density_lower_bound(
        radius, baryonic_squared, fixed_scale_radius_kpc, density_bounds,
    )
    if feasible_min >= density_max:
        raise ValueError("No positive total-speed solution within the density bounds.")

    def evaluate(density):
        halo_squared = nfw_halo_velocity(radius, density, fixed_scale_radius_kpc) ** 2
        model = total_velocity_from_squared(baryonic_squared, halo_squared)
        residuals = observed - model
        return float(np.sum((residuals / error) ** 2)), model, residuals

    grid = np.geomspace(feasible_min, density_max, num=density_grid_points)
    grid_scores = np.asarray([evaluate(density)[0] for density in grid])
    best_index = int(np.argmin(grid_scores))
    candidates = [(grid_scores[best_index], grid[best_index])]
    if 0 < best_index < len(grid) - 1:
        lower_log = np.log(grid[best_index - 1])
        upper_log = np.log(grid[best_index + 1])
        refined = minimize_scalar(
            lambda log_density: evaluate(np.exp(log_density))[0],
            bounds=(lower_log, upper_log), method="bounded",
            options={"xatol": 1e-12},
        )
        if refined.success and np.isfinite(refined.fun):
            candidates.append((float(refined.fun), float(np.exp(refined.x))))
    chi_squared, density = min(candidates, key=lambda pair: pair[0])
    chi_squared, model, residuals = evaluate(density)
    halo_squared = nfw_halo_velocity(radius, density, fixed_scale_radius_kpc) ** 2
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
        "density_grid_points": density_grid_points,
        "radius_kpc": radius,
        "observed_kms": observed,
        "error_kms": error,
        "baryonic_squared_kms2": baryonic_squared,
        "model_kms": model,
        "residuals_kms": residuals,
    }

