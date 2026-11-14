"""Filter, fit, plot-support, and report utilities for the RAR workflow."""

import numpy as np
from scipy.optimize import least_squares

from data_preprocess import load_sparc_dataset


def rar_model(gbar, g0):
    """Return the empirical RAR prediction for baryonic acceleration ``gbar``."""
    return gbar / (1 - np.exp(-np.sqrt(gbar / g0)))


def filter_data(threshold=0.2, *, df=None, output_path="filtered_data.csv"):
    """Apply the RAR relative velocity-error cut and save the retained rows."""
    if df is None:
        df = load_sparc_dataset()
    data = df.assign(rel_err=df["eV_kms"] / df["Vobs_kms"])
    filtered = data.loc[data["rel_err"] < threshold]
    filtered.to_csv(output_path, index=False)
    print(f"Saved {len(filtered)} rows to {output_path}")
    return filtered


def fit_rar(df=None):
    """Fit the RAR acceleration scale ``g0`` to positive acceleration rows."""
    if df is None:
        df = load_sparc_dataset()
    mask = (df["g_bar"] > 0) & (df["g_obs"] > 0)
    gbar = df.loc[mask, "g_bar"].to_numpy()
    gobs = df.loc[mask, "g_obs"].to_numpy()
    gobs_err = df.loc[mask, "g_obs_err"].to_numpy()

    if np.any(np.isnan(gobs_err)) or np.all(gobs_err == 0):
        gobs_err = np.maximum(0.1 * gobs, 1e-12)

    def residuals(parameters):
        g0 = parameters[0]
        return (gobs - rar_model(gbar, g0)) / gobs_err

    result = least_squares(residuals, [1e-10], bounds=(1e-16, 1e-5))
    g0_fit = result.x[0]
    print(f"Best-fit g0: {g0_fit:.3e} m/s^2")
    return g0_fit


def generate_report(df=None, g0=None):
    """Print the RAR sample size and fitted acceleration scale."""
    if df is None:
        df = load_sparc_dataset()
    if g0 is None:
        g0 = fit_rar(df)

    print("===== SPARC SUMMARY REPORT =====")
    print("Number of galaxies:", df["ID"].nunique())
    print("Number of total points:", df.shape[0])
    print("Best-fit g0 (RAR):", g0)
    print("Report generation complete.")


if __name__ == "__main__":
    data = filter_data()
    generate_report(data)
