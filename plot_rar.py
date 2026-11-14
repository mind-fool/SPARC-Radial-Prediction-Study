"""Plot the radial acceleration relation for a prepared SPARC dataset."""

import matplotlib.pyplot as plt
import numpy as np

from data_preprocess import load_sparc_dataset
from rar_analysis import rar_model


def plot_rar(df=None, output_path="RAR_plot.png", *, g0=None, show=True):
    """Save or show the positive-acceleration SPARC RAR and optional fitted curve."""
    if df is None:
        df = load_sparc_dataset()
    data = df.loc[(df["g_bar"] > 0) & (df["g_obs"] > 0)]

    plt.figure(figsize=(7, 6))
    plt.loglog(data["g_bar"], data["g_obs"], "o", markersize=3, alpha=0.6,
               label="SPARC data")

    x = np.logspace(np.log10(data["g_bar"].min()), np.log10(data["g_bar"].max()), 100)
    plt.loglog(x, x, "r--", label="y = x", linewidth=1.5)
    if g0 is not None:
        plt.loglog(x, rar_model(x, g0), color="black", linewidth=2,
                   label=f"RAR fit (g₀ = {g0:.3e} m/s²)")

    plt.xlabel("Baryonic Acceleration g_bar (m/s²)")
    plt.ylabel("Observed Acceleration g_obs (m/s²)")
    plt.title("Radial Acceleration Relation (RAR)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved RAR plot to {output_path}")
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    plot_rar()
