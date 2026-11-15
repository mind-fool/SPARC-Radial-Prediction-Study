"""Make a figure directly from one completed density-floor sweep's CSV data."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_density_floor_sweep(run_dir, *, example_galaxy="NGC0100"):
    """Show sample-wide profile statuses and a fixed training-profile example."""
    run_dir = Path(run_dir)
    counts = pd.read_csv(run_dir / "profile_status_counts.csv")
    grid = pd.read_csv(run_dir / "profile_grid.csv")
    floors = sorted(counts["density_floor_msun_pc3"].unique(), reverse=True)
    labels = [rf"$10^{{{int(round(np.log10(floor)))}}}$" for floor in floors]

    example = grid.loc[grid["galaxy_id"].eq(example_galaxy)]
    if example.empty:
        fig, axes = plt.subplots(1, 1, figsize=(8, 4.5))
        axes = [axes]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), layout="constrained")

    ax = axes[0]
    bottom = np.zeros(len(floors), dtype=int)
    for status, label, color in (
        ("closed", "Closed", "#2b6cb0"),
        ("open_upper", "Upper-open", "#e67e22"),
        ("open_lower", "Lower-open", "#6b7280"),
        ("open_both", "Open at both edges", "#9b59b6"),
    ):
        values = np.asarray([
            int(counts.loc[
                counts["density_floor_msun_pc3"].eq(floor)
                & counts["profile_status"].eq(status), "galaxies"
            ].sum()) for floor in floors
        ])
        ax.bar(range(len(floors)), values, bottom=bottom, label=label, color=color)
        bottom += values
    ax.set_xticks(range(len(floors)), labels)
    ax.set_xlabel(r"NFW density lower bound ($M_\odot\,\mathrm{pc}^{-3}$)")
    ax.set_ylabel("Galaxies")
    ax.set_ylim(0, 62)
    ax.set_title("Nominal NFW profile status by density floor")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)

    if not example.empty:
        ax = axes[1]
        for floor, color in ((1e-5, "#2b6cb0"), (1e-7, "#e67e22"), (1e-9, "#2e8b57")):
            subset = example.loc[
                example["density_floor_msun_pc3"].eq(floor)
                & example["fit_error"].fillna("").eq("")
            ].sort_values("fixed_scale_radius_kpc")
            if subset.empty:
                continue
            ax.plot(
                subset["fixed_scale_radius_kpc"],
                subset["delta_training_chi_squared"],
                marker=".", markersize=3, linewidth=1.5, color=color,
                label=rf"density floor $10^{{{int(round(np.log10(floor)))}}}$",
            )
        ax.axhline(3.84, color="black", linestyle="--", linewidth=1,
                   label=r"Nominal $\Delta\chi^2=3.84$")
        ax.set_xscale("log")
        ax.set_xlim(10, 10_000)
        ax.set_ylim(bottom=0, top=12)
        ax.set_xlabel("Fixed NFW scale radius (kpc)")
        ax.set_ylabel(r"Inner-training $\Delta\chi^2$")
        ax.set_title(f"{example_galaxy}: high-radius profile with the same inner measurements")
        ax.legend(fontsize=8)

    fig.savefig(run_dir / "density_floor_sensitivity.png", dpi=300)
    fig.savefig(run_dir / "density_floor_sensitivity.pdf")
    plt.close(fig)
    return run_dir / "density_floor_sensitivity.png"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot a completed NFW density-floor sweep.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(plot_density_floor_sweep(args.run_dir))
