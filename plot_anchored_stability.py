"""Create manuscript figures from saved anchored-forecast summaries."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


RADIUS_PATTERN = re.compile(r"^radius_ceiling_([0-9]+)_kpc$")


def radius_ceiling_rows(frame):
    """Keep and numerically order radius-ceiling configurations only."""
    subset = frame.loc[frame["family"].eq("radius_ceiling")].copy()
    subset["radius_ceiling_kpc"] = subset["configuration"].map(
        lambda label: int(RADIUS_PATTERN.match(label).group(1))
        if RADIUS_PATTERN.match(label) else None,
    )
    subset = subset.dropna(subset=["radius_ceiling_kpc"])
    if subset.empty:
        raise ValueError("No radius-ceiling configurations were found.")
    subset["radius_ceiling_kpc"] = subset["radius_ceiling_kpc"].astype(int)
    return subset.sort_values(["region", "radius_ceiling_kpc"])


def _matching_input_run(stability_dir, statistics_dir):
    """Require both summaries to describe the same completed forecast run."""
    stability_manifest = json.loads(
        (Path(stability_dir) / "manifest.json").read_text(encoding="utf-8"),
    )
    statistics_manifest = json.loads(
        (Path(statistics_dir) / "manifest.json").read_text(encoding="utf-8"),
    )
    if stability_manifest.get("input_run") != statistics_manifest.get("input_run"):
        raise ValueError("Stability and statistics folders describe different forecast runs.")
    return stability_manifest["input_run"]


def plot_radius_ceiling_stability(stability_dir, statistics_dir, output_dir=None):
    """Plot numerical-bound preference changes and paired-score summaries."""
    stability_dir = Path(stability_dir)
    statistics_dir = Path(statistics_dir)
    input_run = _matching_input_run(stability_dir, statistics_dir)
    stability_path = stability_dir / "configuration_change_summary.csv"
    paired_path = statistics_dir / "paired_score_summary.csv"
    stability = radius_ceiling_rows(pd.read_csv(stability_path))
    paired = radius_ceiling_rows(pd.read_csv(paired_path))

    root = Path(output_dir) if output_dir else Path(input_run).parent / "figures"
    figure_dir = root / datetime.now(timezone.utc).strftime("stability_%Y%m%dT%H%M%S_%fZ")
    figure_dir.mkdir(parents=True, exist_ok=False)
    colors = {"middle": "#1f77b4", "outer": "#d55e00"}
    labels = {"middle": "Middle 20% forecast", "outer": "Far-outer 20% forecast"}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for region in ("middle", "outer"):
        subset = stability.loc[stability["region"].eq(region)]
        ax.plot(
            subset["radius_ceiling_kpc"], subset["preference_changes"], "o-",
            color=colors[region], label=labels[region],
        )
    ax.axvline(100, color="0.35", linestyle="--", linewidth=1, label="100 kpc baseline")
    ax.set_xscale("log")
    ax.set_xlabel("NFW scale-radius ceiling (kpc)")
    ax.set_ylabel("Galaxies with changed preferred model")
    ax.set_title("Forecast model preference under NFW scale-radius ceilings")
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout(pad=1.2)
    preference_png = figure_dir / "radius_ceiling_preference_changes.png"
    preference_pdf = figure_dir / "radius_ceiling_preference_changes.pdf"
    fig.savefig(preference_png, dpi=300)
    fig.savefig(preference_pdf)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for region in ("middle", "outer"):
        subset = paired.loc[paired["region"].eq(region)]
        median = subset["median_nfw_minus_burkert_score"]
        lower = median - subset["bootstrap_95_lower"]
        upper = subset["bootstrap_95_upper"] - median
        ax.errorbar(
            subset["radius_ceiling_kpc"], median, yerr=[lower, upper], fmt="o-",
            color=colors[region], capsize=3, label=labels[region],
        )
    ax.axhline(0, color="0.35", linewidth=1)
    ax.axvline(100, color="0.35", linestyle="--", linewidth=1, label="100 kpc baseline")
    ax.set_xscale("log")
    ax.set_xlabel("NFW scale-radius ceiling (kpc)")
    ax.set_ylabel("Median NFW − Burkert $\\chi^2$ per point")
    ax.set_title("Paired anchored-forecast score sensitivity")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.19, top=0.91)
    score_png = figure_dir / "radius_ceiling_paired_scores.png"
    score_pdf = figure_dir / "radius_ceiling_paired_scores.pdf"
    fig.savefig(score_png, dpi=300)
    fig.savefig(score_pdf)
    plt.close(fig)

    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_forecast_run": input_run,
        "stability_directory": str(stability_dir.resolve()),
        "statistics_directory": str(statistics_dir.resolve()),
        "configuration_change_summary_sha256": sha256(stability_path.read_bytes()).hexdigest(),
        "paired_score_summary_sha256": sha256(paired_path.read_bytes()).hexdigest(),
        "figures": [path.name for path in (preference_png, preference_pdf, score_png, score_pdf)],
        "interpretation": (
            "Negative paired scores favor NFW. Error bars are galaxy-resampled "
            "95% percentile intervals and do not correct for multiple configurations."
        ),
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (figure_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Saved anchored stability figures under {figure_dir}")
    return figure_dir

