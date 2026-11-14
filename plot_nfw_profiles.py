"""Create manuscript figures from saved fixed-scale NFW profile-grid results."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROFILE_ROLES = (
    "closed_profile_typical_scale",
    "upper_open_lowest_training_chi_squared",
    "lower_open_lowest_training_chi_squared",
)


def _matching_input_run(profile_run_dir, statistics_dir):
    """Require the statistics folder to belong to the supplied profile run."""
    statistics_manifest = json.loads(
        (Path(statistics_dir) / "manifest.json").read_text(encoding="utf-8"),
    )
    expected = str(Path(profile_run_dir).resolve())
    if statistics_manifest.get("input_run") != expected:
        raise ValueError("Profile statistics describe a different profile-grid run.")


def representative_case_ids(representatives):
    """Return deterministic figure cases in the documented role order."""
    values = representatives.set_index("role")["galaxy_id"]
    return [(role, values[role]) for role in PROFILE_ROLES if role in values.index]


def plot_nfw_profile_figures(profile_run_dir, statistics_dir, output_dir=None):
    """Plot profile-status counts and deterministic training/forecast cases."""
    profile_run_dir = Path(profile_run_dir)
    statistics_dir = Path(statistics_dir)
    _matching_input_run(profile_run_dir, statistics_dir)
    grid_path = profile_run_dir / "profile_grid.csv"
    status_path = statistics_dir / "profile_status_summary.csv"
    representatives_path = statistics_dir / "representative_profiles.csv"
    grid = pd.read_csv(grid_path)
    status = pd.read_csv(status_path)
    representatives = pd.read_csv(representatives_path)
    cases = representative_case_ids(representatives)
    if not cases:
        raise ValueError("No deterministic representative profile cases were found.")

    root = Path(output_dir) if output_dir else profile_run_dir.parent / "figures"
    figure_dir = root / datetime.now(timezone.utc).strftime("profiles_%Y%m%dT%H%M%S_%fZ")
    figure_dir.mkdir(parents=True, exist_ok=False)

    order = ("closed", "open_upper", "open_lower", "open_both")
    colors = {"closed": "#1b9e77", "open_upper": "#d95f02", "open_lower": "#7570b3", "open_both": "#e7298a"}
    values = status.set_index("profile_status")["galaxies"]
    labels = [label.replace("_", " ") for label in order if label in values.index]
    counts = [values[label] for label in order if label in values.index]
    bar_colors = [colors[label] for label in order if label in values.index]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(labels, counts, color=bar_colors)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 0.5, str(count), ha="center")
    ax.set_ylim(0, max(counts) + 6)
    ax.set_ylabel("Number of galaxies")
    ax.set_title("Nominal 95% NFW training-profile status")
    ax.text(
        0.5, -0.20,
        "Open means the nominal $\\Delta\\chi^2$ profile interval reaches a pre-specified grid edge.",
        ha="center", va="top", transform=ax.transAxes, fontsize=9,
    )
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.25, top=0.88)
    status_png = figure_dir / "nfw_profile_status.png"
    status_pdf = figure_dir / "nfw_profile_status.pdf"
    fig.savefig(status_png, dpi=300)
    fig.savefig(status_pdf)
    plt.close(fig)

    fig, axes = plt.subplots(len(cases), 2, figsize=(10.2, 3.25 * len(cases)), squeeze=False)
    role_labels = {
        "closed_profile_typical_scale": "Typical closed profile",
        "upper_open_lowest_training_chi_squared": "Upper-open profile",
        "lower_open_lowest_training_chi_squared": "Lower-open profile",
    }
    for row, (role, galaxy_id) in enumerate(cases):
        profile = grid.loc[
            grid["galaxy_id"].eq(galaxy_id) & grid["fit_error"].fillna("").eq("")
        ].sort_values("fixed_scale_radius_kpc")
        left, right = axes[row]
        left.plot(profile["fixed_scale_radius_kpc"], profile["delta_training_chi_squared"], "o-", color="#333333")
        left.axhline(1.0, color="#1f77b4", linestyle="--", linewidth=1, label="$\\Delta\\chi^2=1.00$")
        left.axhline(3.84, color="#d55e00", linestyle="--", linewidth=1, label="$\\Delta\\chi^2=3.84$")
        left.set_xscale("log")
        left.set_ylim(bottom=0)
        left.set_xlabel("Fixed NFW scale radius $r_s$ (kpc)")
        left.set_ylabel("Inner-training $\\Delta\\chi^2$")
        left.set_title(f"{galaxy_id}: {role_labels[role]}")
        left.grid(True, which="both", alpha=0.25)
        left.legend(frameon=False, fontsize=8)

        for region, color, label in (
            ("middle", "#1f77b4", "Middle forecast"),
            ("outer", "#d55e00", "Far-outer forecast"),
        ):
            score = pd.to_numeric(profile[f"{region}_chi_squared_per_point"], errors="coerce")
            valid = score.gt(0) & np.isfinite(score)
            right.plot(
                profile.loc[valid, "fixed_scale_radius_kpc"], score.loc[valid], "o-",
                color=color, label=label,
            )
        right.set_xscale("log")
        right.set_yscale("log")
        right.set_xlabel("Fixed NFW scale radius $r_s$ (kpc)")
        right.set_ylabel("Forecast $\\chi^2$ per point")
        right.set_title("Forecast diagnostic; not used to select $r_s$")
        right.grid(True, which="both", alpha=0.25)
        right.legend(frameon=False, fontsize=8)
    fig.tight_layout(pad=1.2)
    cases_png = figure_dir / "nfw_profile_cases.png"
    cases_pdf = figure_dir / "nfw_profile_cases.pdf"
    fig.savefig(cases_png, dpi=300)
    fig.savefig(cases_pdf)
    plt.close(fig)

    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_profile_run": str(profile_run_dir.resolve()),
        "statistics_directory": str(statistics_dir.resolve()),
        "profile_grid_sha256": sha256(grid_path.read_bytes()).hexdigest(),
        "profile_status_summary_sha256": sha256(status_path.read_bytes()).hexdigest(),
        "representative_profiles_sha256": sha256(representatives_path.read_bytes()).hexdigest(),
        "representative_cases": [{"role": role, "galaxy_id": galaxy_id} for role, galaxy_id in cases],
        "figures": [path.name for path in (status_png, status_pdf, cases_png, cases_pdf)],
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (figure_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Saved NFW profile figures under {figure_dir}")
    return figure_dir

