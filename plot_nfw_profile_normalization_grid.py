"""Generate manuscript figures for the tied stellar-normalization grid study."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def disk_normalization(setting):
    """Extract the disk mass-to-light value from a pre-specified grid setting name."""
    if not isinstance(setting, str) or not setting.startswith("disk_"):
        raise ValueError(f"Expected a setting named like 'disk_0.50', got {setting!r}.")
    try:
        value = float(setting.removeprefix("disk_"))
    except ValueError as error:
        raise ValueError(f"Could not parse disk normalization from {setting!r}.") from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid disk normalization in {setting!r}.")
    return value


def normalization_grid_table(profile_changes, scale_radius_changes):
    """Match the two summaries and sort them by their disk normalization."""
    required_profile = {"setting", "galaxies", "profile_status_changes"}
    required_radius = {
        "setting", "factor_of_ten_or_more", "factor_of_ten_with_unchanged_status",
    }
    missing = required_profile.difference(profile_changes.columns)
    if missing:
        raise ValueError(f"Missing profile-change columns: {sorted(missing)}")
    missing = required_radius.difference(scale_radius_changes.columns)
    if missing:
        raise ValueError(f"Missing scale-radius columns: {sorted(missing)}")
    table = profile_changes.merge(
        scale_radius_changes[
            ["setting", "factor_of_ten_or_more", "factor_of_ten_with_unchanged_status"]
        ],
        on="setting", how="inner", validate="one_to_one",
    ).copy()
    if len(table) != len(profile_changes) or len(table) != len(scale_radius_changes):
        raise ValueError("Profile and scale-radius summaries must contain identical settings.")
    table["disk_mass_to_light"] = table["setting"].map(disk_normalization)
    return table.sort_values("disk_mass_to_light").reset_index(drop=True)


def with_baseline_reference(table, baseline=0.50):
    """Add the zero-change reference point used by both figure panels."""
    if (table["disk_mass_to_light"] == baseline).any():
        raise ValueError("The baseline setting must not appear in the change summaries.")
    reference = pd.DataFrame({
        "setting": [f"disk_{baseline:.2f}"],
        "galaxies": [int(table["galaxies"].iloc[0])],
        "profile_status_changes": [0],
        "factor_of_ten_or_more": [0],
        "factor_of_ten_with_unchanged_status": [0],
        "disk_mass_to_light": [baseline],
    })
    return pd.concat([table, reference], ignore_index=True).sort_values(
        "disk_mass_to_light",
    ).reset_index(drop=True)


def plot_normalization_grid_stability(statistics_dir, output_dir=None):
    """Plot classification and scale-radius sensitivity from one completed report."""
    statistics_dir = Path(statistics_dir)
    profile_path = statistics_dir / "profile_change_summary.csv"
    scale_path = statistics_dir / "scale_radius_change_summary.csv"
    manifest_path = statistics_dir / "manifest.json"
    if not all(path.is_file() for path in (profile_path, scale_path, manifest_path)):
        raise FileNotFoundError(
            "Statistics folder needs profile_change_summary.csv, "
            "scale_radius_change_summary.csv, and manifest.json.",
        )
    table = normalization_grid_table(
        pd.read_csv(profile_path), pd.read_csv(scale_path),
    )
    if table.empty:
        raise ValueError("No normalization-grid summary rows were found.")
    galaxy_counts = table["galaxies"].unique()
    if len(galaxy_counts) != 1:
        raise ValueError("All normalization settings must use the same galaxy count.")
    galaxy_count = int(galaxy_counts[0])
    plot_table = with_baseline_reference(table)

    fig, (classification_ax, radius_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(7.2, 6.2), constrained_layout=True,
    )
    x = plot_table["disk_mass_to_light"]
    classification_ax.plot(
        x, plot_table["profile_status_changes"], marker="o", color="C1",
        label="Profile-status changes",
    )
    classification_ax.axvline(0.50, color="0.45", linestyle="--", linewidth=1,
                              label="Baseline disk $M/L$")
    classification_ax.set_ylabel(f"Galaxies changed (of {galaxy_count})")
    classification_ax.set_title("NFW profile sensitivity to tied stellar normalization")
    classification_ax.legend()
    classification_ax.grid(alpha=0.25)

    radius_ax.plot(
        x, plot_table["factor_of_ten_or_more"], marker="o", color="C0",
        label=r"$|r_s/r_{s,0}| \geq 10$ or $\leq 0.1$",
    )
    radius_ax.plot(
        x, plot_table["factor_of_ten_with_unchanged_status"], marker="s", color="C2",
        label="Large shift; status unchanged",
    )
    radius_ax.axvline(0.50, color="0.45", linestyle="--", linewidth=1)
    radius_ax.set_xlabel(r"Disk stellar mass-to-light ratio $\Upsilon_{\rm disk}$")
    radius_ax.set_ylabel(f"Galaxies (of {galaxy_count})")
    radius_ax.legend()
    radius_ax.grid(alpha=0.25)

    root = Path(output_dir) if output_dir else statistics_dir.parent.parent / "figures"
    run_dir = root / datetime.now(timezone.utc).strftime("stability_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    figure_png = run_dir / "normalization_grid_sensitivity.png"
    figure_pdf = run_dir / "normalization_grid_sensitivity.pdf"
    fig.savefig(figure_png, dpi=300)
    fig.savefig(figure_pdf)
    plt.close(fig)
    plot_table.to_csv(run_dir / "normalization_grid_figure_data.csv", index=False)
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_statistics": str(statistics_dir.resolve()),
        "input_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "input_profile_change_summary_sha256": sha256(profile_path.read_bytes()).hexdigest(),
        "input_scale_radius_change_summary_sha256": sha256(scale_path.read_bytes()).hexdigest(),
        "galaxy_count": galaxy_count,
        "baseline_disk_mass_to_light": 0.50,
        "interpretation": (
            "The upper panel counts changes in nominal 95 percent profile status against "
            "the disk-0.50 baseline. The lower panel counts training-selected NFW scale "
            "radii that differ from baseline by at least a factor of ten. These are "
            "sensitivity diagnostics, not physical confidence intervals."
        ),
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Saved normalization-grid stability figures under {run_dir}")
    return run_dir
