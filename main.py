"""Run the SPARC analysis workflows from one entry point."""

from pathlib import Path
import json

from burkert_halo_fit import DEFAULT_MAX_SCALE_RADIUS_KPC
from halo_fit_selection import fit_best_halo
from compare_halo_fits import plot_halo_comparison
from data_preprocess import load_sparc_dataset
from plot_rotation_curve import plot_rotation_curve
from rar_analysis import filter_data, fit_rar, generate_report
from report_converter import convert_rtf_to_text, convert_text_to_csv
from run_mass_to_light_sweep import main as run_mass_to_light_sweep
from run_nfw_bound_sweep import main as run_nfw_bound_sweep
from run_boundary_sensitivity import run_boundary_sensitivity
from run_radial_prediction import run_radial_prediction
from run_anchored_forecast import run_anchored_forecast
from anchored_forecast_statistics import analyze_anchored_forecast
from anchored_configuration_stability import analyze_configuration_stability
from plot_anchored_stability import plot_radius_ceiling_stability
from run_nfw_profile_grid import run_nfw_profile_grid
from run_nfw_profile_mass_to_light import run_nfw_profile_mass_to_light
from run_nfw_profile_normalization_grid import run_nfw_profile_normalization_grid
from nfw_profile_normalization_grid_statistics import analyze_nfw_profile_normalization_grid
from nfw_profile_mass_to_light_statistics import analyze_nfw_profile_mass_to_light
from nfw_profile_statistics import analyze_nfw_profile_grid
from plot_nfw_profiles import plot_nfw_profile_figures
from prediction_statistics import analyze_prediction_run
from horizon_statistics import analyze_horizons
from plot_rar import plot_rar
from run_optimizer_validation import run_validation


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "GALAXY_CSVDATA.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def completed_prediction_runs():
    """Return completed, self-contained outer-radius prediction result folders."""
    root = OUTPUT_DIR / "radial_prediction"
    runs = []
    for candidate in root.glob("outer_holdout_*"):
        manifest_path = candidate / "manifest.json"
        if not (manifest_path.is_file() and (candidate / "fit_predictions.csv").is_file()):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if manifest.get("status") == "complete":
            runs.append(candidate)
    return sorted(runs, reverse=True)


def completed_anchored_forecast_runs():
    """Return completed, self-contained fixed-training forecast result folders."""
    root = OUTPUT_DIR / "anchored_forecast"
    runs = []
    for candidate in root.glob("inner60_*"):
        manifest_path = candidate / "manifest.json"
        if not (manifest_path.is_file() and (candidate / "fit_forecasts.csv").is_file()):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if manifest.get("status") == "complete":
            runs.append(candidate)
    return sorted(runs, reverse=True)


def latest_matching_anchored_summary(run_dir, kind):
    """Find the latest complete summary folder that belongs to one forecast run."""
    roots = {
        "statistics": OUTPUT_DIR / "anchored_forecast" / "statistics",
        "stability": OUTPUT_DIR / "anchored_forecast" / "configuration_stability",
    }
    prefixes = {"statistics": "summary_", "stability": "stability_"}
    if kind not in roots:
        raise ValueError(f"Unknown anchored-summary kind: {kind!r}")
    expected_input = str(Path(run_dir).resolve())
    matches = []
    for candidate in roots[kind].glob(f"{prefixes[kind]}*"):
        manifest_path = candidate / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if manifest.get("status") == "complete" and manifest.get("input_run") == expected_input:
            matches.append(candidate)
    return max(matches, default=None)


def completed_nfw_profile_runs():
    """Return completed fixed-scale NFW profile-grid result folders."""
    root = OUTPUT_DIR / "nfw_profiles"
    runs = []
    for candidate in root.glob("inner60_*"):
        manifest_path = candidate / "manifest.json"
        if not (manifest_path.is_file() and (candidate / "profile_grid.csv").is_file()):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if manifest.get("status") == "complete":
            runs.append(candidate)
    return sorted(runs, reverse=True)


def completed_nfw_profile_mass_to_light_runs():
    """Return completed NFW profile runs across stellar mass-to-light settings."""
    root = OUTPUT_DIR / "nfw_profile_mass_to_light"
    runs = []
    for candidate in root.glob("inner60_*"):
        manifest_path = candidate / "manifest.json"
        if not (manifest_path.is_file() and (candidate / "profile_identifiability.csv").is_file()):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if manifest.get("status") == "complete":
            runs.append(candidate)
    return sorted(runs, reverse=True)


def completed_nfw_profile_normalization_grid_runs():
    """Return completed NFW profile runs across the tied normalization grid."""
    root = OUTPUT_DIR / "nfw_profile_normalization_grid"
    runs = []
    for candidate in root.glob("inner60_*"):
        manifest_path = candidate / "manifest.json"
        if not (manifest_path.is_file() and (candidate / "profile_identifiability.csv").is_file()):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if manifest.get("status") == "complete":
            runs.append(candidate)
    return sorted(runs, reverse=True)


def latest_matching_nfw_profile_statistics(run_dir):
    """Return the latest complete profile summary for one profile-grid run."""
    expected_input = str(Path(run_dir).resolve())
    matches = []
    for candidate in (OUTPUT_DIR / "nfw_profiles" / "statistics").glob("summary_*"):
        manifest_path = candidate / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if manifest.get("status") == "complete" and manifest.get("input_run") == expected_input:
            matches.append(candidate)
    return max(matches, default=None)


def prepare_input():
    """Generate the input CSV on a fresh checkout."""
    if DATA_PATH.is_file():
        return
    text_path = PROJECT_DIR / "Galaxy_textfile.txt"
    if not text_path.is_file():
        result = convert_rtf_to_text(PROJECT_DIR / "Galaxy_textfile.rtf", text_path)
        if result is None:
            raise SystemExit("Could not convert the bundled SPARC report.")
    convert_text_to_csv(text_path, DATA_PATH)


def run_rar_workflow(data):
    """Fit, plot and summarize the same filtered acceleration data."""
    filtered = filter_data(df=data, output_path=OUTPUT_DIR / "filtered_data.csv")
    g0 = fit_rar(filtered)
    plot_rar(filtered, output_path=OUTPUT_DIR / "rar_plots" / "rar_full.png",
             g0=g0, show=False)
    generate_report(df=filtered, g0=g0)


def run_galaxy_workflow(data):
    """Save the baryonic curve, both halo fits and the comparison for each ID."""
    available = {name.casefold(): name for name in data["ID"].unique()}
    requested = input("Enter galaxy IDs separated by commas [NGC0100]: ").strip()
    galaxy_ids = []
    for name in (requested or "NGC0100").split(","):
        name = name.strip()
        if not name:
            continue
        galaxy_id = available.get(name.casefold())
        if galaxy_id is None:
            print(f"Galaxy {name!r} was not found in the preprocessed sample.")
        elif galaxy_id not in galaxy_ids:
            galaxy_ids.append(galaxy_id)
    if not galaxy_ids:
        return

    value = input(f"NFW upper scale radius in kpc [{DEFAULT_MAX_SCALE_RADIUS_KPC:g}]: ").strip()
    try:
        cap = float(value) if value else DEFAULT_MAX_SCALE_RADIUS_KPC
        if not 0.01 < cap < float("inf"):
            raise ValueError
    except ValueError:
        print("Enter a finite NFW upper scale radius greater than 0.01 kpc.")
        return
    suffix = "" if cap == DEFAULT_MAX_SCALE_RADIUS_KPC else f"_cap{cap:g}kpc"
    halo_dir = OUTPUT_DIR / "halo_fits"
    for galaxy_id in galaxy_ids:
        try:
            burkert = fit_best_halo(galaxy_id, df=data, model="burkert", show=False, plot=True, report=True,
                              output_path=halo_dir / f"{galaxy_id}_burkert_fit.png")
            nfw = fit_best_halo(galaxy_id, df=data, model="nfw", show=False, plot=True, report=True,
                          max_scale_radius_kpc=cap,
                          output_path=halo_dir / f"{galaxy_id}_nfw{suffix}_fit.png")
        except (ValueError, RuntimeError) as error:
            print(f"Could not complete halo fits for {galaxy_id}: {error}")
            continue
        plot_rotation_curve(galaxy_id, df=data, show=False,
                            output_path=OUTPUT_DIR / "rotation_curves" / f"{galaxy_id}_rotation_curve.png")
        plot_halo_comparison(galaxy_id, burkert, nfw, show=False,
                             output_path=halo_dir / f"{galaxy_id}_halo_comparison{suffix}.png")


def main():
    print("SPARC analysis")
    print("1. Radial acceleration relation (RAR)")
    print("2. Rotation curves and halo comparisons for selected galaxies")
    print("3. NFW radius-limit sweep (175-galaxy study)")
    print("4. Stellar mass-to-light sweep")
    print("5. Run all analyses")
    print("6. Check optimizer starting-point sensitivity")
    print("7. Test halo-fit boundary sensitivity (full 175-galaxy study)")
    print("8. Predict held-out outer rotation-curve radii")
    print("9. Summarize completed outer-radius predictions")
    print("10. Compare radial prediction horizons on common galaxies")
    print("11. Forecast middle and far-outer radii from one fixed inner fit")
    print("12. Summarize completed anchored forecasts")
    print("13. Test anchored forecasts for numerical-bound stability")
    print("14. Generate anchored forecast stability figures")
    print("15. Profile NFW scale-radius identifiability from anchored inner fits")
    print("16. Summarize completed NFW scale-radius profiles")
    print("17. Generate NFW scale-radius profile figures")
    print("18. Test NFW profile identifiability under stellar mass-to-light assumptions")
    print("19. Summarize completed mass-to-light NFW profile studies")
    print("20. Test NFW profiles across a tied stellar normalization grid")
    print("21. Summarize completed stellar normalization grid studies")
    print("0. Quit")
    choice = input("Choose an analysis [5]: ").strip() or "5"
    if choice == "0":
        return
    if choice not in {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"}:
        raise SystemExit("Choose a number from 0 to 21.")

    prepare_input()
    for folder in ("rotation_curves", "rar_plots", "halo_fits", "sweeps"):
        (OUTPUT_DIR / folder).mkdir(parents=True, exist_ok=True)

    if choice in {"1", "2", "5"}:
        data = load_sparc_dataset(DATA_PATH)
        data.to_csv(OUTPUT_DIR / "preprocessed_data.csv", index=False)
        if choice in {"2", "5"}:
            run_galaxy_workflow(data)
        if choice in {"1", "5"}:
            run_rar_workflow(data)

    if choice in {"3", "5"}:
        run_nfw_bound_sweep(output_path=OUTPUT_DIR / "sweeps" / "nfw_bound_sweep.csv")
    if choice in {"4", "5"}:
        run_mass_to_light_sweep(output_path=OUTPUT_DIR / "sweeps" / "mass_to_light_sweep.csv")

    if choice in {"5", "6"}:
        selected = None
        if choice == "6":
            requested = input("Galaxy IDs separated by commas [all galaxies]: ").strip()
            selected = requested.split(",") if requested else None
        run_validation(output_dir=OUTPUT_DIR / "validation", galaxy_ids=selected,
                       data_path=DATA_PATH)

    if choice == "7":
        run_boundary_sensitivity(
            output_dir=OUTPUT_DIR / "boundary_sensitivity", data_path=DATA_PATH,
        )

    if choice == "8":
        requested = input("Galaxy IDs separated by commas [all eligible galaxies]: ").strip()
        selected = requested.split(",") if requested else None
        fraction_value = input("Outer held-out fraction [0.30]: ").strip()
        try:
            holdout_fraction = float(fraction_value) if fraction_value else 0.30
            if not 0 < holdout_fraction < 1:
                raise ValueError
        except ValueError:
            print("Enter a held-out fraction strictly between zero and one.")
            return
        run_radial_prediction(
            output_dir=OUTPUT_DIR / "radial_prediction", data_path=DATA_PATH,
            galaxy_ids=selected, holdout_fraction=holdout_fraction,
        )

    if choice == "9":
        runs = completed_prediction_runs()
        if not runs:
            print("No completed outer-radius prediction runs were found.")
            return
        for number, run_dir in enumerate(runs, 1):
            print(f"{number}. {run_dir.name}")
        requested = input("Choose a completed run [1]: ").strip()
        try:
            selection = int(requested or "1")
            run_dir = runs[selection - 1]
        except (ValueError, IndexError):
            print(f"Choose a number from 1 to {len(runs)}.")
            return
        analyze_prediction_run(run_dir)

    if choice == "10":
        runs = completed_prediction_runs()
        if len(runs) < 2:
            print("At least two completed outer-radius prediction runs are required.")
            return
        for number, run_dir in enumerate(runs, 1):
            print(f"{number}. {run_dir.name}")
        requested = input("Choose runs separated by commas [all]: ").strip()
        try:
            numbers = [int(value.strip()) for value in requested.split(",") if value.strip()]
            selected = runs if not requested else [runs[number - 1] for number in numbers]
        except (ValueError, IndexError):
            print(f"Choose comma-separated numbers from 1 to {len(runs)}.")
            return
        selected = list(dict.fromkeys(selected))
        if len(selected) < 2:
            print("Choose at least two distinct completed runs.")
            return
        analyze_horizons(selected)

    if choice == "11":
        requested = input("Galaxy IDs separated by commas [all eligible galaxies]: ").strip()
        selected = requested.split(",") if requested else None
        run_anchored_forecast(
            output_dir=OUTPUT_DIR / "anchored_forecast", data_path=DATA_PATH,
            galaxy_ids=selected,
        )

    if choice == "12":
        runs = completed_anchored_forecast_runs()
        if not runs:
            print("No completed anchored forecast runs were found.")
            return
        for number, run_dir in enumerate(runs, 1):
            print(f"{number}. {run_dir.name}")
        requested = input("Choose a completed run [1]: ").strip()
        try:
            selection = int(requested or "1")
            run_dir = runs[selection - 1]
        except (ValueError, IndexError):
            print(f"Choose a number from 1 to {len(runs)}.")
            return
        analyze_anchored_forecast(run_dir)

    if choice == "13":
        runs = completed_anchored_forecast_runs()
        if not runs:
            print("No completed anchored forecast runs were found.")
            return
        for number, run_dir in enumerate(runs, 1):
            print(f"{number}. {run_dir.name}")
        requested = input("Choose a completed run [1]: ").strip()
        try:
            selection = int(requested or "1")
            run_dir = runs[selection - 1]
        except (ValueError, IndexError):
            print(f"Choose a number from 1 to {len(runs)}.")
            return
        analyze_configuration_stability(run_dir)

    if choice == "14":
        runs = completed_anchored_forecast_runs()
        if not runs:
            print("No completed anchored forecast runs were found.")
            return
        for number, run_dir in enumerate(runs, 1):
            print(f"{number}. {run_dir.name}")
        requested = input("Choose a completed run [1]: ").strip()
        try:
            selection = int(requested or "1")
            run_dir = runs[selection - 1]
        except (ValueError, IndexError):
            print(f"Choose a number from 1 to {len(runs)}.")
            return
        statistics_dir = latest_matching_anchored_summary(run_dir, "statistics")
        stability_dir = latest_matching_anchored_summary(run_dir, "stability")
        if statistics_dir is None or stability_dir is None:
            print("Run options 12 and 13 for this forecast before generating figures.")
            return
        plot_radius_ceiling_stability(stability_dir, statistics_dir)

    if choice == "15":
        requested = input("Galaxy IDs separated by commas [all eligible galaxies]: ").strip()
        selected = requested.split(",") if requested else None
        run_nfw_profile_grid(
            output_dir=OUTPUT_DIR / "nfw_profiles", data_path=DATA_PATH,
            galaxy_ids=selected,
        )

    if choice == "16":
        runs = completed_nfw_profile_runs()
        if not runs:
            print("No completed NFW profile-grid runs were found.")
            return
        for number, run_dir in enumerate(runs, 1):
            print(f"{number}. {run_dir.name}")
        requested = input("Choose a completed run [1]: ").strip()
        try:
            selection = int(requested or "1")
            run_dir = runs[selection - 1]
        except (ValueError, IndexError):
            print(f"Choose a number from 1 to {len(runs)}.")
            return
        analyze_nfw_profile_grid(run_dir)

    if choice == "17":
        runs = completed_nfw_profile_runs()
        if not runs:
            print("No completed NFW profile-grid runs were found.")
            return
        for number, run_dir in enumerate(runs, 1):
            print(f"{number}. {run_dir.name}")
        requested = input("Choose a completed run [1]: ").strip()
        try:
            selection = int(requested or "1")
            run_dir = runs[selection - 1]
        except (ValueError, IndexError):
            print(f"Choose a number from 1 to {len(runs)}.")
            return
        statistics_dir = latest_matching_nfw_profile_statistics(run_dir)
        if statistics_dir is None:
            print("Run option 16 for this profile grid before generating figures.")
            return
        plot_nfw_profile_figures(run_dir, statistics_dir)

    if choice == "18":
        requested = input("Galaxy IDs separated by commas [all eligible galaxies]: ").strip()
        selected = requested.split(",") if requested else None
        run_nfw_profile_mass_to_light(
            output_dir=OUTPUT_DIR / "nfw_profile_mass_to_light", data_path=DATA_PATH,
            galaxy_ids=selected,
        )

    if choice == "19":
        runs = completed_nfw_profile_mass_to_light_runs()
        if not runs:
            print("No completed mass-to-light NFW profile runs were found.")
            return
        for number, run_dir in enumerate(runs, 1):
            print(f"{number}. {run_dir.name}")
        requested = input("Choose a completed run [1]: ").strip()
        try:
            selection = int(requested or "1")
            run_dir = runs[selection - 1]
        except (ValueError, IndexError):
            print(f"Choose a number from 1 to {len(runs)}.")
            return
        analyze_nfw_profile_mass_to_light(run_dir)

    if choice == "20":
        requested = input("Galaxy IDs separated by commas [all eligible galaxies]: ").strip()
        selected = requested.split(",") if requested else None
        run_nfw_profile_normalization_grid(
            output_dir=OUTPUT_DIR / "nfw_profile_normalization_grid", data_path=DATA_PATH,
            galaxy_ids=selected,
        )

    if choice == "21":
        runs = completed_nfw_profile_normalization_grid_runs()
        if not runs:
            print("No completed stellar normalization grid runs were found.")
            return
        for number, run_dir in enumerate(runs, 1):
            print(f"{number}. {run_dir.name}")
        requested = input("Choose a completed run [1]: ").strip()
        try:
            selection = int(requested or "1")
            run_dir = runs[selection - 1]
        except (ValueError, IndexError):
            print(f"Choose a number from 1 to {len(runs)}.")
            return
        analyze_nfw_profile_normalization_grid(run_dir)

    print(f"Results saved under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
