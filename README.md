# SPARC Radial Prediction Study

A reproducible Python study of NFW scale-radius identifiability and radial
rotation-curve prediction using public SPARC data.

### Overview

The repository retains a general RAR and rotation-curve workflow, then extends it
to the paper-specific question: when an NFW halo is fitted only to a galaxy's
inner rotation curve, does that curve constrain its scale radius, and how stable
is the result under numerical and stellar mass-to-light assumptions? The
rotation-curve workflow compares baryons plus a Burkert halo with baryons plus an
NFW halo using the same measured velocities and quoted uncertainties.

The rotation curves and baryonic components come from the [SPARC Newtonian mass-model table](https://astroweb.case.edu/SPARC/MassModels_Lelli2016c.mrt).

### Features

* Convert SPARC RTF reports to text and CSV formats (report_converter.py)

* Preprocess SPARC dataset and compute observed and baryonic accelerations (data_preprocess.py)

* Apply RAR quality cuts, fit the acceleration scale, and generate an RAR summary (rar_analysis.py)

* Plot rotation curves for individual galaxies (plot_rotation_curve.py)

* Plot the full Radial Acceleration Relation (plot_rar.py)

* Fit baryons plus a Burkert or NFW halo and flag fitted parameters near their bounds (`halo_models.py`, `burkert_halo_fit.py`).

* Compare both models on the same galaxy with a rotation-curve plot and a residual panel (`compare_halo_fits.py`).

* Verify the halo equations by numerical integration and test parameter recovery on synthetic rotation curves (`tests/`).


## Installation

* Clone the repository: git clone https://github.com/mind-fool/SPARC-Radial-Prediction-Study.git
 and navigate to the project folder.

* Create and activate a Python virtual environment: python3 -m venv ven and then source ven/bin/activate (Linux/macOS) or ven\Scripts\activate (Windows).

* Install the required packages: pip install -r requirements.txt.

## Usage

Run `python3 main.py` from the project folder, or open `main.py` and press Run.
The menu offers the RAR analysis, selected-galaxy rotation curves and halo fits,
the NFW radius-limit sweep, the stellar mass-to-light sweep, and an optimizer
starting-point check. Choose **5**
to run all analyses. The pipeline prepares the input CSV if needed and saves
the results under `outputs/`.

The RAR plot, fitted `g0`, and RAR summary use the same relative-error cut.
The halo fits use the preprocessed rotation curves before that cut.
The two sweeps save the CSVs and manuscript outputs in `outputs/sweeps/`.

* Option **3** saves `nfw_bound_sweep.csv`, `nfw_bound_sweep_table_rows.tex`,
  and `ngc0100_nfw_bound_sweep.png`.
* Option **4** saves `mass_to_light_sweep.csv` and `mass_to_light_sweep.png`.
* Option **5** generates these five files and runs the optimizer check.
* Option **6** runs only the optimizer check, leaving existing sweep outputs untouched.
* Option **7** runs the separate full-sample halo-bound sensitivity study. It is not
  included in option **5**, because it deliberately performs many more fits.
* Option **8** fits only each galaxy's inner radii and predicts its outer measured
  radii. Press Enter to run every eligible galaxy, or enter a comma-separated list
  for a short verification run.
* Option **9** creates a separate galaxy-level statistical summary for a completed
  outer-radius prediction run. It uses the latest completed run by default.
* Option **10** compares two or more completed prediction horizons using only the
  galaxies eligible at every selected horizon.
* Option **11** uses one fixed inner fit to forecast both a middle and a far-outer
  region. It is separate from option **5** because it is a paper-specific study.
* Options **12–14** summarize, test numerical-bound stability for, and generate
  figures from a completed anchored forecast without rerunning its fits.
* Option **15** profiles NFW scale-radius identifiability from the same fixed inner
  data; options **16–17** summarize and plot a completed profile grid.
* Option **18** repeats the fixed-radius NFW profile study under three
  pre-specified stellar mass-to-light settings while retaining identical shared
  observations, galaxies, radial splits, radius grid, and density bounds.
* Option **19** summarizes one completed option-18 run, including profile-status
  counts and every matched baseline-to-alternative classification change.

### Compare Burkert and NFW

From the repository folder with the virtual environment activated, generate
the CSV from the bundled SPARC data and run a benchmark.

```bash
python3 report_converter.py
python3 compare_halo_fits.py NGC0100
```

Generate the CSV once per checkout. Run `python3 compare_halo_fits.py` to enter
a galaxy ID interactively. The script saves the figure as
`<galaxy_id>_halo_comparison.png`.

#### Change the NFW scale-radius limit

The default upper limit on the fitted scale radius is 100 kpc. Use `--max-scale-radius-kpc` to set a different limit for a single run.

```bash
# Fit NFW with the default 100 kpc limit
python3 burkert_halo_fit.py NGC0100 --model nfw

# Repeat with a 300 kpc limit
python3 burkert_halo_fit.py NGC0100 --model nfw --max-scale-radius-kpc 300
```

Running `compare_halo_fits.py` from the editor prompts for a galaxy ID and uses the default limit for both fits. Run `python3 burkert_halo_fit.py --help` to see the individual fitter's options.

### Selecting halo fits

Galaxy plots and both halo sweeps select the lowest total chi-squared among the
ten starting-point trials described below. The same selected fit supplies the
reported parameters, model curve, residuals, and comparison score. Plots do not
rerun the optimizer. Individual halo scripts use this selection too.

The sweep CSVs record the selection strategy, the chosen trial, successful and
failed trial counts, and the number agreeing with the lowest score. The
`burkert_selection_trials_json` and `nfw_selection_trials_json` columns retain
all trial diagnostics as JSON records, including failures and warnings.
The existing bound flags describe the selected trial; they do not establish
that a parameter is physically well constrained.

The validation workflow still compares the original single-start result with
the best among the tested starts. Option **5** runs both the production sweeps
and this separate validation, so it repeats fits and takes longer. Use options
**3** or **4** to regenerate just the corresponding sweep.

### Check optimizer starting points

Run `main.py` and choose **6**. Enter galaxy IDs separated by commas, or press
Enter to check all galaxies. For a short first run, enter `NGC0100,UGC01281,NGC6195`.
Option **5** also runs this check for all galaxies after the other analyses.

Each fit is repeated from ten starting points: the existing default and nine
combinations of density and radius. The nine starts use the 10%, 50%, and 90%
positions within each logarithmic parameter interval. The density interval is
`1e-5` to `10 Msun/pc^3`; the radius interval is `0.01 kpc` to the selected ceiling.
The check covers Burkert at 100 kpc and NFW at 100, 300, and 1,000 kpc with the
baseline stellar ratios, plus both models at 100 kpc for the two other stellar
settings. This gives 14,000 trial fits for 175 galaxies and may take several minutes.

Each run creates a new dated folder under `outputs/validation/` containing:

* `optimizer_trials.csv`: every requested and effective starting point, fitted
  parameters, chi-squared, optimizer status, boundary flags, warnings, and failures.
* `optimizer_summary.csv`: the default and best-found fit for each case, trial
  counts, agreement counts, and the parameter ranges among agreeing fits.
* `optimizer_comparisons.csv`: the default and best-found Burkert/NFW rankings.
* `manifest.json`: the input and source hashes, code revision, software versions,
  starting points, settings, tolerances, and whether the run completed.

Successful trials agree with the best-found fit when their total chi-squared
values differ by no more than `1e-6 + 1e-6 * abs(best_chi_squared)`. This is a
numerical tolerance, not a statistical significance threshold. Model ranking
uses the same reduced-chi-squared near-tie rule as the existing sweeps.
A failed default fit is recorded separately from an improved fit. A run interrupted
between configurations can leave partial CSVs; its manifest will still say `running`.

Signed baryonic contributions can require moving a starting point into the region
where total squared speed is positive. Both the requested and effective starts are
saved, and the summary counts distinct effective starts among successful trials.
Agreement across these starts does not prove that the global minimum was found.
The check does not automatically replace the original sweep results.

### Test halo-fit boundaries

Run `main.py` and choose **7** to test the numerical boundaries without changing
the original paper sweep. The study changes one bound family at a time, while the
other limits retain their baseline values:

* scale-radius ceilings of 10, 30, 100, 300, 1,000, 3,000, and 10,000 kpc;
* density floors of `1e-8`, `1e-7`, `1e-6`, `1e-5`, and `1e-4 Msun/pc^3`;
* density ceilings of 10, 30, 100, 300, and `1e3 Msun/pc^3`;
* scale-radius floors of 0.001, 0.01, and 0.1 kpc.

Duplicate baseline configurations are run once, giving 17 configurations in total.
Both halo models use the same bound configuration in each matched comparison.
The study uses the same ten-start selection as the rest of the pipeline. It saves
a dated folder under `outputs/boundary_sensitivity/` containing every fit,
Burkert--NFW comparisons, configuration summaries, and a manifest with the source
revision, input hash, software versions, and exact configurations. This is a
numerical-sensitivity study: it does not establish a global optimum or identify a
physically preferred model on its own.

### Predict held-out outer radii

Option **8** is the paper-specific prediction workflow. For every eligible galaxy,
the innermost 70% of measured radii are used for fitting and the outermost 30% are
held out for scoring. A galaxy must provide at least eight training points and four
held-out points, yielding 107 eligible SPARC galaxies. The held-out baryonic
decomposition remains available as a model input, but held-out velocities are never
provided to the fitter.

Each of the 17 pre-specified numerical-bound configurations is evaluated separately
for Burkert and NFW. The workflow does not select a configuration using held-out
velocities. It saves every inner fit, every outer prediction and residual, matched
Burkert--NFW prediction comparisons, summaries, and a dated manifest under
`outputs/radial_prediction/`. The held-out score is chi-squared per held-out point;
it is not reduced chi-squared because no parameters are fit to those points.

The menu accepts another fraction when needed for a pre-specified radial-horizon
replication. Every run stores that exact fraction and its resulting eligible-galaxy
set in the manifest; do not compare runs with different fractions as though they
used identical samples.

Option **9** resamples galaxies, not individual velocity points, when computing
the paired-score bootstrap intervals. It also writes paired score summaries,
training-to-prediction preference transitions, and selected-bound-hit summaries
to a separate dated folder under `outputs/radial_prediction/statistics/`. These
are descriptive diagnostics, not a substitute for modelling correlated velocity
errors or for correcting tests across the multiple bound configurations.

Option **10** writes common-sample paired-score summaries and per-galaxy winning
model transitions across the selected horizons. It retains the exact input folders,
input manifests, shared galaxy IDs, bootstrap settings, and its limitations in a
new dated `outputs/radial_prediction/horizon_statistics/` folder.

### Forecast middle and far-outer radii from one fixed inner fit

Option **11** is the stricter anchored forecast. It splits each eligible curve by
sorted measured radius into an inner 60% training region, a middle 20% forecast
region, and a far-outer 20% forecast region. The fit uses only the inner region,
then the exact same fitted parameters are used to predict both later regions. The
model is not refit after the inner region.

A galaxy must have at least 20 usable measured radii: 12 inner, four middle, and
four far-outer points. The current SPARC input has 60 eligible galaxies. As in the
other prediction workflow, the baryonic decomposition at the forecast radii remains
available as model input, but their observed velocities are withheld from fitting.

The runner evaluates Burkert and NFW separately under each of the 17 pre-specified
numerical-bound configurations. It saves a dated folder in
`outputs/anchored_forecast/` containing:

* `fit_forecasts.csv`: the one inner-region fit per galaxy, model, and configuration,
  followed by its middle and far-outer prediction scores;
* `forecast_points.csv`: every forecast velocity, residual, and uncertainty-normalized
  residual, labelled as middle or outer;
* `model_comparisons.csv`: matched Burkert--NFW prediction comparisons within each
  configuration and region;
* `forecast_summary.csv`: configuration-level prediction and failure summaries;
* `manifest.json`: the input hash, source hashes, software versions, exact split,
  numerical configurations, and completion status.

This design separates two questions that a changing holdout fraction confounds:
whether a model can extrapolate farther from a fixed inner fit, and whether it has
more or fewer inner points for fitting. The middle and far-outer points still come
from the same published rotation curve, so they are not independent surveys.

### Profile NFW scale-radius identifiability

Option **15** tests whether the inner rotation curve actually constrains an NFW
scale radius. It uses the same 60/20/20 anchored split and the same 60 eligible
galaxies as option **11**, but holds the NFW scale radius fixed at 41 logarithmically
spaced values from 0.01 to 10,000 kpc. At each fixed radius, it refits only the NFW
density normalization using the inner 60% of points. The middle and far-outer
velocities are then scored without being used to select a radius.

The grid endpoints match the minimum radius and the largest radius ceiling in the
pre-specified boundary study. Each profile row records its inner-training
chi-squared, density fit, and middle/far-outer prediction scores. The profile grid
is deterministic: it searches the allowed log-density interval and refines the best
grid interval. Negative baryonic squared-speed contributions are treated with the
same positive-total-speed requirement as the main halo fitter.

For each galaxy, `profile_summary.csv` records the scale radius at minimum training
chi-squared plus the grid span satisfying `Delta chi-squared <= 1.00` and
`Delta chi-squared <= 3.84`. These are labelled *nominal* one-parameter profile
intervals. They are diagnostics under the stated fitting assumptions, not physical
confidence intervals: correlated rotation-curve errors and astrophysical systematic
uncertainties are not modeled.

Option **16** writes a separate dated folder under `outputs/nfw_profiles/statistics/`
containing the profile-status counts, training-selected profile rows, deterministic
representative cases, and any infeasible profile points. A profile is called
`closed`, `open_upper`, `open_lower`, or `open_both` according to whether its
nominal 95% profile interval reaches one of the pre-specified grid edges.

Option **17** generates PNG and PDF figures only from matching completed profile
and summary folders. It writes a profile-status count figure and profile/forecast
diagnostics for deterministic representative cases under `outputs/nfw_profiles/figures/`.
The forecast panels are explicitly diagnostic; held-out scores are never used to
choose a scale radius.

### Audit the NFW density floor

Option **23** repeats the fixed-radius inner-training profiles for the same eligible
galaxies and 41 radii, changing only the NFW density lower bound through
`1e-5`, `1e-6`, `1e-7`, `1e-8`, and `1e-9` solar masses per cubic parsec. The
upper density bound remains 10. The sweep is an exploratory numerical-bound
check prompted by fits that hit the original `1e-5` floor. It writes a dated
folder under `outputs/nfw_density_floor_sweep/` with the complete profile grid,
profile status for every galaxy and floor, status counts, changes from the
original floor, a PNG/PDF figure generated from those CSVs, and a manifest
recording settings and source hashes. A failed
fit at either radius-grid endpoint is flagged separately; the status labels
follow the original profile rule for a direct comparison. No held-out
observations enter these fits.

The original option-16 closed/open counts are conditional on its `1e-5` density
floor. Use the option-23 output when interpreting whether a nominal profile
closure survives lowering that numerical floor.

### Test profile identifiability under stellar mass-to-light assumptions

Option **18** tests whether the nominal NFW profile classification depends on the
adopted stellar mass-to-light ratios. It repeats the option-15 profile calculation
on the exact same 60 anchored-split galaxies and shared ID--radius observations.
The fixed NFW radius grid (0.01 to 10,000 kpc), density bounds, 60/20/20 radial
split, and nominal profile thresholds remain unchanged. Only the stellar settings
change: baseline disk/bulge values of 0.5/0.7, a higher-disk case of 0.6/0.7, and
a lower-bulge case of 0.5/0.5.

Option **19** writes profile-status counts, matched galaxy-level changes, and
baseline-to-alternative status transitions under
`outputs/nfw_profile_mass_to_light/statistics/`. For the completed 2026-11-14
60-galaxy run, the baseline contained 54 closed, five upper-open, and one
lower-open nominal profiles. The higher-disk setting changed one classification;
the lower-bulge setting changed three. Thus 59 of 60 and 57 of 60 nominal profile
classifications, respectively, remained unchanged under these two alternative
settings. This is a controlled sensitivity result, not a statement that every
best-fit scale radius is physically robust or that the nominal intervals are
physical confidence intervals.

### Tests

Run the full test suite from the repository folder.

```bash
python3 -m unittest discover -s tests -v
```

### Future Work
* Validate a conformal-gravity implementation against a reference rotation curve, then compare it with the halo fits using the same SPARC measurements.

### License

*  This project is released under the MIT License.
