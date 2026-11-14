"""Fixed-training radial splits for near-outer and far-outer prediction."""

import numpy as np


TRAINING_FRACTION = 0.60
MIDDLE_FRACTION = 0.20
OUTER_FRACTION = 0.20
MIN_TRAINING_POINTS = 12
MIN_MIDDLE_POINTS = 4
MIN_OUTER_POINTS = 4


def split_inner_middle_outer(
    galaxy, *, training_fraction=TRAINING_FRACTION,
    middle_fraction=MIDDLE_FRACTION, outer_fraction=OUTER_FRACTION,
    min_training_points=MIN_TRAINING_POINTS,
    min_middle_points=MIN_MIDDLE_POINTS,
    min_outer_points=MIN_OUTER_POINTS,
):
    """Split one ordered curve into fixed inner training, middle, and outer sets.

    A model is fitted only to ``training``. The same fixed parameters are then
    scored on both later radial regions, so changes in performance can be
    attributed to prediction distance rather than a changed training sample.
    """
    fractions = np.array([training_fraction, middle_fraction, outer_fraction], dtype=float)
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("Training, middle, and outer fractions must be positive and sum to one.")
    ordered = galaxy.sort_values("R_kpc").copy()
    required = ("R_kpc", "Vobs_kms", "eV_kms", "Vbar_squared_kms2")
    missing = [column for column in required if column not in ordered]
    if missing:
        raise ValueError(f"Galaxy data are missing required columns: {', '.join(missing)}.")
    if not np.all(np.isfinite(ordered.loc[:, required].to_numpy(dtype=float))):
        raise ValueError("Anchored radial splits require finite fit and prediction inputs.")
    if not ordered["R_kpc"].is_unique:
        raise ValueError("Anchored radial splits require unique measured radii within a galaxy.")

    n_total = len(ordered)
    n_training = int(np.floor(n_total * training_fraction))
    n_middle = int(np.floor(n_total * middle_fraction))
    n_outer = n_total - n_training - n_middle
    if (
        n_training < min_training_points
        or n_middle < min_middle_points
        or n_outer < min_outer_points
    ):
        raise ValueError(
            "Need at least 12 inner training, four middle, and four outer points; "
            f"this galaxy has {n_total} measurements."
        )
    training = ordered.iloc[:n_training].copy()
    middle = ordered.iloc[n_training:n_training + n_middle].copy()
    outer = ordered.iloc[n_training + n_middle:].copy()
    if (
        training["R_kpc"].max() >= middle["R_kpc"].min()
        or middle["R_kpc"].max() >= outer["R_kpc"].min()
    ):
        raise ValueError("Training, middle, and outer regions must occupy distinct radii.")
    return training, middle, outer


def eligible_anchored_galaxy_ids(dataset, **split_kwargs):
    """Return IDs satisfying the fixed-training split requirements."""
    eligible = []
    for galaxy_id, galaxy in dataset.groupby("ID", sort=True):
        try:
            split_inner_middle_outer(galaxy, **split_kwargs)
        except ValueError:
            continue
        eligible.append(galaxy_id)
    return eligible
