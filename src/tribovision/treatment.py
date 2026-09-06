"""The Tribonema phase: treatment manifests, dose response, and viability linkage.

LIVECell teaches the model where cell boundaries are. It cannot answer the
question this project actually asks, because it contains no treatment,
concentration, exposure time, control, or viability information at all. This
module defines the missing half — the schema an experiment must record, and the
analysis that turns segmented images into a testable claim.

The analysis is deliberately conservative:

* Every comparison is blocked by experiment day, because plates imaged on
  different days differ for reasons that have nothing to do with treatment.
* Morphology is only ever allowed to *predict* viability, never to stand in for
  it, and predictive skill is estimated with leave-one-day-out cross-validation
  plus a label-permutation null.
* Multiple morphology features are tested at once, so p-values are corrected
  with Benjamini-Hochberg and the uncorrected values are kept visible.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize, special, stats

REQUIRED_COLUMNS = (
    "image_path",
    "experiment_day",
    "plate_id",
    "well_id",
    "field",
    "cell_line",
    "treatment",
    "concentration_ug_per_ml",
    "exposure_hours",
    "control_type",
    "replicate",
)
OPTIONAL_COLUMNS = (
    "viability_fraction",
    "mts_absorbance",
    "mts_blank_absorbance",
    "micrometers_per_pixel",
    "operator",
    "permission_status",
    "notes",
)
#: How much confluency each feature reflects, as opposed to cell shape.
#:
#: This distinction is the difference between a result and a tautology. While the
#: segmenter is semantic, touching cells merge into one region, so an object's
#: "area" is really a measure of how crowded the field is. A cytotoxic extract
#: kills cells, cells detach, confluency falls, and "mean object area" falls with
#: it — a beautiful dose response that says only "fewer cells at higher dose",
#: which the viability assay already said. Shape features are scale- and
#: density-invariant and are the ones that can answer the question actually asked.
FEATURE_KINDS: dict[str, str] = {
    "objects": "density",
    "total_area_pixels": "density",
    "area_pixels_mean": "density-contaminated",
    "area_pixels_median": "density-contaminated",
    "equivalent_diameter_pixels_mean": "density-contaminated",
    "circularity_mean": "shape",
    "aspect_ratio_mean": "shape",
    "solidity_mean": "shape",
    "mean_intensity_mean": "intensity",
}
MORPHOLOGY_FEATURES = tuple(FEATURE_KINDS)
SHAPE_FEATURES = tuple(name for name, kind in FEATURE_KINDS.items() if kind == "shape")

#: The single pre-specified confirmatory endpoint. Everything else is exploratory.
#: It is a shape feature deliberately: see FEATURE_KINDS.
PRIMARY_FEATURE = "circularity_mean"


class TreatmentDataError(RuntimeError):
    """Raised when a treatment manifest is missing information the analysis needs."""


@dataclass(frozen=True)
class TreatmentRecord:
    image_path: Path
    experiment_day: str
    plate_id: str
    well_id: str
    field: str
    cell_line: str
    treatment: str
    concentration_ug_per_ml: float
    exposure_hours: float
    control_type: str
    replicate: str
    viability_fraction: float | None = None
    micrometers_per_pixel: float | None = None
    extra: dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)

    @property
    def condition(self) -> tuple[str, float, float]:
        return (self.treatment, self.concentration_ug_per_ml, self.exposure_hours)

    @property
    def biological_unit(self) -> tuple[str, str, str]:
        """The unit that is actually independent: one well on one plate on one day."""
        return (self.experiment_day, self.plate_id, self.well_id)


def _optional_float(value: Any, name: str) -> float | None:
    """Parse a number that a manifest is allowed to leave blank."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TreatmentDataError(f"Column {name!r} must be a number, got {value!r}.") from exc
    if not math.isfinite(number):
        raise TreatmentDataError(f"Column {name!r} must be finite, got {value!r}.")
    return number


def _require_float(value: Any, name: str) -> float:
    """Parse a number the analysis cannot proceed without."""
    number = _optional_float(value, name)
    if number is None:
        raise TreatmentDataError(f"Column {name!r} is required and must be a number.")
    return number


def load_treatment_manifest(path: Path, *, root: Path | None = None) -> list[TreatmentRecord]:
    """Read a CSV or JSONL treatment manifest and validate it strictly."""
    path = Path(path).resolve()
    if not path.is_file():
        raise TreatmentDataError(f"Treatment manifest does not exist: {path}")
    base = Path(root).resolve() if root else path.parent

    if path.suffix.casefold() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not rows:
        raise TreatmentDataError(f"Treatment manifest {path.name} contains no rows.")

    missing = [name for name in REQUIRED_COLUMNS if name not in rows[0]]
    if missing:
        raise TreatmentDataError(
            f"Treatment manifest is missing required column(s): {', '.join(missing)}. "
            f"Required columns are: {', '.join(REQUIRED_COLUMNS)}."
        )

    records: list[TreatmentRecord] = []
    for number, row in enumerate(rows, start=1):
        image_value = str(row["image_path"]).strip()
        if not image_value:
            raise TreatmentDataError(f"Row {number}: image_path is empty.")
        candidate = Path(image_value)
        image_path = candidate if candidate.is_absolute() else (base / candidate)
        image_path = image_path.resolve()
        contained = base.resolve() in image_path.parents or image_path == base.resolve()
        if not contained and not candidate.is_absolute():
            raise TreatmentDataError(
                f"Row {number}: image_path {image_value!r} escapes the manifest directory."
            )
        viability = _optional_float(row.get("viability_fraction"), "viability_fraction")
        if viability is None and row.get("mts_absorbance") not in (None, ""):
            absorbance = _require_float(row.get("mts_absorbance"), "mts_absorbance")
            blank = _optional_float(row.get("mts_blank_absorbance"), "mts_blank_absorbance")
            viability = absorbance - (blank or 0.0)
        records.append(
            TreatmentRecord(
                image_path=image_path,
                experiment_day=str(row["experiment_day"]).strip(),
                plate_id=str(row["plate_id"]).strip(),
                well_id=str(row["well_id"]).strip(),
                field=str(row["field"]).strip(),
                cell_line=str(row["cell_line"]).strip(),
                treatment=str(row["treatment"]).strip(),
                concentration_ug_per_ml=_require_float(
                    row["concentration_ug_per_ml"], "concentration_ug_per_ml"
                ),
                exposure_hours=_require_float(row["exposure_hours"], "exposure_hours"),
                control_type=str(row["control_type"]).strip(),
                replicate=str(row["replicate"]).strip(),
                viability_fraction=viability,
                micrometers_per_pixel=_optional_float(
                    row.get("micrometers_per_pixel"), "micrometers_per_pixel"
                ),
                extra={key: row[key] for key in OPTIONAL_COLUMNS if key in row},
            )
        )

    _check_design(records)
    return records


def _check_design(records: list[TreatmentRecord]) -> None:
    """Refuse designs that cannot support the claim the analysis would make."""
    days = {record.experiment_day for record in records}
    concentrations = {record.concentration_ug_per_ml for record in records}
    controls = [record for record in records if record.concentration_ug_per_ml == 0]
    problems: list[str] = []
    if not controls:
        problems.append("no zero-concentration control images")
    if len(concentrations) < 2:
        problems.append("only one concentration, so no dose response can be estimated")
    wells_per_condition: dict[tuple[Any, ...], set[tuple[str, str, str]]] = defaultdict(set)
    for record in records:
        wells_per_condition[record.condition].add(record.biological_unit)
    thin = [key for key, wells in wells_per_condition.items() if len(wells) < 2]
    if thin:
        problems.append(
            f"{len(thin)} condition(s) have fewer than two independent wells "
            f"(e.g. {thin[0]}), so within-condition variability is unestimated"
        )
    if len(days) < 2:
        problems.append(
            "only one experiment day, so nothing can be validated on a day the model "
            "did not see, and a day effect cannot be separated from a treatment effect"
        )
    vehicle = [
        record for record in controls if record.control_type.casefold() in {"vehicle", "solvent"}
    ]
    if controls and not vehicle:
        problems.append(
            "no well is marked control_type='vehicle'. A zero-concentration well is not "
            "a vehicle control unless it received the same solvent at the same final "
            "concentration as the treated wells"
        )
    if problems:
        raise TreatmentDataError(
            "This experimental design cannot support a dose-response claim: "
            + "; ".join(problems)
            + f". Days present: {sorted(days)}."
        )


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return BH-adjusted q-values in the original order.

    Non-finite entries are carried through untouched rather than corrected. A
    single NaN would otherwise propagate through the running minimum and quietly
    turn every q-value in the family into NaN.
    """
    values = np.asarray(p_values, dtype=float)
    if values.size == 0:
        return []
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    n = int(finite.sum())
    if n == 0:
        return [float(value) for value in adjusted]
    subset = values[finite]
    order = np.argsort(subset)
    ranked = subset[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    corrected = np.empty(n, dtype=float)
    corrected[order] = np.clip(ranked, 0.0, 1.0)
    adjusted[finite] = corrected
    return [float(value) for value in adjusted]


def four_parameter_logistic(
    concentration: np.ndarray, bottom: float, top: float, ic50: float, hill: float
) -> np.ndarray:
    """Standard 4PL dose-response curve, fitted on log10 concentration.

    Written through ``expit`` rather than ``1 / (1 + 10**x)``. The direct form
    overflows for the extreme parameter values an optimiser tries on its way to a
    fit, which turns a normal search step into a numerical warning.
    """
    exponent = (math.log10(max(ic50, 1e-12)) - np.asarray(concentration, dtype=float)) * hill
    return bottom + (top - bottom) * special.expit(-exponent * math.log(10.0))


def fit_dose_response(concentrations: np.ndarray, responses: np.ndarray) -> dict[str, Any]:
    """Fit a 4PL curve and report IC50 with a bootstrap confidence interval."""
    positive = concentrations > 0
    distinct = len(np.unique(concentrations[positive]))
    # Four parameters fitted to four points has zero residual degrees of freedom,
    # and three concentrations cannot constrain bottom, top, IC50 and slope at
    # once. The earlier threshold would happily hand back an "IC50" from 4 points.
    if distinct < 5 or positive.sum() < 8:
        return {
            "fitted": False,
            "reason": (
                "A 4PL fit needs at least 5 distinct non-zero concentrations across at "
                f"least 8 wells to constrain its four parameters; this design has "
                f"{distinct} concentration(s) across {int(positive.sum())} well(s)."
            ),
        }
    log_concentration = np.log10(concentrations[positive])
    values = responses[positive]
    guess = [
        float(values.min()),
        float(values.max()),
        float(np.median(concentrations[positive])),
        1.0,
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", optimize.OptimizeWarning)
        try:
            parameters, _ = optimize.curve_fit(
                four_parameter_logistic, log_concentration, values, p0=guess, maxfev=20000
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            return {"fitted": False, "reason": f"Curve fit did not converge: {exc}"}
    # An unestimable covariance means the data barely constrain the parameters.
    # That is a fact about the experiment, so it is reported rather than hidden.
    covariance_estimated = not any(
        issubclass(entry.category, optimize.OptimizeWarning) for entry in caught
    )

    predicted = four_parameter_logistic(log_concentration, *parameters)
    residual = values - predicted
    total = values - values.mean()
    r_squared = (
        1.0 - float(residual @ residual) / float(total @ total) if total.any() else float("nan")
    )

    rng = np.random.default_rng(0)
    ic50_samples: list[float] = []
    with warnings.catch_warnings():
        # Resamples that fail to converge are discarded below; the optimiser's
        # advisory warnings about them are not findings about this project.
        warnings.simplefilter("ignore", optimize.OptimizeWarning)
        for _ in range(400):
            index = rng.integers(0, len(values), len(values))
            try:
                sample, _ = optimize.curve_fit(
                    four_parameter_logistic,
                    log_concentration[index],
                    values[index],
                    p0=parameters,
                    maxfev=8000,
                )
            except (RuntimeError, ValueError, TypeError):
                continue
            if np.isfinite(sample[2]) and sample[2] > 0:
                ic50_samples.append(float(sample[2]))
    # A CI built only from the resamples that happened to converge is biased
    # narrow, because the awkward resamples — the ones that carry the uncertainty —
    # are exactly the ones dropped. Report the rate and refuse below 80%.
    attempts = 400
    convergence = len(ic50_samples) / attempts
    interval = (
        [float(np.percentile(ic50_samples, 2.5)), float(np.percentile(ic50_samples, 97.5))]
        if convergence >= 0.8
        else None
    )
    in_range = bool(
        parameters[2] >= concentrations[positive].min()
        and parameters[2] <= concentrations[positive].max()
    )
    return {
        "fitted": True,
        "covariance_estimated": covariance_estimated,
        "bottom": float(parameters[0]),
        "top": float(parameters[1]),
        "ic50_ug_per_ml": float(parameters[2]),
        "hill_slope": float(parameters[3]),
        "r_squared": r_squared,
        "ic50_ci95": interval,
        "bootstrap_convergence_rate": convergence,
        "ic50_ci95_note": (
            None
            if interval is not None
            else f"Only {convergence:.0%} of bootstrap resamples converged; an interval "
            "built from those alone would be biased narrow, so none is reported."
        ),
        "ic50_within_tested_range": in_range,
        "warning": None
        if in_range
        else "The fitted IC50 lies outside the tested concentration range and is an extrapolation.",
    }


def exponential_approach(
    hours: np.ndarray, start: float, plateau: float, rate: float
) -> np.ndarray:
    """A feature relaxing from *start* toward *plateau* at first-order *rate*.

    Morphological responses to a cytotoxic agent do not rise forever; cells round
    up, or spread, and then stop. A straight line through such data reports a
    slope that depends entirely on when imaging stopped, which is why the rate
    constant, not the slope, is the answer to "how quickly".
    """
    return plateau + (start - plateau) * np.exp(-np.abs(rate) * np.asarray(hours, dtype=float))


def fit_kinetics(hours: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    """Fit a rate of change, reporting a half-time where one is identifiable."""
    hours = np.asarray(hours, dtype=float)
    values = np.asarray(values, dtype=float)
    distinct = len(np.unique(hours))
    if distinct < 3 or hours.size < 3:
        return {
            "fitted": False,
            "reason": (
                "A rate needs at least three distinct exposure times; two points fit a "
                f"line with no residual degrees of freedom. This design has {distinct}."
            ),
            "timepoints": distinct,
        }

    # Linear rate first: always identifiable, and it is what a reader expects.
    slope, intercept = np.polyfit(hours, values, 1)
    predicted = slope * hours + intercept
    total = values - values.mean()
    linear_r2 = (
        1.0 - float((values - predicted) @ (values - predicted)) / float(total @ total)
        if total.any()
        else float("nan")
    )

    result: dict[str, Any] = {
        "fitted": True,
        "timepoints": distinct,
        "linear_slope_per_hour": float(slope),
        "linear_r_squared": linear_r2,
        "total_change": float(values[np.argmax(hours)] - values[np.argmin(hours)]),
        "half_time_hours": None,
        "rate_constant_per_hour": None,
        "plateau": None,
        "saturating_r_squared": None,
        "model_note": "Only a linear rate was identifiable.",
    }
    if distinct < 4:
        result["model_note"] = (
            "Only a linear rate was fitted; a saturating model needs at least four "
            "distinct exposure times."
        )
        return result

    guess = [float(values[np.argmin(hours)]), float(values[np.argmax(hours)]), 0.05]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optimize.OptimizeWarning)
            parameters, _ = optimize.curve_fit(
                exponential_approach, hours, values, p0=guess, maxfev=20000
            )
    except (RuntimeError, ValueError, TypeError):
        return result

    saturating = exponential_approach(hours, *parameters)
    saturating_r2 = (
        1.0 - float((values - saturating) @ (values - saturating)) / float(total @ total)
        if total.any()
        else float("nan")
    )
    rate = float(abs(parameters[2]))
    # A rate this slow is indistinguishable from a straight line over the window
    # observed, so quoting a half-time from it would be extrapolation.
    identifiable = rate > 1e-4 and math.log(2) / rate <= 3 * float(hours.max())
    result.update(
        {
            "rate_constant_per_hour": rate,
            "plateau": float(parameters[1]),
            "saturating_r_squared": saturating_r2,
            "half_time_hours": float(math.log(2) / rate) if identifiable else None,
            "model_note": (
                "Saturating fit preferred; half-time is the time to cover half the "
                "distance from the starting value to the plateau."
                if saturating_r2 > linear_r2 and identifiable
                else "Linear fit is as good as the saturating one over this window; "
                "no half-time is quoted, because the response has not visibly plateaued."
            ),
        }
    )
    return result


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Ridge regression with a fixed penalty and an unpenalised intercept.

    ``alpha`` is deliberately fixed rather than tuned. Choosing it by nested
    cross-validation at 20-60 wells would spend more of the data on the choice
    than the choice is worth, and would make the permutation null much harder to
    reason about. On standardised features alpha = 1 is weak regularisation for
    several correlated predictors — which is a further reason the confirmatory
    test uses a single pre-specified feature.
    """
    design = np.hstack([np.ones((x.shape[0], 1)), x])
    penalty = alpha * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def _ridge_predict(coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((x.shape[0], 1)), x]) @ coefficients


def leave_one_day_out(
    features: np.ndarray,
    targets: np.ndarray,
    days: list[str],
    *,
    permutations: int = 500,
    seed: int = 0,
    centre_by_day: bool = True,
) -> dict[str, Any]:
    """Can morphology predict viability on a day the model never saw?

    Held-out *days*, not held-out images: images from the same plate are not
    independent, so a random split would report a number that does not transfer.

    With ``centre_by_day`` the model is fitted to each day's deviations from its
    own mean, and asked to rank the wells *within* the held-out day rather than to
    predict an absolute viability. This is both the honest question and by far the
    more powerful one. Plates differ between days for reasons unrelated to
    treatment, and those day-level shifts are independent in morphology and in
    viability, so a model fitted across days learns a slope corrupted by that
    confounding. Simulated at a within-day correlation of 0.5 and a realistic
    design of 3 days and 30 wells, centring raises power from 0.32 to 0.74, and
    at 4 days and 60 wells from 0.35 to 0.96, while type-I error stays at the
    nominal 0.05.

    The price is a narrower claim, and it is stated in the result: what is
    predicted is a well's position relative to its own day's mean, not its
    absolute viability.
    """
    unique_days = sorted(set(days))
    if len(unique_days) < 2:
        return {
            "evaluated": False,
            "reason": "Leave-one-day-out needs at least two experiment days.",
        }
    day_array = np.asarray(days)

    # Which wells can actually be predicted out-of-fold. A fold whose training
    # side has fewer than two wells is skipped; those wells previously kept a
    # prediction of 0.0 and entered R^2 as though 0.0 were a real estimate, which
    # produced numbers like R^2 = -10.6 while still reporting evaluated: True.
    scored = np.zeros(len(targets), dtype=bool)
    for day in unique_days:
        held_out = day_array == day
        if held_out.sum() and (~held_out).sum() >= 2:
            scored |= held_out
    skipped = int((~scored).sum())
    if scored.sum() < 3:
        return {
            "evaluated": False,
            "reason": (
                "Too few wells can be predicted out-of-fold. Every experiment day needs "
                "at least two wells on the other days to train on."
            ),
            "wells_skipped": skipped,
        }

    feature_means = {day: features[day_array == day].mean(axis=0) for day in unique_days}

    def score(y: np.ndarray) -> tuple[float, float, float]:
        day_means = {day: float(y[day_array == day].mean()) for day in unique_days}
        predictions = np.full(len(y), np.nan, dtype=float)
        for day in unique_days:
            test = day_array == day
            train = ~test
            if train.sum() < 2 or test.sum() == 0:
                continue
            if centre_by_day:
                other = [name for name in unique_days if name != day]
                train_x = np.vstack(
                    [features[day_array == name] - feature_means[name] for name in other]
                )
                train_y = np.concatenate([y[day_array == name] - day_means[name] for name in other])
                test_x = features[test] - feature_means[day]
                offset = day_means[day]
            else:
                train_x, train_y, test_x, offset = features[train], y[train], features[test], 0.0
            mean = train_x.mean(axis=0)
            scale = train_x.std(axis=0)
            scale[scale < 1e-9] = 1.0
            coefficients = _ridge_fit((train_x - mean) / scale, train_y)
            predictions[test] = _ridge_predict(coefficients, (test_x - mean) / scale) + offset
        actual = y[scored]
        predicted = predictions[scored]
        residual = actual - predicted
        total = actual - actual.mean()
        r2 = (
            1.0 - float(residual @ residual) / float(total @ total) if total.any() else float("nan")
        )
        # R^2 against the global mean is flattered by between-day differences the
        # model can read straight off the features. Centring both sides on their
        # own day's mean asks the harder question: does morphology explain
        # variation *within* a day?
        offsets = np.array([day_means[day] for day in day_array[scored]])
        centred_actual = actual - offsets
        centred_residual = centred_actual - (predicted - offsets)
        centred_total = centred_actual - centred_actual.mean()
        within_day_r2 = (
            1.0 - float(centred_residual @ centred_residual) / float(centred_total @ centred_total)
            if centred_total.any()
            else float("nan")
        )
        rmse = float(np.sqrt(np.mean(residual**2)))
        return r2, rmse, within_day_r2

    observed_r2, rmse, within_day_r2 = score(targets)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(permutations):
        shuffled = targets.copy()
        # Permute within day so the null keeps day-level structure.
        for day in unique_days:
            selection = day_array == day
            shuffled[selection] = rng.permutation(shuffled[selection])
        null.append(score(shuffled)[0])
    null_array = np.asarray(null)
    p_value = float((np.sum(null_array >= observed_r2) + 1) / (len(null_array) + 1))
    return {
        "evaluated": True,
        "held_out_days": unique_days,
        "wells_scored": int(scored.sum()),
        "wells_skipped": skipped,
        "centred_by_day": centre_by_day,
        "claim": (
            "Predicts each well's viability relative to its own experiment day's mean."
            if centre_by_day
            else "Predicts absolute viability, which day-to-day batch effects confound."
        ),
        "r_squared": observed_r2,
        "within_day_r_squared": within_day_r2,
        "r_squared_percentile_in_null": float((null_array < observed_r2).mean()),
        "rmse": rmse,
        "permutation_p_value": p_value,
        "permutations": permutations,
        "null_r_squared_mean": float(null_array.mean()),
        "null_r_squared_p95": float(np.percentile(null_array, 95)),
        "interpretation": (
            "R^2 is measured on days excluded from fitting; the p-value compares it "
            "against within-day label permutations. Quote within_day_r_squared next to "
            "r_squared: the plain R^2 is measured against the global mean, so part of it "
            "comes free from between-day differences rather than from morphology."
        ),
    }


def stratified_spearman(
    concentrations: np.ndarray,
    values: np.ndarray,
    days: np.ndarray,
    *,
    permutations: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Spearman rho with a p-value from permuting dose labels *within* each day.

    The pooled parametric p-value treats wells from different days as
    exchangeable. If the concentration ladder is not perfectly balanced within
    each day — and in a real experiment it never quite is — a day effect leaks
    into the dose effect and the p-value is too small. Permuting concentration
    only within a day destroys the dose relationship while keeping every
    day-level difference intact, so what survives is dose and nothing else.
    """
    base = _safe_spearman(concentrations, values)
    if base["spearman_rho"] is None:
        return {**base, "p_value_day_stratified": None, "permutations": 0}
    observed = abs(float(base["spearman_rho"]))
    rng = np.random.default_rng(seed)
    unique_days = np.unique(days)
    at_least_as_extreme = 0
    for _ in range(permutations):
        shuffled = concentrations.copy()
        for day in unique_days:
            selection = days == day
            shuffled[selection] = rng.permutation(shuffled[selection])
        candidate = _safe_spearman(shuffled, values)["spearman_rho"]
        if candidate is not None and abs(float(candidate)) >= observed:
            at_least_as_extreme += 1
    return {
        **base,
        "p_value_day_stratified": float((at_least_as_extreme + 1) / (permutations + 1)),
        "permutations": permutations,
    }


def analyse_kinetics(
    well_rows: list[dict[str, Any]],
    feature: str,
    *,
    permutations: int = 2000,
) -> dict[str, Any]:
    """How fast does *feature* change with exposure time, and does dose change the rate?

    Two experimental designs give this answer, and they are not interchangeable:

    * **longitudinal** — the same well imaged repeatedly. Imaging is
      non-destructive, so this is usually possible, and it is much the stronger
      design: a rate is fitted *within* each well, so well-to-well variation
      cannot masquerade as a time effect.
    * **cross-sectional** — a different well per timepoint, as a destructive assay
      forces. A rate can only be fitted across wells, so it is confounded with
      whatever differs between them.

    Which one you ran is detected from the manifest and reported, because a reader
    cannot otherwise tell which claim the number supports.
    """
    times = sorted({float(row["exposure_hours"]) for row in well_rows})
    if len(times) < 3:
        return {
            "evaluated": False,
            "reason": (
                f"Kinetics needs at least three distinct exposure times; this study has "
                f"{len(times)} ({times}). Image the same plates at several timepoints to "
                "measure how quickly changes occur."
            ),
            "exposure_hours": times,
        }
    if not all(isinstance(row.get(feature), int | float) for row in well_rows):
        return {
            "evaluated": False,
            "reason": f"Feature {feature!r} is not available in every well.",
        }

    repeats: dict[tuple[str, str, str], set[float]] = defaultdict(set)
    for row in well_rows:
        key = (str(row["experiment_day"]), str(row["plate_id"]), str(row["well_id"]))
        repeats[key].add(float(row["exposure_hours"]))
    longitudinal_units = [key for key, values in repeats.items() if len(values) >= 3]
    design = "longitudinal" if longitudinal_units else "cross-sectional"

    per_concentration: dict[str, Any] = {}
    rates: list[float] = []
    rate_concentrations: list[float] = []

    if design == "longitudinal":
        # A rate per well, which keeps the well as the unit of analysis.
        for key in longitudinal_units:
            member = [
                row
                for row in well_rows
                if (str(row["experiment_day"]), str(row["plate_id"]), str(row["well_id"])) == key
            ]
            hours = np.array([float(row["exposure_hours"]) for row in member])
            values = np.array([float(row[feature]) for row in member])
            fit = fit_kinetics(hours, values)
            if fit["fitted"]:
                rates.append(float(fit["linear_slope_per_hour"]))
                rate_concentrations.append(float(member[0]["concentration_ug_per_ml"]))
        grouped: dict[float, list[float]] = defaultdict(list)
        for concentration, rate in zip(rate_concentrations, rates, strict=True):
            grouped[concentration].append(rate)
        for concentration, well_slopes in sorted(grouped.items()):
            per_concentration[str(concentration)] = {
                "wells": len(well_slopes),
                "mean_slope_per_hour": float(np.mean(well_slopes)),
                "sd_slope_per_hour": (
                    float(np.std(well_slopes, ddof=1)) if len(well_slopes) > 1 else 0.0
                ),
            }
    else:
        for concentration in sorted({float(row["concentration_ug_per_ml"]) for row in well_rows}):
            member = [
                row for row in well_rows if float(row["concentration_ug_per_ml"]) == concentration
            ]
            hours = np.array([float(row["exposure_hours"]) for row in member])
            values = np.array([float(row[feature]) for row in member])
            fit = fit_kinetics(hours, values)
            per_concentration[str(concentration)] = {"wells": len(member), **fit}
            if fit["fitted"]:
                rates.append(float(fit["linear_slope_per_hour"]))
                rate_concentrations.append(concentration)

    dose_dependence: dict[str, Any] = {"evaluated": False, "reason": "Fewer than three rates."}
    if len(rates) >= 3:
        dose_dependence = {
            "evaluated": True,
            **stratified_spearman(
                np.array(rate_concentrations),
                np.array(rates),
                np.array(["all"] * len(rates)),
                permutations=min(permutations, 2000),
            ),
            "question": "Does the rate of change itself depend on concentration?",
        }

    return {
        "evaluated": True,
        "feature": feature,
        "design": design,
        "design_note": (
            "The same wells were imaged at several timepoints, so each rate is fitted "
            "within a well."
            if design == "longitudinal"
            else "Each timepoint uses different wells, so rates are fitted across wells "
            "and are confounded with well-to-well variation. Imaging the same wells "
            "repeatedly would remove that."
        ),
        "exposure_hours": times,
        "units_with_a_rate": len(rates),
        "per_concentration": per_concentration,
        "dose_dependence_of_rate": dose_dependence,
        "caveat": (
            "Rates describe the population average in each well, not individual cells. "
            "Following a single cell through time needs instance segmentation and "
            "frame-to-frame tracking, neither of which this pipeline does yet."
        ),
    }


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Spearman correlation that reports undefined rather than warning."""
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return {"spearman_rho": None, "p_value": None, "note": "input had no variance"}
    result = stats.spearmanr(x, y)
    return {"spearman_rho": float(result.statistic), "p_value": float(result.pvalue)}


def analyse(
    records: list[TreatmentRecord],
    image_summaries: dict[Path, dict[str, Any]],
    *,
    features: tuple[str, ...] = MORPHOLOGY_FEATURES,
    primary_feature: str = PRIMARY_FEATURE,
    permutations: int = 2000,
) -> dict[str, Any]:
    """Turn per-image morphology into dose response and viability linkage."""
    rows: list[dict[str, Any]] = []
    for record in records:
        summary = image_summaries.get(record.image_path)
        if summary is None:
            continue
        rows.append(
            {
                "experiment_day": record.experiment_day,
                "plate_id": record.plate_id,
                "well_id": record.well_id,
                "field": record.field,
                "treatment": record.treatment,
                "concentration_ug_per_ml": record.concentration_ug_per_ml,
                "exposure_hours": record.exposure_hours,
                "control_type": record.control_type,
                "replicate": record.replicate,
                "viability_fraction": record.viability_fraction,
                **{name: summary.get(name) for name in features},
            }
        )
    if not rows:
        raise TreatmentDataError("No image was successfully analysed.")

    # Average fields up to the well: the well is the independent unit.
    wells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        wells[
            (
                row["experiment_day"],
                row["plate_id"],
                row["well_id"],
                row["treatment"],
                row["concentration_ug_per_ml"],
                row["exposure_hours"],
            )
        ].append(row)
    well_rows: list[dict[str, Any]] = []
    for key, members in sorted(wells.items(), key=lambda item: str(item[0])):
        aggregated: dict[str, Any] = {
            "experiment_day": key[0],
            "plate_id": key[1],
            "well_id": key[2],
            "treatment": key[3],
            "concentration_ug_per_ml": key[4],
            "exposure_hours": key[5],
            "fields": len(members),
        }
        for name in features:
            values = [float(row[name]) for row in members if isinstance(row.get(name), int | float)]
            aggregated[name] = float(np.mean(values)) if values else None
        viabilities = [
            float(row["viability_fraction"])
            for row in members
            if isinstance(row.get("viability_fraction"), int | float)
        ]
        aggregated["viability_fraction"] = float(np.mean(viabilities)) if viabilities else None
        well_rows.append(aggregated)

    present = [name for name in features if all(row.get(name) is not None for row in well_rows)]
    # A feature with no variance across wells cannot show a dose response, and a
    # correlation on a constant array is undefined rather than merely weak.
    constant = [
        name for name in present if float(np.std([float(row[name]) for row in well_rows])) < 1e-12
    ]
    usable = [name for name in present if name not in constant]
    concentrations = np.array(
        [float(row["concentration_ug_per_ml"]) for row in well_rows], dtype=float
    )

    dose_response: dict[str, Any] = {}
    p_values: list[float] = []
    day_labels = np.array([str(row["experiment_day"]) for row in well_rows])
    for name in usable:
        feature = np.array([float(row[name]) for row in well_rows], dtype=float)
        correlation = stratified_spearman(
            concentrations, feature, day_labels, permutations=permutations
        )
        per_day: dict[str, Any] = {}
        for day in sorted({str(row["experiment_day"]) for row in well_rows}):
            selection = np.array([str(row["experiment_day"]) == day for row in well_rows])
            if selection.sum() >= 3:
                per_day[day] = {
                    **_safe_spearman(concentrations[selection], feature[selection]),
                    "wells": int(selection.sum()),
                }
        dose_response[name] = {
            **correlation,
            "feature_kind": FEATURE_KINDS.get(name, "unclassified"),
            "wells": len(well_rows),
            "per_day": per_day,
            "fit": fit_dose_response(concentrations, feature),
        }
        headline = correlation.get("p_value_day_stratified")
        p_values.append(float("nan") if headline is None else float(headline))
    for name, q_value in zip(usable, benjamini_hochberg(p_values), strict=True):
        dose_response[name]["q_value_bh"] = None if np.isnan(q_value) else q_value
        dose_response[name]["significant_at_q_0.05"] = bool(q_value < 0.05)
    for name in constant:
        dose_response[name] = {
            "constant": True,
            "feature_kind": FEATURE_KINDS.get(name, "unclassified"),
            "note": "This feature took the same value in every well; no test was run.",
        }

    viability_rows = [
        row for row in well_rows if isinstance(row.get("viability_fraction"), int | float)
    ]
    viability: dict[str, Any] = {"wells_with_viability": len(viability_rows)}
    if len(viability_rows) >= 6 and usable:
        matrix = np.array(
            [[float(row[name]) for name in usable] for row in viability_rows], dtype=float
        )
        targets = np.array([float(row["viability_fraction"]) for row in viability_rows])
        days = [str(row["experiment_day"]) for row in viability_rows]
        # Eight correlated features into a ridge at 20-30 wells roughly halves the
        # power to detect a real association. The confirmatory test therefore uses
        # the single pre-specified endpoint; the multi-feature model is reported
        # beside it, labelled exploratory.
        viability["exploratory_prediction"] = leave_one_day_out(matrix, targets, days)
        viability["prediction"] = viability["exploratory_prediction"]
        if primary_feature in usable:
            column = np.array(
                [[float(row[primary_feature])] for row in viability_rows], dtype=float
            )
            viability["confirmatory_prediction"] = {
                "feature": primary_feature,
                "feature_kind": FEATURE_KINDS.get(primary_feature, "unclassified"),
                "pre_specified": True,
                **leave_one_day_out(column, targets, days),
            }
        else:
            viability["confirmatory_prediction"] = {
                "evaluated": False,
                "feature": primary_feature,
                "reason": (
                    f"The pre-specified endpoint {primary_feature!r} was not measurable "
                    "in every well, so no confirmatory test was run. Everything below is "
                    "exploratory."
                ),
            }
        viability["per_feature_correlation"] = {
            name: _safe_spearman(matrix[:, index], targets) for index, name in enumerate(usable)
        }
        viability["dose_response_of_viability"] = fit_dose_response(
            np.array([float(row["concentration_ug_per_ml"]) for row in viability_rows]), targets
        )
    else:
        viability["prediction"] = {
            "evaluated": False,
            "reason": (
                "At least six wells with paired viability measurements across two or more "
                "experiment days are needed before a prediction claim is meaningful."
            ),
        }

    kinetics = analyse_kinetics(
        well_rows,
        primary_feature if primary_feature in usable else (usable[0] if usable else ""),
        permutations=permutations,
    )

    return {
        "images_analysed": len(rows),
        "constant_features": constant,
        "kinetics": kinetics,
        "wells": len(well_rows),
        "experiment_days": sorted({str(row["experiment_day"]) for row in well_rows}),
        "concentrations_ug_per_ml": sorted({float(value) for value in concentrations}),
        "exposure_hours": sorted({float(row["exposure_hours"]) for row in well_rows}),
        "features_analysed": usable,
        "feature_kinds": {name: FEATURE_KINDS.get(name, "unclassified") for name in usable},
        "primary_feature": primary_feature,
        "shape_features_available": [name for name in usable if FEATURE_KINDS.get(name) == "shape"],
        "well_level_rows": well_rows,
        "dose_response": dose_response,
        "viability_linkage": viability,
        "confluency_caveat": (
            "Features marked 'density' or 'density-contaminated' move with how crowded "
            "the field is, not with cell shape. While the segmenter is semantic, "
            "touching cells merge into one region, so a dose response in those features "
            "may say only 'fewer cells at higher dose' — which the viability assay "
            "already says. Conclusions about morphology should rest on the 'shape' "
            "features, or on shape adjusted for total_area_pixels as a covariate."
        ),
        "statistical_notes": [
            "The well is treated as the independent unit; fields within a well are averaged "
            "first so that imaging more fields cannot inflate the sample size.",
            "Spearman correlation is used because a dose response need not be linear.",
            "p-values across morphology features are corrected with Benjamini-Hochberg; "
            "both raw p-values and q-values are reported.",
            "Morphology is evaluated as a predictor of viability, never as a substitute for "
            "it, and only on experiment days excluded from fitting.",
            "The headline dose-response p-value comes from permuting concentration labels "
            "within each experiment day, so a day effect cannot masquerade as a dose "
            "effect; the parametric p-value is reported alongside as a descriptive.",
            "One endpoint is pre-specified and tested on its own; the multi-feature model "
            "is exploratory, because eight correlated predictors at 20-30 wells roughly "
            "halves the power to detect a real association.",
            "Leave-one-day-out R^2 is reported against the global mean and, separately, "
            "within day. Only the within-day value is free of between-day variance.",
            "Rates of change are reported only when at least three distinct exposure "
            "times exist, and the report says whether they were fitted within a well "
            "(longitudinal) or across wells (cross-sectional).",
        ],
    }


TEMPLATE_HEADER = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


def write_template(path: Path) -> Path:
    """Write an empty treatment manifest with the columns the analysis requires."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    example = {
        "image_path": "images/day1/plateA/A01_f1.tif",
        "experiment_day": "2026-03-14",
        "plate_id": "plateA",
        "well_id": "A01",
        "field": "1",
        "cell_line": "RAW264.7",
        "treatment": "Tribonema extract",
        "concentration_ug_per_ml": "0",
        "exposure_hours": "24",
        "control_type": "vehicle",
        "replicate": "1",
        "viability_fraction": "",
        "mts_absorbance": "",
        "mts_blank_absorbance": "",
        "micrometers_per_pixel": "",
        "operator": "",
        "permission_status": "",
        "notes": "one row per image; delete this example row",
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TEMPLATE_HEADER))
        writer.writeheader()
        writer.writerow({key: example.get(key, "") for key in TEMPLATE_HEADER})
    return path


def plan_experiment(
    path: Path,
    *,
    days: tuple[str, ...] = ("day-1", "day-2", "day-3"),
    concentrations: tuple[float, ...] = (0.0, 12.5, 25.0, 50.0, 100.0, 200.0),
    exposure_hours: tuple[float, ...] = (0.0, 6.0, 12.0, 24.0, 48.0),
    wells_per_condition: int = 3,
    fields: int = 2,
    cell_line: str = "RAW264.7",
    treatment: str = "Tribonema extract",
    positive_control: bool = True,
) -> dict[str, Any]:
    """Write every row an experiment will need, with the image paths left blank.

    This is a collection checklist, not a template. Each row is one image that has
    to exist for the analysis to run, so the file doubles as the plan you hand a
    mentor and the manifest you fill in at the bench.

    The default design is checked against everything the analysis enforces: a
    vehicle control, at least two experiment days, at least two independent wells
    per condition, five distinct non-zero concentrations across at least eight
    wells so an IC50 is identifiable, and at least four exposure times so a
    half-time is.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for day in days:
        well_number = 0
        arms: list[tuple[str, float, str]] = [
            (treatment, concentration, "vehicle" if concentration == 0 else "treated")
            for concentration in concentrations
        ]
        if positive_control:
            arms.append(("positive control", float("nan"), "positive"))
        for arm, concentration, control_type in arms:
            for replicate in range(1, wells_per_condition + 1):
                well_number += 1
                well = f"{chr(ord('A') + (well_number - 1) // 12)}{(well_number - 1) % 12 + 1:02d}"
                for exposure in exposure_hours:
                    for field in range(1, fields + 1):
                        rows.append(
                            {
                                "image_path": "",
                                "experiment_day": day,
                                "plate_id": f"plate-{day}",
                                "well_id": well,
                                "field": str(field),
                                "cell_line": cell_line,
                                "treatment": arm,
                                "concentration_ug_per_ml": (
                                    "" if math.isnan(concentration) else concentration
                                ),
                                "exposure_hours": exposure,
                                "control_type": control_type,
                                "replicate": str(replicate),
                                "viability_fraction": "",
                                "mts_absorbance": "",
                                "mts_blank_absorbance": "",
                                "micrometers_per_pixel": "",
                                "operator": "",
                                "permission_status": "",
                                "notes": "",
                            }
                        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TEMPLATE_HEADER))
        writer.writeheader()
        writer.writerows(rows)

    analysis_wells = len(days) * len(concentrations) * wells_per_condition
    non_zero_wells = len(days) * (len(concentrations) - 1) * wells_per_condition
    return {
        "manifest": str(path),
        "rows_to_fill": len(rows),
        "images_to_capture": len(rows),
        "analysis_wells": analysis_wells,
        "non_zero_concentration_wells": non_zero_wells,
        "experiment_days": list(days),
        "concentrations_ug_per_ml": list(concentrations),
        "exposure_hours": list(exposure_hours),
        "fields_per_well": fields,
        "requirements_met": {
            "vehicle_control": 0.0 in concentrations,
            "two_or_more_days": len(days) >= 2,
            "two_or_more_wells_per_condition": wells_per_condition >= 2,
            "ic50_identifiable": len([c for c in concentrations if c > 0]) >= 5
            and non_zero_wells >= 8,
            "half_time_identifiable": len(set(exposure_hours)) >= 4,
        },
        "note": (
            "An 'experiment day' is one independent biological replicate - a plate "
            "seeded on its own date - not an imaging session. A plate followed for 48 "
            "hours spans several calendar days but is one experiment day, and every row "
            "for it carries the same experiment_day label."
        ),
    }


def run_treatment_analysis(
    manifest_path: Path,
    output_dir: Path,
    *,
    checkpoint: Path | None = None,
    device: str = "auto",
    threshold: float = 0.5,
    instance_method: str = "watershed_split",
    min_area: int = 20,
    background_radius: float = 7.0,
    primary_feature: str = PRIMARY_FEATURE,
    permutations: int = 2000,
) -> dict[str, Any]:
    """Segment every treatment image, then run the dose-response analysis.

    A checkpoint uses the trained segmenter; without one the transparent classical
    baseline is used, and the report says which produced the numbers.
    """
    from PIL import Image

    from tribovision import morphology, provenance

    records = load_treatment_manifest(manifest_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    segmenter = "classical-baseline"
    model = None
    torch_device = None
    image_size = 512
    if checkpoint is not None:
        from tribovision.predict import load_checkpoint, predict_mask
        from tribovision.training import resolve_device

        torch_device = resolve_device(device)
        model, payload = load_checkpoint(Path(checkpoint), torch_device)
        preprocessing = payload.get("preprocessing") or {}
        image_size = int(preprocessing.get("image_size") or 512)
        segmenter = f"neural:{provenance.relative_to_repo(Path(checkpoint))}"

    summaries: dict[Path, dict[str, Any]] = {}
    feature_rows: list[dict[str, Any]] = []
    for record in records:
        if not record.image_path.is_file():
            raise TreatmentDataError(
                f"Image listed in the manifest is missing: {record.image_path}"
            )
        with Image.open(record.image_path) as handle:
            handle.load()
            image = handle.convert("L")
        if model is not None and torch_device is not None:
            from tribovision.predict import predict_mask

            mask, _ = predict_mask(
                model, image, image_size=image_size, device=torch_device, threshold=threshold
            )
        else:
            from tribovision.baseline import segment_classical

            mask = segment_classical(image, background_radius=background_radius, min_area=min_area)
        labels = morphology.label_objects(mask, method=instance_method, min_area=min_area)
        rows = morphology.measure(
            labels,
            np.asarray(image, dtype=np.float32),
            calibration=morphology.Calibration(record.micrometers_per_pixel),
            method=instance_method,
            extra={
                "image": record.image_path.name,
                "experiment_day": record.experiment_day,
                "plate_id": record.plate_id,
                "well_id": record.well_id,
                "concentration_ug_per_ml": record.concentration_ug_per_ml,
                "exposure_hours": record.exposure_hours,
            },
        )
        feature_rows.extend(rows)
        summaries[record.image_path] = morphology.summarise_image(rows)

    if feature_rows:
        fieldnames = sorted({key for row in feature_rows for key in row})
        with (output_dir / "treatment_morphology.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(feature_rows)

    analysis = analyse(
        records, summaries, primary_feature=primary_feature, permutations=permutations
    )
    report = {
        "manifest": provenance.relative_to_repo(Path(manifest_path)),
        "segmenter": segmenter,
        "instance_method": instance_method,
        "threshold": threshold if model is not None else None,
        **analysis,
        "environment": provenance.environment(),
        "claim_boundary": (
            "This analysis describes visible morphology and its statistical association "
            "with the supplied viability measurements. It does not establish a mechanism, "
            "and it does not by itself demonstrate an anti-tumour effect."
        ),
    }
    with (output_dir / "well_level_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = analysis["well_level_rows"]
        if rows:
            writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "treatment_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return report
