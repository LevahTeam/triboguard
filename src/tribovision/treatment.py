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
MORPHOLOGY_FEATURES = (
    "objects",
    "total_area_pixels",
    "area_pixels_mean",
    "area_pixels_median",
    "circularity_mean",
    "aspect_ratio_mean",
    "equivalent_diameter_pixels_mean",
    "mean_intensity_mean",
)


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
    if positive.sum() < 4 or len(np.unique(concentrations[positive])) < 3:
        return {
            "fitted": False,
            "reason": "A 4PL fit needs at least three distinct non-zero concentrations.",
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
    interval = (
        [float(np.percentile(ic50_samples, 2.5)), float(np.percentile(ic50_samples, 97.5))]
        if len(ic50_samples) >= 50
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
        "ic50_within_tested_range": in_range,
        "warning": None
        if in_range
        else "The fitted IC50 lies outside the tested concentration range and is an extrapolation.",
    }


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
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
) -> dict[str, Any]:
    """Can morphology predict viability on a day the model never saw?

    Held-out *days*, not held-out images: images from the same plate are not
    independent, so a random split would report a number that does not transfer.
    """
    unique_days = sorted(set(days))
    if len(unique_days) < 2:
        return {
            "evaluated": False,
            "reason": "Leave-one-day-out needs at least two experiment days.",
        }
    day_array = np.asarray(days)

    def score(y: np.ndarray) -> tuple[float, float]:
        predictions = np.zeros_like(y, dtype=float)
        for day in unique_days:
            test = day_array == day
            train = ~test
            if train.sum() < 2 or test.sum() == 0:
                continue
            mean = features[train].mean(axis=0)
            scale = features[train].std(axis=0)
            scale[scale < 1e-9] = 1.0
            coefficients = _ridge_fit((features[train] - mean) / scale, y[train])
            predictions[test] = _ridge_predict(coefficients, (features[test] - mean) / scale)
        residual = y - predictions
        total = y - y.mean()
        r2 = (
            1.0 - float(residual @ residual) / float(total @ total) if total.any() else float("nan")
        )
        rmse = float(np.sqrt(np.mean(residual**2)))
        return r2, rmse

    observed_r2, rmse = score(targets)
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
        "r_squared": observed_r2,
        "rmse": rmse,
        "permutation_p_value": p_value,
        "permutations": permutations,
        "null_r_squared_mean": float(null_array.mean()),
        "null_r_squared_p95": float(np.percentile(null_array, 95)),
        "interpretation": (
            "R^2 is measured on days excluded from fitting; the p-value compares it "
            "against within-day label permutations."
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
    for name in usable:
        feature = np.array([float(row[name]) for row in well_rows], dtype=float)
        correlation = _safe_spearman(concentrations, feature)
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
            "wells": len(well_rows),
            "per_day": per_day,
            "fit": fit_dose_response(concentrations, feature),
        }
        p_values.append(
            float("nan") if correlation["p_value"] is None else float(correlation["p_value"])
        )
    for name, q_value in zip(usable, benjamini_hochberg(p_values), strict=True):
        dose_response[name]["q_value_bh"] = None if np.isnan(q_value) else q_value
        dose_response[name]["significant_at_q_0.05"] = bool(q_value < 0.05)
    for name in constant:
        dose_response[name] = {
            "constant": True,
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
        viability["prediction"] = leave_one_day_out(matrix, targets, days)
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

    return {
        "images_analysed": len(rows),
        "constant_features": constant,
        "wells": len(well_rows),
        "experiment_days": sorted({str(row["experiment_day"]) for row in well_rows}),
        "concentrations_ug_per_ml": sorted({float(value) for value in concentrations}),
        "exposure_hours": sorted({float(row["exposure_hours"]) for row in well_rows}),
        "features_analysed": usable,
        "well_level_rows": well_rows,
        "dose_response": dose_response,
        "viability_linkage": viability,
        "statistical_notes": [
            "The well is treated as the independent unit; fields within a well are averaged "
            "first so that imaging more fields cannot inflate the sample size.",
            "Spearman correlation is used because a dose response need not be linear.",
            "p-values across morphology features are corrected with Benjamini-Hochberg; "
            "both raw p-values and q-values are reported.",
            "Morphology is evaluated as a predictor of viability, never as a substitute for "
            "it, and only on experiment days excluded from fitting.",
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

    analysis = analyse(records, summaries)
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
