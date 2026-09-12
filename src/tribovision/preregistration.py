"""Checking a finished experiment against criteria written before it ran.

The criteria live in a committed plan file and this module applies them
mechanically. Nothing here makes a judgement call after the data exists: the
thresholds, the cell lines that count, and what "replicated" means are all read
from the plan. The only way to change a verdict is to change a file, and the
evaluation refuses any artifact that was scored under a different version of
that file -- so a criterion edited after the results were seen does not quietly
re-score the old results, it invalidates them.

Four outcomes are kept apart on purpose. A result can pass, fail, be *void*
(a validity check failed -- the segmenter could not see healthy cells, or the
coin-flip split produced a gap with no damage applied -- so it cannot be
interpreted either way) or be *missing*. Collapsing the last two into "fail"
would hide a broken experiment behind a negative finding; collapsing them into
"pass" would be worse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tribovision import provenance

REQUIRED_KEYS = (
    "id",
    "design",
    "primary_endpoint",
    "confirmatory_cell_lines",
    "exploratory_cell_lines",
    "criteria",
)
CRITERIA = (
    "baseline_detection",
    "negative_control",
    "replication",
    "segmenter_generality",
    "threshold_robustness",
    "dose_response",
)

PASS = "pass"
FAIL = "fail"
VOID = "void"
MISSING = "missing"


class PreregistrationError(ValueError):
    """Raised when a plan is malformed or an artifact was scored under another plan."""


def load_plan(path: str | Path) -> dict[str, Any]:
    """Read a plan and check it can be applied without guessing."""
    plan: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_KEYS if key not in plan]
    if missing:
        raise PreregistrationError(f"plan is missing {missing}.")
    absent = [name for name in CRITERIA if name not in plan["criteria"]]
    if absent:
        raise PreregistrationError(f"plan defines no criterion for {absent}.")
    overlap = set(plan["confirmatory_cell_lines"]) & set(plan["exploratory_cell_lines"])
    if overlap:
        raise PreregistrationError(
            f"a cell line cannot be both confirmatory and exploratory: {sorted(overlap)}."
        )
    if not plan["confirmatory_cell_lines"]:
        raise PreregistrationError("a plan with no confirmatory cell line confirms nothing.")
    primary = plan["primary_endpoint"].get("primary_segmenter")
    if primary not in plan["design"].get("segmenters", []):
        raise PreregistrationError(f"primary segmenter {primary!r} is not in the design.")
    return plan


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(provenance.repo_root()))
    except ValueError:
        return str(path)


def plan_fingerprint(path: str | Path) -> dict[str, str]:
    """What a run records about the plan it was executed under."""
    location = Path(path)
    return {"path": _display(location), "sha256": provenance.sha256_file(location)}


def summary_key(cell_line: str, severity: float, segmenter: str, coverage: float) -> str:
    """The key the damage sweep writes each summary row under."""
    return f"{cell_line}/{severity:.1f}/{segmenter}/cov{coverage:.1f}"


def _interval(
    summary: dict[str, Any], cell_line: str, severity: float, segmenter: str, coverage: float
) -> dict[str, Any] | None:
    row = summary.get(summary_key(cell_line, severity, segmenter, coverage))
    if row is None:
        return None
    interval = row.get("paired_difference") or {}
    if not interval.get("evaluated", False):
        return None
    return dict(interval)


def _brief(interval: dict[str, Any] | None) -> dict[str, float] | None:
    if interval is None:
        return None
    return {key: float(interval[key]) for key in ("difference", "ci_low", "ci_high")}


def _lines(plan: dict[str, Any]) -> list[str]:
    return list(plan["confirmatory_cell_lines"]) + list(plan["exploratory_cell_lines"])


def check_baseline_detection(plan: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """The primary segmenter must see healthy cells before it can be said to lose damaged ones.

    Without this, a segmenter that fails on a cell line outright -- a cell size
    it was never tuned for, say -- finds almost nothing in either arm, shows no
    gap, and reads as evidence *against* damage blindness. That would be a
    broken measurement reported as a negative finding.
    """
    spec = plan["criteria"]["baseline_detection"]
    primary = plan["primary_endpoint"]["primary_segmenter"]
    floor = float(spec["minimum_rate"])
    results: dict[str, Any] = {}
    for line in _lines(plan):
        row = summary.get(summary_key(line, spec["severity"], primary, spec["coverage_threshold"]))
        rate = None if row is None else row.get("mean_intact_rate")
        status = MISSING if rate is None else PASS if float(rate) >= floor else FAIL
        results[line] = {"status": status, "mean_intact_rate": rate}
    return results


def check_negative_control(plan: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """With no damage applied, every segmenter must show no gap."""
    spec = plan["criteria"]["negative_control"]
    margin = float(spec["equivalence_margin"])
    results: dict[str, Any] = {}
    for line in _lines(plan):
        results[line] = {}
        for segmenter in plan["design"]["segmenters"]:
            interval = _interval(
                summary, line, spec["severity"], segmenter, spec["coverage_threshold"]
            )
            if interval is None:
                status = MISSING
            elif -margin <= interval["ci_low"] and interval["ci_high"] <= margin:
                status = PASS
            else:
                status = FAIL
            results[line][segmenter] = {"status": status, "interval": _brief(interval)}
    return results


def check_replication(plan: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """The primary endpoint: the primary segmenter's gap clears the minimum effect."""
    spec = plan["criteria"]["replication"]
    primary = plan["primary_endpoint"]["primary_segmenter"]
    floor = float(spec["minimum_effect"])
    results: dict[str, Any] = {}
    for line in _lines(plan):
        interval = _interval(summary, line, spec["severity"], primary, spec["coverage_threshold"])
        status = MISSING if interval is None else PASS if interval["ci_low"] > floor else FAIL
        results[line] = {"status": status, "interval": _brief(interval)}
    return results


def check_segmenter_generality(plan: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """Whether the gap belongs to segmenters in general or to one of them."""
    spec = plan["criteria"]["segmenter_generality"]
    floor = float(spec["minimum_effect"])
    required = int(spec["required_segmenters"])
    results: dict[str, Any] = {}
    for line in _lines(plan):
        passing, missing = [], []
        for segmenter in plan["design"]["segmenters"]:
            interval = _interval(
                summary, line, spec["severity"], segmenter, spec["coverage_threshold"]
            )
            if interval is None:
                missing.append(segmenter)
            elif interval["ci_low"] > floor:
                passing.append(segmenter)
        if len(passing) >= required:
            status = PASS
        elif len(passing) + len(missing) >= required:
            # The absent results could still decide it, so it is not a failure.
            status = MISSING
        else:
            status = FAIL
        results[line] = {"status": status, "passing": passing, "missing": missing}
    return results


def check_threshold_robustness(plan: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """Whether the result survives moving the detection line."""
    spec = plan["criteria"]["threshold_robustness"]
    primary = plan["primary_endpoint"]["primary_segmenter"]
    floor = float(spec["minimum_effect"])
    results: dict[str, Any] = {}
    for line in _lines(plan):
        estimates: dict[str, float | None] = {}
        for coverage in spec["coverage_thresholds"]:
            interval = _interval(summary, line, spec["severity"], primary, coverage)
            estimates[f"{coverage:.1f}"] = None if interval is None else interval["difference"]
        if any(value is None for value in estimates.values()):
            status = MISSING
        else:
            status = PASS if all(v is not None and v > floor for v in estimates.values()) else FAIL
        results[line] = {"status": status, "estimates": estimates}
    return results


def check_dose_response(plan: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """Whether more damage never buys a meaningfully smaller gap."""
    spec = plan["criteria"]["dose_response"]
    primary = plan["primary_endpoint"]["primary_segmenter"]
    tolerance = float(spec["tolerance"])
    results: dict[str, Any] = {}
    for line in _lines(plan):
        estimates: list[float | None] = []
        for severity in spec["severities"]:
            interval = _interval(summary, line, severity, primary, spec["coverage_threshold"])
            estimates.append(None if interval is None else interval["difference"])
        known = [value for value in estimates if value is not None]
        if len(known) != len(estimates):
            status = MISSING
        else:
            # Consecutive pairs: the second list is one shorter by construction.
            falls = [
                later < earlier - tolerance
                for earlier, later in zip(known, known[1:], strict=False)
            ]
            status = FAIL if any(falls) else PASS
        results[line] = {
            "status": status,
            "estimates": dict(
                zip([f"{s:.1f}" for s in spec["severities"]], estimates, strict=True)
            ),
        }
    return results


def evaluate(plan: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """Apply every criterion, then the verdict rule, exactly as the plan states them."""
    baseline = check_baseline_detection(plan, summary)
    control = check_negative_control(plan, summary)
    replication = check_replication(plan, summary)
    primary = plan["primary_endpoint"]["primary_segmenter"]

    def classify(line: str) -> str:
        validity = (baseline[line]["status"], control[line][primary]["status"])
        replication_status = replication[line]["status"]
        if MISSING in (*validity, replication_status):
            return MISSING
        if FAIL in validity:
            return VOID
        return str(replication_status)

    confirmatory = {line: classify(line) for line in plan["confirmatory_cell_lines"]}
    statuses = list(confirmatory.values())
    if statuses and all(status == PASS for status in statuses):
        verdict = "replicated"
    elif any(status == PASS for status in statuses):
        verdict = "partially replicated"
    elif any(status == FAIL for status in statuses):
        verdict = "not replicated"
    else:
        verdict = "not evaluable"

    return {
        "plan_id": plan["id"],
        "verdict": verdict,
        "confirmatory": confirmatory,
        "exploratory": {line: classify(line) for line in plan["exploratory_cell_lines"]},
        "criteria": {
            "baseline_detection": baseline,
            "negative_control": control,
            "replication": replication,
            "segmenter_generality": check_segmenter_generality(plan, summary),
            "threshold_robustness": check_threshold_robustness(plan, summary),
            "dose_response": check_dose_response(plan, summary),
        },
        "verdict_rule": plan.get("verdict_rule"),
        "note": (
            "Exploratory cell lines are scored with the same criteria and reported, "
            "but never enter the verdict."
        ),
    }


def merge_artifacts(
    paths: list[Path], plan_path: str | Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Combine per-cell-line artifacts, refusing any not run under this exact plan."""
    if not paths:
        raise PreregistrationError("there are no artifacts to evaluate.")
    expected = provenance.sha256_file(Path(plan_path))
    merged: dict[str, Any] = {}
    sources: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        recorded = (payload.get("design") or {}).get("preregistration")
        if not recorded:
            raise PreregistrationError(
                f"{path} records no plan, so it was not run under one and cannot be scored "
                "against it."
            )
        if recorded.get("sha256") != expected:
            raise PreregistrationError(
                f"{path} was run under a different version of the plan (recorded "
                f"{str(recorded.get('sha256'))[:12]}, current {expected[:12]}). A plan edited "
                "after a run does not re-score that run; run the experiment again."
            )
        for key, row in payload["summary"].items():
            if key in merged:
                raise PreregistrationError(f"{key} appears in more than one artifact.")
            merged[key] = row
        environment = payload.get("environment") or {}
        sources.append(
            {
                "path": _display(Path(path)),
                "recorded_at_utc": environment.get("recorded_at_utc"),
                "git": environment.get("git"),
            }
        )
    return merged, sources
