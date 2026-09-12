"""Applying the finished system to the study that motivated it.

The Tribonema report is the case study, not the training data. It contains a few
dozen numbers, which is far too few to fit or calibrate anything, and the
numbers it does contain are the wrong *kind*: MTS reports mitochondrial
dehydrogenase activity, and the model here is written in cells. So this does not
estimate rates from the paper, and saying otherwise would be the exact
overreach the project exists to prevent.

What it does instead is decide, claim by claim, whether the design could have
supported the conclusion drawn from it. Each claim declares what it requires;
the design declares what it provides; support is set inclusion. That is
mechanical and auditable, which matters because the alternative -- a considered
opinion about somebody else's paper -- is not checkable by anyone.

The one quantitative statement made here is about the design rather than the
data: given a well count, how tightly could *any* experiment of this shape
constrain the mechanism? That question needs no measurements at all, only the
chi-squared width from :mod:`triboguard.design`, which is why it can be answered
for a study whose replicate count was never reported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from triboguard import design, selectivity

#: Capabilities a design can offer. A claim is supportable only if every
#: capability it requires is present.
CAPABILITIES = (
    "metabolic_readout",
    "cell_counts",
    "replicate_variance",
    "single_cell_resolution",
    "longitudinal_same_wells",
    "death_markers",
    "matched_untreated_control",
    "matched_normal_population",
    "positive_control_drug",
    "morphology_snapshot",
    "dose_series",
    "time_series",
    "recovery_test",
)

#: Why a missing capability blocks a claim, in words a reader can check against
#: the paper rather than take on trust.
WHY_MISSING = {
    "cell_counts": (
        "MTS reports mitochondrial dehydrogenase activity, not cell number. A 40% "
        "fall in signal is consistent with 40% of cells dying, with every cell "
        "surviving at 60% of its former metabolic rate, and with any mixture of "
        "the two"
    ),
    "replicate_variance": (
        "telling killing from growth inhibition needs the spread between wells, "
        "and the report states no replicate count anywhere"
    ),
    "death_markers": (
        "apoptosis and necrosis are distinguished by membrane integrity and "
        "caspase activity, which methylene blue on formalin-fixed cells cannot "
        "show; fixation also destroys the membrane-integrity evidence"
    ),
    "recovery_test": (
        "a surviving fraction is only 'resistant' if it stays alive on re-exposure, "
        "which was not tested"
    ),
    "matched_normal_population": (
        "the normal comparator is peritoneal lymphocytes, a different lineage in "
        "primary culture, so the contrast confounds cancer-versus-normal with "
        "macrophage-versus-lymphocyte and cell-line-versus-primary. The confound "
        "is quantitative rather than merely conceptual: the cancer arm divides "
        "and the lymphocytes largely do not, and because cytotoxicity is scored "
        "against a same-time control, a compound that only stops division scores "
        "in the first arm and not in the second"
    ),
    "single_cell_resolution": (
        "the published figure is a representative field, not a measured population"
    ),
    "longitudinal_same_wells": (
        "MTS destroys the well it reads, so no well contributes to two timepoints"
    ),
}


class CaseStudyError(ValueError):
    """Raised when a case-study description is missing something it must state."""


def load(path: str | Path) -> dict[str, Any]:
    """Read a case-study description and check it is self-consistent."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("source", "design", "provides", "claims", "not_reported"):
        if key not in payload:
            raise CaseStudyError(f"case study is missing required section {key!r}.")
    unknown = set(payload["provides"]) - set(CAPABILITIES)
    if unknown:
        raise CaseStudyError(f"unknown capabilities in provides: {sorted(unknown)}.")
    for claim in payload["claims"]:
        missing = set(claim["requires"]) - set(CAPABILITIES)
        if missing:
            raise CaseStudyError(f"claim {claim['id']!r} requires unknown {sorted(missing)}.")
    return payload


def assess_claims(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Which conclusions the design could carry, and which it could not."""
    provided = set(payload["provides"])
    assessed: list[dict[str, Any]] = []
    for claim in payload["claims"]:
        required = set(claim["requires"])
        missing = sorted(required - provided)
        assessed.append(
            {
                "id": claim["id"],
                "claim": claim["text"],
                "supported": not missing,
                "missing_capabilities": missing,
                "reasons": [
                    WHY_MISSING.get(name, f"the design does not provide {name}") for name in missing
                ],
                "verdict": (
                    "supported by this design" if not missing else "not established by this design"
                ),
            }
        )
    return assessed


def resolving_power(
    well_counts: tuple[int, ...] = (2, 3, 4, 6, 8, 12, 24, 48),
    *,
    confidence: float = 0.95,
) -> list[dict[str, Any]]:
    """What any experiment of this shape could constrain, per well count.

    Answerable without the paper's numbers because the width depends on the
    well count alone. That is the only reason a study which never reported its
    replicate count can be analysed at all.
    """
    rows = []
    for wells in well_counts:
        width = design.relative_turnover_width(wells, confidence)
        ratio = design.turnover_ratio(wells, confidence)
        rows.append(
            {
                "wells": wells,
                # Reported side by side and never interchangeably. The ratio is
                # what "known to within a factor of" means; the width is the
                # interval's span over the estimate. At three wells they are 146
                # and 39, and an earlier version printed the second under the
                # first one's name.
                "turnover_ratio": ratio if ratio < 1e6 else None,
                "relative_turnover_width": width if width < 1e6 else None,
                "mechanism_estimable": wells >= design.MINIMUM_USEFUL_WELLS,
            }
        )
    return rows


def follow_up(
    payload: dict[str, Any],
    *,
    assumed_wells: int = 3,
    target_width: float = 1.0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """The experiment that would settle what the original could not.

    ``assumed_wells`` is a stated assumption, not a measurement: the report does
    not say how many wells it used. Three is the smallest number a reader would
    charitably assume from a plate experiment, so it is the most favourable
    reading, and the conclusion below survives it.
    """
    times = tuple(float(hour) for hour in payload["design"]["times_hours"])
    if len(times) < 2:
        raise CaseStudyError("a follow-up needs at least two timepoints to plan around.")
    mts = design.Assay(
        per_well=0.80, per_measurement=0.35, destructive=True, name=payload["design"]["assay"]
    )
    imaging = design.Assay(
        per_well=0.80, per_measurement=0.20, destructive=False, name="live-cell imaging"
    )
    current = design.Design(wells=assumed_wells, times=times)
    return {
        "assumed_wells": assumed_wells,
        "assumption_note": (
            "the report states no replicate count; three is assumed as the most "
            "favourable reading a plate experiment would support"
        ),
        "as_run_with_mts": design.recommend(
            current, mts, target_width=target_width, confidence=confidence
        ),
        "switched_to_imaging": design.recommend(
            current, imaging, target_width=target_width, confidence=confidence
        ),
    }


def selectivity_check(payload: dict[str, Any]) -> dict[str, Any]:
    """Could a compound with no selective action produce the reported contrast?

    Deliberately narrow. It asks only whether an explanation exists that
    involves no selectivity, because that question is answerable from the
    reported percentage and the exposure time alone -- no replicate count, no
    raw data, and no assumed growth rate for either arm.

    The full analysis, including what the contrast looks like when the two arms
    are matched for division rate, lives in :mod:`triboguard.selectivity` and is
    written to its own artifact. Only the assumption-free part is repeated here,
    so the two cannot disagree.
    """
    results = payload.get("reported_results", {})
    observed = results.get("cytotoxicity_at_24h_percent")
    exposure = payload["design"].get("morphology", {}).get("exposure_hours")
    if observed is None or exposure is None:
        return {"evaluated": False, "reason": "No cytotoxicity percentage and exposure reported."}
    fraction = observed / 100.0
    birth = selectivity.birth_rate_for_pure_stalling(fraction, float(exposure))
    return {
        "evaluated": True,
        "observed_cytotoxicity": fraction,
        "hours": float(exposure),
        # A division-event rate, not a growth rate. The two coincide only when
        # basal death is zero, so quoting a doubling time here would exclude
        # slow-growing populations that divide and die quickly and would admit
        # fast-growing ones that never divide enough. The rate is the primitive.
        "threshold_birth_rate": birth,
        "mean_hours_between_divisions": 1.0 / birth,
        "non_selective_explanation_exists": True,
        "statement": (
            f"A compound that kills nothing and only slows division reproduces the "
            f"reported {fraction:.0%} in any control whose cells divide at least "
            f"{birth:.4f} times per hour -- once every {1.0 / birth:.0f} h on average "
            f"-- and reproduces the reported absence of cytotoxicity in a "
            f"non-dividing comparator at the same time. The reported contrast "
            f"therefore does not by itself establish selectivity."
        ),
        "not_reproduced": (
            "The normal arm's viability was reported as more than doubled, not merely "
            "unharmed. No cell-number model produces a signal above the untreated "
            "control, so that part of the reported result is unexplained rather than "
            "accounted for."
        ),
        "does_not_show": (
            "This does not show the extract is non-selective. It shows the design "
            "cannot separate a selective compound from a cytostatic one."
        ),
    }


def report(
    path: str | Path,
    *,
    assumed_wells: int = 3,
    target_width: float = 1.0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """The whole case study: claims, resolving power, and what to run next."""
    payload = load(path)
    claims = assess_claims(payload)
    unsupported = [claim for claim in claims if not claim["supported"]]
    return {
        "source": payload["source"],
        "system": payload.get("system", {}),
        "claims": claims,
        "supported": len(claims) - len(unsupported),
        "unsupported": len(unsupported),
        "not_reported": payload["not_reported"],
        "internal_inconsistencies": payload.get("internal_inconsistencies", []),
        "resolving_power": resolving_power(confidence=confidence),
        "selectivity_check": selectivity_check(payload),
        "follow_up": follow_up(
            payload, assumed_wells=assumed_wells, target_width=target_width, confidence=confidence
        ),
        "statement": _statement(payload, claims, assumed_wells, confidence),
    }


def _statement(
    payload: dict[str, Any], claims: list[dict[str, Any]], assumed_wells: int, confidence: float
) -> str:
    """The finding, written so it cannot be read as a verdict on the biology."""
    unsupported = [claim for claim in claims if not claim["supported"]]
    ratio = design.turnover_ratio(assumed_wells, confidence)
    needed = design.wells_for_ratio(2.0, confidence)
    return (
        f"{payload['design']['assay']} measures "
        f"{payload['design']['assay_measures']}, so the study's own readout cannot "
        "count cells. Of the "
        f"{len(claims)} conclusions examined, {len(claims) - len(unsupported)} are "
        f"supported by the design and {len(unsupported)} are not. At an assumed "
        f"{assumed_wells} wells per condition the birth-to-death split would be pinned "
        f"only to within a factor of about {ratio:.0f}, and narrowing that to a factor "
        f"of two would take roughly {needed} wells per condition. None of this is evidence "
        "against the extract having an effect; the reported effect stands. It is a "
        "statement about which further conclusions the measurements can carry."
    )
