"""Read a reported count through the segmentation blindness the sweep measured.

    python scripts/apply_blindness.py

**Exploratory**, and labelled so in its output: this analysis was not part of the
pre-registered plan. It asks one question of the measured blindness. Suppose the
source study's 40% fall had been measured as an image-based cell count instead
of by MTS -- which is what switching to imaging would mean. What fraction of the
cells could that count establish as genuinely absent?

The blindness of a segmenter at one severity is taken as a range, not a number:
the 95% interval bounds on the per-image detection gap, each divided by that
line's mean intact detection rate, across all four cell lines. Dividing by the
mean treats the intact rate as fixed; it varies far less than the gap does. And
taking the extremes across lines makes the range wider than any single line's
interval, so the construction errs toward abstaining rather than toward an answer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from triboguard import blindness
from tribovision import preregistration, provenance

PAPER = Path("case_studies/tribonema_2022.json")
PLAN = Path("preregistration/damage_blindness_v1.json")
ARTIFACTS = Path("runs/damage")
OUT = Path("runs/damage/blindness_applied.json")
SEVERITIES = (0.2, 0.4, 0.6)
COVERAGE = 0.5


def _blindness_range(
    summary: dict[str, Any], plan: dict[str, Any], segmenter: str, severity: float
) -> tuple[float, float]:
    bounds: list[float] = []
    for line in plan["confirmatory_cell_lines"] + plan["exploratory_cell_lines"]:
        row = summary[preregistration.summary_key(line, severity, segmenter, COVERAGE)]
        intact = float(row["mean_intact_rate"])
        if intact <= 0.0:
            raise ValueError(f"{line}/{segmenter} found no intact cells; blindness is undefined.")
        interval = row["paired_difference"]
        for edge in (interval["ci_low"], interval["ci_high"]):
            bounds.append(min(1.0, max(0.0, float(edge) / intact)))
    return min(bounds), max(bounds)


def main() -> None:
    plan = preregistration.load_plan(PLAN)
    paper = json.loads(PAPER.read_text(encoding="utf-8"))
    ratio = 1.0 - paper["reported_results"]["cytotoxicity_at_24h_percent"] / 100.0
    # merge_artifacts refuses any result scored under a different plan, so this
    # exploratory reading rests on exactly the data the verdict was given.
    summary, sources = preregistration.merge_artifacts(
        sorted(ARTIFACTS.glob("blindness_combined_*.json")), PLAN
    )

    rows: list[dict[str, Any]] = []
    for segmenter in plan["design"]["segmenters"]:
        for severity in SEVERITIES:
            low, high = _blindness_range(summary, plan, segmenter, severity)
            assessed = blindness.assess(ratio, ratio, low, high)
            rows.append(
                {"segmenter": segmenter, "severity": severity, "blindness": [low, high], **assessed}
            )

    report = {
        "exploratory": True,
        "question": (
            "If the study's reported fall had been measured as an image-based cell count, "
            "what fraction of cells could that count establish as genuinely absent, given "
            "the blindness each segmenter showed?"
        ),
        "counted_ratio": ratio,
        "construction": (
            "Blindness range: the 95% interval bounds on the per-image detection gap, each "
            "divided by that line's mean intact rate, taken across all four cell lines at "
            f"coverage threshold {COVERAGE}. The study measured MTS, not a count; this is a "
            "hypothetical reading of a count of the same size."
        ),
        "rows": rows,
        "plan": preregistration.plan_fingerprint(PLAN),
        "artifacts": sources,
        "environment": provenance.environment(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"A count showing {1 - ratio:.0%} fewer cells, read through measured blindness:\n")
    for row in rows:
        low, high = row["blindness"]
        print(
            f"  {row['segmenter']:15s} severity {row['severity']:.1f}  blindness "
            f"[{low:.2f}, {high:.2f}]  absent [{row['absent_low']:.0%}, {row['absent_high']:.0%}]"
            f"  {row['verdict']}"
        )
    print(f"\nWROTE {OUT}  (exploratory)")


if __name__ == "__main__":
    main()
