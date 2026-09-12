"""Can the study's selectivity claim be explained without a selective compound?

The report compares a dividing cancer line against primary lymphocytes and
concludes the extract spares normal cells. This asks a narrower question that
needs no replicate count and no raw data: **is there a non-selective compound
that reproduces both reported numbers?**

If there is, the claim is not refuted -- the extract may well be selective --
but the experiment cannot distinguish the two, which is the only thing this
project ever claims.

Every figure written here is derived from the numbers already transcribed in the
case-study payload, so the report and this analysis cannot drift apart.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from triboguard import selectivity as sel
from triboguard.kinetics import BirthDeath
from tribovision import provenance

PAPER = Path("case_studies/tribonema_2022.json")
OUT = Path("runs/triboguard/selectivity.json")

#: Doubling times swept for the cancer arm. The report never states one, so no
#: single value is assumed; the headline result below is the *inverse* question,
#: which needs no assumption at all.
DOUBLING_HOURS = (8.0, 12.0, 16.0, 24.0, 32.0, 48.0)

#: Basal death rate for a line in log-phase growth. Its value barely matters
#: here -- everything that follows is driven by the birth rate -- but it is
#: named rather than buried so the assumption is visible.
BASAL_DEATH = 0.007


def _rates_for_doubling(hours: float, death: float = BASAL_DEATH) -> BirthDeath:
    """Birth and death giving a chosen doubling time, at a stated basal death."""
    return BirthDeath(birth=math.log(2.0) / hours + death, death=death)


def main() -> None:
    paper = json.loads(PAPER.read_text(encoding="utf-8"))
    results = paper["reported_results"]
    observed = results["cytotoxicity_at_24h_percent"] / 100.0
    exposure = float(paper["design"]["morphology"]["exposure_hours"])

    # 1. The question that needs no assumption: how fast must a control divide
    #    before stalling alone accounts for the reported number?
    threshold_birth = sel.birth_rate_for_pure_stalling(observed, exposure)

    # 2. What that costs at a range of plausible division rates.
    sweep = []
    for doubling in DOUBLING_HOURS:
        control = _rates_for_doubling(doubling)
        fraction = sel.stalling_fraction(control, observed, exposure)
        sweep.append(
            {
                "control_doubling_hours": doubling,
                "control_birth": control.birth,
                "divisions_that_must_stop": fraction,
                "stalling_alone_suffices": fraction is not None,
                "cells_killed_under_that_reading": 0.0 if fraction is not None else None,
            }
        )

    # 3. One identical, non-selective compound applied to both arms.
    cancer = sel.Population(paper["system"]["cell_line"], _rates_for_doubling(12.0))
    normal = sel.Population("peritoneal lymphocytes", BirthDeath(birth=0.0, death=0.005))
    stall = sel.stalling_fraction(cancer.control, observed, exposure)
    assert stall is not None, "a 12 h doubling must clear the threshold"
    demonstration = sel.illusion(cancer, normal, hours=exposure, birth_scale=1.0 - stall)

    # 4. Apparent selectivity as a function of the *normal* arm's division rate,
    #    which is the variable the compound has nothing to do with.
    contrast = []
    for doubling in (None, 96.0, 48.0, 24.0, 12.0):
        arm = (
            BirthDeath(birth=0.0, death=0.005)
            if doubling is None
            else _rates_for_doubling(doubling, death=0.005)
        )
        row = sel.illusion(
            cancer, sel.Population("normal", arm), hours=exposure, birth_scale=1.0 - stall
        )
        contrast.append(
            {
                "normal_doubling_hours": doubling,
                "apparent_in_cancer": row["arms"][cancer.name]["apparent_cytotoxicity"],
                "apparent_in_normal": row["arms"]["normal"]["apparent_cytotoxicity"],
                "apparent_gap": row["apparent_gap"],
            }
        )

    # 5. Everything consistent with the reported number, end to end.
    family = sel.explaining_family(cancer.control, observed, exposure, steps=6)

    # 6. Does one unchanging treatment reproduce the reported time course?
    points = [
        (float(p["hours"]), p["cytotoxicity_percent"] / 100.0)
        for p in results["time_course_percent"]
    ]
    points.append((exposure, observed))
    course = sel.time_course_consistency(sorted(points))

    report: dict[str, Any] = {
        "question": (
            "Is there a compound with no selective action that reproduces the "
            "cytotoxicity reported in the cancer arm together with the absence of "
            "cytotoxicity reported in the normal arm?"
        ),
        "answer": "yes",
        "what_it_does_not_reproduce": (
            "The normal arm was not merely unharmed: its viability was reported as "
            "more than doubled. No cell-number model produces a signal above the "
            "untreated control, so that half of the reported result is left "
            "unexplained here rather than accounted for."
        ),
        "observed": {
            "cytotoxicity": observed,
            "hours": exposure,
            "normal_arm_as_reported": results["lymphocyte_effect"],
            "comparator": paper["system"]["normal_comparator"],
        },
        "threshold": {
            "birth_rate": threshold_birth,
            "mean_hours_between_divisions": 1.0 / threshold_birth,
            "doubling_hours_if_nothing_died": math.log(2.0) / threshold_birth,
            "units_note": (
                "The threshold is a division-event rate, not a growth rate. It is "
                "stated that way because the two coincide only when basal death is "
                "zero: a population doubling slowly while dividing and dying quickly "
                "can clear this bar, and one doubling faster than "
                f"{math.log(2.0) / threshold_birth:.1f} h can fail it if it never dies."
            ),
            "statement": (
                f"A control whose cells divide at least {threshold_birth:.4f} times per hour "
                f"-- one division per cell every {1.0 / threshold_birth:.0f} h on average -- "
                f"can produce the reported {observed:.0%} in {exposure:.0f} h with no cell "
                f"death at all. This needs no replicate count and no assumed growth rate."
            ),
        },
        "sweep": sweep,
        "demonstration": demonstration,
        "contrast_is_the_division_rate": contrast,
        "explaining_family": family,
        "time_course": course,
        "what_this_does_not_show": [
            "It does not reproduce the reported rise in lymphocyte viability, only "
            "the reported absence of cytotoxicity there. The rise is outside every "
            "model used here.",
            "It does not show the extract is inactive. The reported effect stands.",
            "It does not show the extract is non-selective. It shows the design "
            "cannot separate a selective compound from a cytostatic one.",
            "MTS absorbance is treated as proportional to viable cell number. A "
            "compound that lowers metabolism per cell produces the same fall, which "
            "is a second route to the same wrong conclusion and is not modelled here.",
            "The normal arm's reported viability rose rather than fell. Under a "
            "cell-number reading that is negative cytotoxicity, which needs either "
            "proliferation or raised metabolism to explain and cannot be read as "
            "'the extract is safe' without distinguishing them. That reading is the "
            "one piece of the reported result this analysis does not account for.",
        ],
        "environment": provenance.environment(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(report["threshold"]["statement"])
    print()
    print("One identical compound, both arms, nothing killed anywhere:")
    for name, arm in demonstration["arms"].items():
        print(
            f"  {name:24s} apparent {arm['apparent_cytotoxicity']:6.1%}   "
            f"actually killed {arm['cells_actually_killed']:5.1%}"
        )
    print()
    print("Reported time course under one unchanging treatment:")
    for point in course["points"]:
        print(
            f"  {point['hours']:>4.0f} h  {point['cytotoxicity']:5.0%}  "
            f"implies net change {point['implied_net_change']:+.4f}/h"
        )
    print(
        f"  spread {course['implied_rate_spread']:.2f}x  "
        f"consistent with a constant-rate effect: {course['constant_rate_consistent']}"
    )
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()
