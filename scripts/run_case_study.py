"""Run the finished system against the study that motivated it.

Nothing here is fitted to the paper. The system was built, calibrated and
boundary-mapped on simulated experiments before this was written, and the paper
supplies a design and a set of conclusions to check that design against.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from triboguard import case_study

PAPER = Path("case_studies/tribonema_2022.json")


def main() -> None:
    report = case_study.report(PAPER)
    source = report["source"]
    print(f"{source['title']}\n{source['authors']} — {source['date']}\n")

    print("CONCLUSIONS, AND WHETHER THE DESIGN CAN CARRY THEM")
    for claim in report["claims"]:
        mark = "supported    " if claim["supported"] else "not supported"
        print(f"  [{mark}] {claim['claim']}")
        for reason in claim["reasons"]:
            print(
                textwrap.fill(reason, 88, initial_indent="       - ", subsequent_indent="         ")
            )
    print(f"\n  {report['supported']} supported, {report['unsupported']} not established\n")

    print("WHAT ANY EXPERIMENT OF THIS SHAPE COULD CONSTRAIN")
    print("  (needs none of the paper's numbers: the width depends on the well count alone)")
    print(f"    {'wells':>5s}  {'split known to within':>21s}  {'relative interval width':>23s}")
    for row in report["resolving_power"]:
        ratio, width = row["turnover_ratio"], row["relative_turnover_width"]
        left = "unbounded" if ratio is None else f"x{ratio:,.1f}"
        right = "unbounded" if width is None else f"{width:,.2f}"
        print(f"    {row['wells']:5d}  {left:>21s}  {right:>23s}")

    follow = report["follow_up"]
    mts = follow["as_run_with_mts"]["current"]
    imaging = follow["switched_to_imaging"]["current"]
    print(f"\nCOST OF THE SAME INFORMATION ({follow['assumed_wells']} wells assumed)")
    for label, row in (("MTS (destructive)", mts), ("imaging", imaging)):
        print(f"    {label:18s}: {row['wells_consumed']:3d} wells consumed, cost {row['cost']:.2f}")

    print("\nNOT REPORTED BY THE STUDY")
    for item in report["not_reported"]:
        print(f"    - {item}")

    if report["internal_inconsistencies"]:
        print("\nINTERNAL INCONSISTENCY")
        for item in report["internal_inconsistencies"]:
            print(textwrap.fill(item, 88, initial_indent="    - ", subsequent_indent="      "))

    print("\nSTATEMENT")
    print(textwrap.fill(report["statement"], 88, initial_indent="    ", subsequent_indent="    "))

    out = Path("runs/triboguard/tribonema_case_study.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    from tribovision import provenance

    report["environment"] = provenance.environment()
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWROTE {out}")


if __name__ == "__main__":
    main()
