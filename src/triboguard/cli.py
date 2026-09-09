"""Command line for TriboGuard.

The package was reachable only by importing it from a script that first pushed
``src`` onto ``sys.path``, which works from the repository root and nowhere
else. An installed entry point makes the three things a user actually wants --
"can my design answer this", "what does this paper's design support", and "where
does the method stop working" -- available from any directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from triboguard import boundary, case_study, design


def _assay(name: str) -> design.Assay:
    """The two assay shapes that differ in what a timepoint costs."""
    if name == "mts":
        return design.Assay(per_well=0.80, per_measurement=0.35, destructive=True, name="MTS")
    return design.Assay(
        per_well=0.80, per_measurement=0.20, destructive=False, name="live-cell imaging"
    )


def _design_command(args: argparse.Namespace) -> int:
    times = tuple(float(value) for value in args.times)
    current = design.Design(wells=args.wells, times=times)
    assay = _assay(args.assay)
    report = design.recommend(current, assay, target_width=args.target_width)
    report["turnover_ratio"] = design.turnover_ratio(args.wells)
    report["wells_for_a_factor_of_two"] = design.wells_for_ratio(2.0)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    current_state = report["current"]
    print(f"{args.wells} wells x {len(times)} timepoints, {assay.name}")
    print(f"  wells consumed        {current_state['wells_consumed']}")
    print(f"  cost                  {current_state['cost']:.2f}")
    ratio = report["turnover_ratio"]
    width = current_state["relative_turnover_width"]
    # Two different quantities, printed with their own names. Side by side and
    # unlabelled they read as a contradiction, which is how they came to be
    # conflated in the first place.
    print(f"  split known to within x{ratio:,.1f}  (ratio of the interval's ends)")
    if width is not None:
        print(f"  relative interval width {width:,.2f}  (its span over the estimate)")
    print(f"  a factor of two needs   {report['wells_for_a_factor_of_two']} wells per condition")
    print(f"\n{report['statement']}")
    return 0


def _case_study_command(args: argparse.Namespace) -> int:
    report = case_study.report(args.path, assumed_wells=args.assumed_wells)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    print(f"{report['source']['title']}\n")
    for claim in report["claims"]:
        mark = "supported    " if claim["supported"] else "not supported"
        print(f"  [{mark}] {claim['claim']}")
    print(f"\n  {report['supported']} supported, {report['unsupported']} not established")
    print(f"\n{report['statement']}")
    # A case study that supports nothing is usually a broken input rather than a
    # devastating verdict, so it exits non-zero to be noticed in a pipeline.
    return 0 if report["supported"] else 1


def _boundary_command(args: argparse.Namespace) -> int:
    report = boundary.sweep(
        wells=tuple(args.wells),
        effects=tuple(args.effects),
        calibration_experiments=args.calibration,
        test_experiments=args.test,
        resamples=args.resamples,
        seed=args.seed,
        progress=not args.json,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    print(f"\n{json.dumps(report['summary'], indent=2)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triboguard",
        description=(
            "Decide whether a cell-treatment experiment can tell killing from growth "
            "inhibition, and what the answer would cost."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser(
        "design", help="What can this design constrain, and at what price?"
    )
    plan.add_argument("--wells", type=int, default=3, help="wells per condition")
    plan.add_argument(
        "--times", nargs="+", default=["0", "24", "48"], help="exposure times in hours"
    )
    plan.add_argument("--assay", choices=("mts", "imaging"), default="mts")
    plan.add_argument("--target-width", type=float, default=1.0)
    plan.add_argument("--json", action="store_true")
    plan.set_defaults(handler=_design_command)

    study = subparsers.add_parser(
        "case-study", help="Which conclusions a published design can carry."
    )
    study.add_argument("path", type=Path, help="a case-study JSON description")
    study.add_argument("--assumed-wells", type=int, default=3)
    study.add_argument("--json", action="store_true")
    study.set_defaults(handler=_case_study_command)

    sweep = subparsers.add_parser(
        "boundary", help="Where the method is reliable, abstains, or misleads."
    )
    sweep.add_argument("--wells", type=int, nargs="+", default=[3, 6, 12, 24])
    sweep.add_argument("--effects", type=float, nargs="+", default=[0.25, 0.40, 0.70])
    sweep.add_argument("--calibration", type=int, default=40)
    sweep.add_argument("--test", type=int, default=40)
    sweep.add_argument("--resamples", type=int, default=100)
    sweep.add_argument("--seed", type=int, default=1)
    sweep.add_argument("--json", action="store_true")
    sweep.set_defaults(handler=_boundary_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Any = args.handler
    try:
        return int(handler(args))
    except (design.DesignError, case_study.CaseStudyError, boundary.BoundaryError) as error:
        # A refusal is the tool working. It belongs on stderr with a clear
        # message, not as a traceback the caller has to read backwards.
        print(f"triboguard: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())
