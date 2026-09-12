"""Regenerate every number the TriboGuard report quotes, in dependency order.

    python scripts/reproduce.py --list      # the stages, what each makes, what each needs
    python scripts/reproduce.py --fast      # everything except the damage sweep
    python scripts/reproduce.py             # everything
    python scripts/reproduce.py --only selectivity case-study

Each stage runs a script that writes a JSON artifact recording the code revision
and package versions that produced it. The run ends by checking that every
artifact the report cites exists, so a missing number is a failed command rather
than a sentence nobody can check.

A stage whose inputs are missing says which stage makes them, or -- for inputs no
stage makes, like trained checkpoints -- where to get them, instead of crashing
halfway through. The results table is only replaced when the command that
regenerates it succeeds, so a failure cannot leave it half-written.

**What this does not regenerate:** the segmentation models. Training takes hours
per model and runs on different devices agree to about three decimal places, not
exactly. Their commands are in ``docs/RESULTS.md``; the damage sweep here uses the
trained checkpoints as they are.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPORT = Path("demo/triboguard_report.html")
PREREGISTRATION = "preregistration/damage_blindness_v1.json"

#: The cell lines the pre-registered plan covers, confirmatory and exploratory.
#: A test checks this against the plan so the two cannot disagree.
DAMAGE_LINES = ("A172", "MCF7", "SHSY5Y", "SkBr3")
MANIFESTS = {
    "A172": "data/livecell/manifests/test.jsonl",
    "MCF7": "data/livecell/manifests_mcf7/test.jsonl",
    "SHSY5Y": "data/livecell/manifests_shsy5y/test.jsonl",
    "SkBr3": "data/livecell/manifests_skbr3/test.jsonl",
}


@dataclass(frozen=True)
class Stage:
    """One step: the scripts it runs, what it writes, and what it reads."""

    name: str
    commands: tuple[tuple[str, ...], ...]
    outputs: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    stdout_to: str | None = None
    needs_cellpose: bool = False
    slow: bool = False
    note: str = ""


def _sweep_output(line: str) -> str:
    return f"runs/damage/blindness_combined_{line}.json"


STAGES: tuple[Stage, ...] = (
    Stage(
        "browser-tables",
        (("scripts/generate_chi2_table.py",), ("scripts/generate_parity_fixtures.py",)),
        outputs=("web/lib/chi2.ts",),
        note="Quantile tables and parity fixtures the website checks its arithmetic against.",
    ),
    Stage(
        "failure-boundary",
        (("scripts/run_failure_boundary.py",),),
        outputs=("runs/triboguard/failure_boundary.json",),
    ),
    Stage(
        "case-study",
        (("scripts/run_case_study.py",),),
        outputs=("runs/triboguard/tribonema_case_study.json",),
        requires=("case_studies/tribonema_2022.json",),
    ),
    Stage(
        "selectivity",
        (("scripts/run_selectivity.py",),),
        outputs=("runs/triboguard/selectivity.json",),
        requires=("case_studies/tribonema_2022.json",),
    ),
    Stage(
        "damage-sweep",
        tuple(
            (
                "scripts/run_damage_blindness.py",
                "--axis",
                "combined",
                "--cell-lines",
                line,
                "--out",
                _sweep_output(line),
            )
            for line in DAMAGE_LINES
        ),
        outputs=tuple(_sweep_output(line) for line in DAMAGE_LINES),
        requires=(
            "runs/scale_304/best_model.pt",
            "runs/baseline/best_model.pt",
            PREREGISTRATION,
            *(MANIFESTS[line] for line in DAMAGE_LINES),
        ),
        needs_cellpose=True,
        slow=True,
        note=(
            "Roughly an hour per cell line on a laptop GPU. One process per line, "
            "so an interruption keeps every line already finished."
        ),
    ),
    Stage(
        "preregistered-verdict",
        (("scripts/evaluate_preregistration.py",),),
        outputs=("runs/damage/preregistered_outcome.json",),
        requires=tuple(_sweep_output(line) for line in DAMAGE_LINES),
    ),
    Stage(
        "blindness-applied",
        (("scripts/apply_blindness.py",),),
        outputs=("runs/damage/blindness_applied.json",),
        requires=(
            *(_sweep_output(line) for line in DAMAGE_LINES),
            "case_studies/tribonema_2022.json",
        ),
        note=(
            "Exploratory: the study's fall read through the measured blindness. Not pre-registered."
        ),
    ),
    Stage(
        "results-table",
        (("scripts/collect_results.py", "runs"),),
        outputs=("docs/RESULTS.md",),
        stdout_to="docs/RESULTS.md",
    ),
    Stage(
        "publish",
        (("scripts/collect_results.py", "runs", "--publish", "results"),),
        outputs=("results/README.md",),
        note="Copies the small JSON artifacts into the tracked results/ directory.",
    ),
)


def producer_of(path: str) -> Stage | None:
    """The stage that writes ``path``, if any does."""
    for stage in STAGES:
        if path in stage.outputs:
            return stage
    return None


def select(only: list[str] | None, fast: bool) -> list[Stage]:
    """The stages to run, in dependency order whatever order they were named in."""
    names = {stage.name for stage in STAGES}
    if only:
        unknown = sorted(set(only) - names)
        if unknown:
            raise ValueError(f"no stage named {unknown}; --list shows them.")
        chosen = [stage for stage in STAGES if stage.name in only]
    else:
        chosen = list(STAGES)
    if fast:
        chosen = [stage for stage in chosen if not stage.slow]
    return chosen


def cited_artifacts(report: Path = REPORT) -> list[str]:
    """Every run artifact the report names as a source."""
    text = (REPO / report).read_text(encoding="utf-8")
    return sorted(set(re.findall(r"runs/[A-Za-z0-9_./-]+\.json", text)))


def _cellpose_available() -> bool:
    return importlib.util.find_spec("cellpose") is not None


def _execute(stage: Stage, argv: list[str], env: dict[str, str]) -> int:
    if stage.stdout_to is None:
        return subprocess.run(argv, cwd=REPO, env=env).returncode
    # Write beside the target and swap it in only on success: a failing command
    # must not leave the table it was regenerating truncated.
    target = REPO / stage.stdout_to
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, suffix=".partial")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as sink:
            code = subprocess.run(argv, cwd=REPO, env=env, stdout=sink).returncode
        if code == 0:
            os.replace(temporary, target)
        return code
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run(stages: list[Stage], *, dry_run: bool = False) -> int:
    """Run each stage in order; return how many failed."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO / "src"), env.get("PYTHONPATH", "")) if part
    )
    selected = {stage.name for stage in stages}
    failures = 0
    for stage in stages:
        missing = [path for path in stage.requires if not (REPO / path).exists()]
        if missing:
            makers = [producer_of(path) for path in missing]
            if all(maker is not None and maker.name not in selected for maker in makers):
                upstream = sorted({maker.name for maker in makers if maker is not None})
                print(f"SKIP  {stage.name}: needs output of {upstream}, not run this time")
                continue
            for path, maker in zip(missing, makers, strict=True):
                source = (
                    f"made by stage '{maker.name}'"
                    if maker is not None
                    else "made by no stage here; see docs/RESULTS.md"
                )
                print(f"FAIL  {stage.name}: missing {path} ({source})")
            failures += 1
            continue
        if stage.needs_cellpose and not _cellpose_available():
            print(
                f"FAIL  {stage.name}: needs Cellpose in {sys.executable}; "
                "install it with: pip install -e '.[cellpose]'"
            )
            failures += 1
            continue
        for command in stage.commands:
            argv = [sys.executable, *command]
            shown = " ".join(command) + (f" > {stage.stdout_to}" if stage.stdout_to else "")
            print(f"RUN   {stage.name}: {shown}", flush=True)
            if dry_run:
                continue
            code = _execute(stage, argv, env)
            if code != 0:
                print(f"FAIL  {stage.name}: exited with {code}")
                failures += 1
                break
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the report's evidence.")
    parser.add_argument("--list", action="store_true", help="show the stages and exit")
    parser.add_argument("--fast", action="store_true", help="skip the slow stages")
    parser.add_argument("--only", nargs="+", default=None, help="run only these stages")
    parser.add_argument("--dry-run", action="store_true", help="show what would run")
    args = parser.parse_args(argv)

    if args.list:
        for stage in STAGES:
            flags = ", ".join(
                flag
                for flag, on in (("slow", stage.slow), ("needs Cellpose", stage.needs_cellpose))
                if on
            )
            print(f"{stage.name}{f'  [{flags}]' if flags else ''}")
            for output in stage.outputs:
                print(f"    writes {output}")
            if stage.note:
                print(f"    {stage.note}")
        return 0

    try:
        stages = select(args.only, args.fast)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    failures = run(stages, dry_run=args.dry_run)
    if args.dry_run:
        return 0
    absent = [path for path in cited_artifacts() if not (REPO / path).exists()]
    for path in absent:
        print(f"MISSING  the report cites {path}, which does not exist")
    return 1 if failures or absent else 0


if __name__ == "__main__":
    sys.exit(main())
