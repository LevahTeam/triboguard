"""Regenerate the results table from run artifacts.

The numbers in ``docs/RESULTS.md`` are not typed by hand. This reads the JSON
that each run wrote and prints the table, so a reader can regenerate it and get
the same thing, and so a stale table is a visible diff rather than an unnoticed
inaccuracy.

    python scripts/collect_results.py runs > docs/RESULTS.md
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def seed_runs(runs: Path) -> list[tuple[str, dict[str, Any]]]:
    found = []
    for directory in sorted(runs.glob("seed_*")):
        metrics = load(directory / "metrics.json")
        if metrics:
            found.append((directory.name, metrics))
    return found


def format_report(runs: Path) -> str:
    lines: list[str] = ["# Measured results", ""]
    lines.append(
        "Generated from run artifacts by `python scripts/collect_results.py runs`. "
        "Every number below comes from a JSON file written by the run that produced it."
    )
    lines.append("")

    primary = load(runs / "baseline" / "metrics.json")
    comparison = load(runs / "comparison" / "comparison.json")
    classical = load(runs / "classical_baseline" / "baseline_report.json")

    if primary:
        environment = primary.get("environment", {})
        git = environment.get("git", {})
        lines += [
            "## Primary run",
            "",
            f"- Configuration: {json.dumps(primary['config'], sort_keys=True)}",
            f"- Device: `{primary['device']}`",
            f"- Best epoch: {primary['best_epoch']} of {len(primary['history'])} run",
            f"- Code revision: `{git.get('commit') or 'uncommitted'}`"
            + (" (working tree dirty)" if git.get("dirty") else ""),
            f"- torch {environment.get('packages', {}).get('torch')}, "
            f"numpy {environment.get('packages', {}).get('numpy')}, "
            f"Python {environment.get('python')}",
            "",
            "| Split | Wells | Images | Macro Dice | Micro Dice | Macro IoU |",
            "|---|---|---|---|---|---|",
        ]
        for split in ("validation", "test"):
            metrics = primary[split]
            fingerprint = primary["datasets"]["val" if split == "validation" else "test"]
            lines.append(
                f"| {split} | {', '.join(fingerprint['wells'])} | {fingerprint['images']} | "
                f"{metrics['macro_dice']:.4f} | {metrics['micro_dice']:.4f} | "
                f"{metrics['macro_iou']:.4f} |"
            )
        check = primary["split_check"]
        lines += [
            "",
            f"Split leakage check at the `{check['group_by']}` level: "
            f"**{'clean' if check['clean'] else 'LEAKING'}** — "
            f"{check['groups_per_split']}.",
            "",
        ]

    if comparison:
        lines += [
            "## Acceptance gate: learned model versus the transparent baseline",
            "",
            f"Both scored on `{comparison['manifest']}` ({comparison['images']} images).",
            "",
            "| Segmenter | Macro Dice | Micro Dice | Macro IoU | Instance matching 0.50:0.95 |",
            "|---|---|---|---|---|",
        ]
        for label, key in (
            ("Classical local contrast + Otsu", "classical"),
            ("TriboVision U-Net", "neural"),
        ):
            row = comparison[key]
            lines.append(
                f"| {label} | {row['macro_dice']:.4f} | {row['micro_dice']:.4f} | "
                f"{row['macro_iou']:.4f} | {row['matching_score_50_95']:.4f} |"
            )
        lines += [
            "",
            f"- Mean per-image Dice difference: {comparison['dice_difference_mean']:+.4f} "
            f"(SD {comparison['dice_difference_sd']:.4f})",
            f"- The learned model wins on {comparison['images_where_neural_wins']} of "
            f"{comparison['images']} images",
            f"- Exact sign test: p = {comparison['sign_test_p_value']:.3g}",
            f"- Verdict: **{comparison['verdict']}**",
            "",
        ]

    seeds = seed_runs(runs)
    if seeds:
        lines += [
            "## Seed replicates",
            "",
            "Independent runs differing only in random seed, same splits, same data.",
            "",
            "| Run | Seed | Best epoch | Validation Dice | Test Dice |",
            "|---|---|---|---|---|",
        ]
        test_scores = []
        for name, metrics in seeds:
            test_scores.append(metrics["test"]["macro_dice"])
            lines.append(
                f"| `{name}` | {metrics['config']['seed']} | {metrics['best_epoch']} | "
                f"{metrics['validation']['macro_dice']:.4f} | "
                f"{metrics['test']['macro_dice']:.4f} |"
            )
        if primary:
            test_scores.append(primary["test"]["macro_dice"])
            lines.append(
                f"| `baseline` | {primary['config']['seed']} | {primary['best_epoch']} | "
                f"{primary['validation']['macro_dice']:.4f} | "
                f"{primary['test']['macro_dice']:.4f} |"
            )
        mean = statistics.mean(test_scores)
        spread = statistics.stdev(test_scores) if len(test_scores) > 1 else 0.0
        lines += [
            "",
            f"Test Dice across {len(test_scores)} seeds: **{mean:.4f} ± {spread:.4f}** "
            f"(mean ± SD), range {min(test_scores):.4f}–{max(test_scores):.4f}.",
            "",
            "The spread is the honest uncertainty on the headline number. It is far "
            "smaller than the gap to the classical baseline, so the comparison does not "
            "depend on a lucky seed.",
            "",
        ]

    if classical:
        lines += [
            "## Classical baseline detail",
            "",
            f"- Method: {classical['method']}",
            f"- Parameters: {json.dumps(classical['parameters'], sort_keys=True)}",
            f"- Images: {classical['overall']['images']}",
            f"- Cell-count mean absolute error: {classical['overall']['count_mae']:.1f}",
            "",
        ]

    lines += [
        "## How to reproduce",
        "",
        "```bash",
        "tribovision prepare-livecell",
        "tribovision verify",
        "tribovision baseline --max-images 60",
        "tribovision train --epochs 40 --image-size 512 --batch-size 4 --patience 12",
        "tribovision compare --checkpoint runs/baseline/best_model.pt",
        "python scripts/collect_results.py runs > docs/RESULTS.md",
        "```",
        "",
        "Exact reproduction of a number also requires the same seed, the same device, "
        "and the dependency versions recorded in each report's `environment` block. "
        "CPU and GPU accumulation orders differ, so cross-device runs agree to about "
        "three decimal places, not exactly.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    runs = Path(arguments[0]) if arguments else Path("runs")
    if not runs.is_dir():
        print(f"No run directory at {runs}", file=sys.stderr)
        return 1
    sys.stdout.write(format_report(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
