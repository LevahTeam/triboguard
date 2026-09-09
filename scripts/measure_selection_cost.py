"""What did selecting on one metric cost the other one?

The three-class checkpoint is chosen on validation boundary recall alone, on the
reasoning that boundaries are what separate touching cells. But decoding an
instance needs both: interior probability finds the cells and the boundary splits
them. If the epoch with the best boundary recall is an epoch where interior
recall has not developed, the rule has bought separation with detection, and the
saved checkpoint is worse than one the same run passed through.

This measures that trade directly from the training histories already recorded in
every metrics.json, so it costs no GPU time and no re-training.

Written after noticing that one arm of the diversity experiment selected at epoch
5 while its comparator selected at 22-30. It is post-hoc and labelled so in
docs/PRE_SPECIFICATION.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tribovision import provenance

RUNS = {
    "mixed_42": "runs/diversity/mixed_42",
    "mixed_1": "runs/diversity/mixed_1",
    "mixed_2": "runs/diversity/mixed_2",
    "a172_42": "runs/scale_304",
    "a172_1": "runs/seedrun_304_1",
    "a172_2": "runs/seedrun_304_2",
}


def examine(path: Path) -> dict[str, Any] | None:
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    history = metrics.get("history") or []
    if not history:
        return None
    boundary = np.array([entry["val"]["boundary_recall"] for entry in history])
    interior = np.array([entry["val"]["interior_recall"] for entry in history])
    chosen = int(np.argmax(boundary))
    # The best epoch by the sum of the two recalls: what a rule that valued both
    # would have picked, judged only on data this run already produced.
    balanced = int(np.argmax(boundary + interior))
    return {
        "epochs_run": len(history),
        "selected_epoch": chosen + 1,
        "selected_boundary_recall": float(boundary[chosen]),
        "selected_interior_recall": float(interior[chosen]),
        # How much of the interior recall this run demonstrably reached was given
        # up by the checkpoint that was actually saved.
        "interior_recall_available": float(interior.max()),
        "interior_recall_forfeited": float(interior.max() - interior[chosen]),
        "balanced_epoch": balanced + 1,
        "balanced_boundary_recall": float(boundary[balanced]),
        "balanced_interior_recall": float(interior[balanced]),
    }


def main() -> None:
    report: dict[str, Any] = {
        "selection_metric": "validation boundary recall",
        "note": (
            "balanced_* columns are what the sum of both recalls would have chosen "
            "from the same run. No checkpoint was saved at that epoch, so this "
            "bounds the cost of the rule rather than measuring a better model."
        ),
        "runs": {},
    }
    for name, directory in RUNS.items():
        path = Path(directory)
        if not (path / "metrics.json").is_file():
            continue
        result = examine(path)
        if result:
            report["runs"][name] = result
    if not report["runs"]:
        print("no run histories found", file=sys.stderr)
        raise SystemExit(1)
    report["environment"] = provenance.environment()
    out = Path("runs/diversity/selection_cost.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        f"{'run':10s} {'epochs':>6s} {'sel':>4s} {'bnd':>6s} {'int':>6s} "
        f"{'int max':>8s} {'lost':>6s} {'bal ep':>7s}"
    )
    for name, row in report["runs"].items():
        print(
            f"{name:10s} {row['epochs_run']:6d} {row['selected_epoch']:4d} "
            f"{row['selected_boundary_recall']:6.3f} {row['selected_interior_recall']:6.3f} "
            f"{row['interior_recall_available']:8.3f} {row['interior_recall_forfeited']:6.3f} "
            f"{row['balanced_epoch']:7d}"
        )
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
