"""Diversity versus volume again, with a checkpoint rule that values both heads.

The first run of this experiment used the original selection rule: keep the
epoch with the best validation boundary recall. That rule turned out not to be
neutral between the arms. Boundary *recall* rewards over-predicting boundaries,
which an under-trained model does, so it can peak early. On one mixed run it
peaked 29% of the way through training with interior recall still owing 0.18,
where the single-line runs peaked 65-75% of the way through with 0.03-0.05 left.

That happened to one seed, not to every mixed run: another selected an epoch in
the same range as the single-line runs. So the rule is not reliably hostile to
this arm -- it is capable of selecting a badly under-trained checkpoint and did so
once. A three-seed mean containing one such checkpoint is still dragged down by
it, which is reason enough to remove the rule from the comparison.

This re-runs both arms end to end with ``selection_metric="balanced"``, which
averages interior and boundary recall. Both arms are re-trained: reusing the
existing single-line checkpoints would compare a model chosen by one rule against
a model chosen by another, which is the confound this exists to remove.

Everything else is unchanged -- 304 training images per arm, the same three
seeds, the same held-out SHSY5Y manifest that neither arm ever trains on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, "src")

from tribovision import evaluation, provenance  # noqa: E402
from tribovision.instance_model import (  # noqa: E402
    InstanceConfig,
    score_on_manifest,
    train_instance_model,
)

SEEDS = (42, 1, 2)
TRAIN_LIMIT = 304
SELECTION = "balanced"
HELD_OUT = Path("data/livecell/manifests_shsy5y/test.jsonl")
SHSY5Y_CEILING = 0.12150083530255071
OUT = Path("runs/diversity_balanced")
ARMS = {"mixed": "manifests_mixed", "a172_only": "manifests"}


def train(arm: str, seed: int) -> Path:
    run_dir = OUT / f"{arm}_{seed}"
    checkpoint = run_dir / "best_model.pt"
    if checkpoint.exists():
        return checkpoint
    print(f"training {arm} seed={seed}", flush=True)
    train_instance_model(
        InstanceConfig(
            data_dir="data/livecell",
            output_dir=str(run_dir),
            manifests_dirname=ARMS[arm],
            selection_metric=SELECTION,
            train_limit=TRAIN_LIMIT,
            epochs=40,
            batch_size=4,
            learning_rate=1e-3,
            weight_decay=1e-4,
            boundary_weight=3.0,
            base_channels=32,
            depth=3,
            image_size=512,
            patience=12,
            seed=seed,
            device="mps",
        )
    )
    return checkpoint


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scored: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    for seed in SEEDS:
        for arm in ARMS:
            checkpoint = train(arm, seed)
            print(f"scoring {arm} seed={seed} on SHSY5Y", flush=True)
            result = score_on_manifest(checkpoint, HELD_OUT, device="mps")
            scored[arm][seed] = result
            print(f"  {arm:9s} seed={seed} -> {result['matching_50_95']:.4f}", flush=True)

    per_seed = {
        arm: {str(seed): res["per_image"] for seed, res in runs.items()}
        for arm, runs in scored.items()
    }
    groups = scored["mixed"][SEEDS[0]]["groups"]
    report: dict[str, Any] = {
        "design": {
            "train_images_per_arm": TRAIN_LIMIT,
            "held_out_line": "SHSY5Y",
            "held_out_ceiling": SHSY5Y_CEILING,
            "seeds": list(SEEDS),
            "mixed_lines": ["A172", "MCF7", "SkBr3"],
            "selection_metric": SELECTION,
        },
        "per_seed_score": {
            arm: {str(s): r["matching_50_95"] for s, r in runs.items()}
            for arm, runs in scored.items()
        },
        "mixed": evaluation.replicate_interval(per_seed["mixed"], groups=groups),
        "a172_only": evaluation.replicate_interval(per_seed["a172_only"], groups=groups),
        "mixed_minus_a172": evaluation.replicate_difference(
            per_seed["mixed"], per_seed["a172_only"], groups=groups
        ),
        "selected_epochs": {
            f"{arm}_{seed}": json.loads(
                (OUT / f"{arm}_{seed}" / "metrics.json").read_text(encoding="utf-8")
            )["best_epoch"]
            for arm in ARMS
            for seed in SEEDS
        },
        "environment": provenance.environment(),
    }
    for arm in ARMS:
        report[arm]["fraction_of_ceiling"] = report[arm]["mean"] / SHSY5Y_CEILING
    (OUT / "diversity_balanced.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report["mixed_minus_a172"], indent=2))
    print(f"WROTE {OUT / 'diversity_balanced.json'}")


if __name__ == "__main__":
    main()
