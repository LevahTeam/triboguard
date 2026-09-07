"""Diversity versus volume: does a mixed training set transfer better than a
single-line one of the same size?

Both arms see 304 training images. The A172 arm sees one cell line; the mixed
arm sees three (A172, MCF7, SkBr3). Neither sees SHSY5Y, which is the held-out
line both are scored on. If diversity beats volume, the mixed arm wins at
matched training size -- and the win cannot be explained by more data, because
there is no more data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, "src")

from tribovision import evaluation, provenance  # noqa: E402
from tribovision.instance_model import InstanceConfig, score_on_manifest, train_instance_model  # noqa: E402

SEEDS = (42, 1, 2)
TRAIN_LIMIT = 304
HELD_OUT = Path("data/livecell/manifests_shsy5y/test.jsonl")
# Measured in the transfer experiment: the ground-truth mask itself scores this
# through the same instance decoder, so it is the most a perfect segmenter of
# this representation could reach on SHSY5Y.
SHSY5Y_CEILING = 0.12150083530255071
A172_RUNS = {42: "runs/scale_304", 1: "runs/seedrun_304_1", 2: "runs/seedrun_304_2"}


def _score(checkpoint: Path) -> dict[str, Any]:
    return score_on_manifest(checkpoint, HELD_OUT, device="mps")


def main() -> None:
    out = Path("runs/diversity")
    out.mkdir(parents=True, exist_ok=True)
    arms: dict[str, dict[int, dict[str, Any]]] = {"a172_only": {}, "mixed": {}}

    for seed in SEEDS:
        run_dir = out / f"mixed_{seed}"
        checkpoint = run_dir / "best_model.pt"
        if not checkpoint.exists():
            print(f"training mixed seed={seed}", flush=True)
            train_instance_model(
                InstanceConfig(
                    data_dir="data/livecell",
                    output_dir=str(run_dir),
                    manifests_dirname="manifests_mixed",
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
        print(f"scoring mixed seed={seed} on SHSY5Y", flush=True)
        arms["mixed"][seed] = _score(checkpoint)
        print(f"  mixed  seed={seed} -> {arms['mixed'][seed]['matching_50_95']:.4f}", flush=True)

        print(f"scoring a172-only seed={seed} on SHSY5Y", flush=True)
        arms["a172_only"][seed] = _score(Path(A172_RUNS[seed]) / "best_model.pt")
        print(f"  a172   seed={seed} -> {arms['a172_only'][seed]['matching_50_95']:.4f}", flush=True)

    # Two-level interval: over seeds and over acquisition groups, so the
    # comparison is not resting on one lucky initialisation.
    per_seed = {
        arm: {str(seed): res["per_image"] for seed, res in runs.items()}
        for arm, runs in arms.items()
    }
    groups = arms["mixed"][SEEDS[0]]["groups"]
    report = {
        "design": {
            "train_images_per_arm": TRAIN_LIMIT,
            "held_out_line": "SHSY5Y",
            "held_out_ceiling": SHSY5Y_CEILING,
            "seeds": list(SEEDS),
            "mixed_lines": ["A172", "MCF7", "SkBr3"],
        },
        "per_seed_score": {
            arm: {str(s): r["matching_50_95"] for s, r in runs.items()} for arm, runs in arms.items()
        },
        "mixed": evaluation.replicate_interval(per_seed["mixed"], groups=groups),
        "a172_only": evaluation.replicate_interval(per_seed["a172_only"], groups=groups),
        "mixed_minus_a172": evaluation.replicate_difference(
            per_seed["mixed"], per_seed["a172_only"], groups=groups
        ),
        "environment": provenance.environment(),
    }
    for arm in ("mixed", "a172_only"):
        report[arm]["fraction_of_ceiling"] = report[arm]["mean"] / SHSY5Y_CEILING
    (out / "diversity.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["mixed_minus_a172"], indent=2))
    print("WROTE runs/diversity/diversity.json")


if __name__ == "__main__":
    main()
