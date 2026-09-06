# Measured results

Generated from run artifacts by `python scripts/collect_results.py runs`. Every number below comes from a JSON file written by the run that produced it.

## Primary run

- Configuration: {"augment": true, "base_channels": 32, "batch_size": 4, "data_dir": "data/livecell", "depth": 3, "device": "mps", "epochs": 40, "group_by": "well", "image_size": 512, "learning_rate": 0.001, "output_dir": "runs/baseline", "patience": 12, "photometric_augment": true, "seed": 42, "verify_hashes": true, "weight_decay": 0.0001, "workers": 0}
- Device: `mps`
- Best epoch: 36 of 40 run
- Code revision: `f428dd529ba818f29d645714998a5fd05543c829`
- torch 2.11.0, numpy 2.3.5, Python 3.13.9

| Split | Wells | Images | Macro Dice | Micro Dice | Macro IoU |
|---|---|---|---|---|---|
| validation | B7 | 41 | 0.9482 | 0.9524 | 0.9020 |
| test | C7 | 60 | 0.9515 | 0.9563 | 0.9081 |

Split leakage check at the `well` level: **clean** — {'test': 1, 'train': 2, 'val': 1}.

## Acceptance gate: the learned model against every reference

Both scored on `data/livecell/manifests/test.jsonl` (60 images).

| Predictor | Macro Dice | Micro Dice | Macro IoU | Instance matching 0.50:0.95 |
|---|---|---|---|---|
| Every pixel labelled background | 0.0000 | 0.0000 | 0.0000 | - |
| Every pixel labelled cell (no learning) | 0.7103 | 0.7444 | 0.5929 | - |
| Classical local contrast + Otsu | 0.4251 | 0.3949 | 0.2747 | 0.0072 |
| TriboVision U-Net | 0.9510 | 0.9559 | 0.9071 | 0.0455 |
| *The ground-truth mask itself, same instance step* | *1.0000* | *1.0000* | *1.0000* | *0.1677* |

These frames average 59.3% foreground, so labelling every pixel a cell already scores 0.7103 Dice — well above the classical rule. That trivial predictor, not the classical one, is the floor the model has to clear.

- Reference floor: 0.7103 Dice
- Margin over the floor: +0.2407 Dice, +0.3142 IoU (IoU separates them far more sharply)
- Instance ceiling: the perfect mask scores 0.1677 through the same instance step, so the model's 0.0455 should be read against that and not against 1.0.
- Mean per-image Dice difference against the classical rule: +0.5259 (SD 0.1029)
- The learned model wins on 60 of 60 images
- Sign test over 33 independent units — acquisition group (field of view at one timestamp): p = 2.33e-10 (the per-image value, 1.73e-18, is pseudoreplication and is not quoted)
- Verdict: **neural beats every reference predictor**

## Input resolution

Letterboxing 704x520 into a square loses detail. Pushing the *ground truth*
through the transform and back, with no model at all, caps Dice at 0.9861 at
512 and at 1.0000 at 768 — so part of the residual error at 512 is resampling
rather than the model.

| Input size | Round-trip ceiling | Best epoch | Validation Dice | Test Dice |
|---|---|---|---|---|
| 512 | 0.9861 | 36 | 0.9482 | 0.9515 |
| 768 | 1.0000 | 37 | 0.9501 | 0.9566 |

Training at 768 gains +0.0051 test Dice for roughly 2.2x the compute per epoch. It recovers part, not all, of the resampling headroom, which says the remaining error is genuinely the model's.

## Seed replicates

Independent runs differing only in random seed, same splits, same data.

| Run | Seed | Best epoch | Validation Dice | Test Dice | Code revision |
|---|---|---|---|---|---|
| `seed_1` | 1 | 12 | 0.9464 | 0.9525 | `7a5f0f9d` (dirty) |
| `seed_2` | 2 | 20 | 0.9481 | 0.9548 | `afa034bb` |
| `seed_3` | 3 | 19 | 0.9461 | 0.9557 | `c7cd9af9` |
| `baseline` | 42 | 36 | 0.9482 | 0.9515 | `f428dd52` |

Test Dice across 4 seeds: **0.9536 ± 0.0019** (mean ± SD), range 0.9515–0.9557.

The spread is the honest uncertainty on the headline number, and it is two orders of magnitude smaller than the 0.24 margin over the trivial-predictor floor — so the comparison does not depend on a lucky seed.

The revision column matters: runs made at different commits are not strictly interchangeable. Check it before quoting these as replicates of one another.

## Classical baseline detail

- Method: absolute local contrast + Otsu threshold + morphological closing
- Parameters: {"background_radius": 7.0, "instance_method": "connected_components", "micrometers_per_pixel": null, "min_area": 20}
- Images: 60
- Cell-count mean absolute error: 56.2

## How to reproduce

```bash
tribovision prepare-livecell
tribovision verify
tribovision baseline --max-images 60
tribovision train --epochs 40 --image-size 512 --batch-size 4 --patience 12
tribovision compare --checkpoint runs/baseline/best_model.pt
python scripts/collect_results.py runs > docs/RESULTS.md
```

Exact reproduction of a number also requires the same seed, the same device, and the dependency versions recorded in each report's `environment` block. CPU and GPU accumulation orders differ, so cross-device runs agree to about three decimal places, not exactly.
