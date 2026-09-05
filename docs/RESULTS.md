# Measured results

Generated from run artifacts by `python scripts/collect_results.py runs`. Every number below comes from a JSON file written by the run that produced it.

## Primary run

- Configuration: {"augment": true, "base_channels": 32, "batch_size": 4, "data_dir": "data/livecell", "depth": 3, "device": "mps", "epochs": 40, "group_by": "well", "image_size": 512, "learning_rate": 0.001, "output_dir": "runs/baseline", "patience": 12, "seed": 42, "verify_hashes": true, "weight_decay": 0.0001, "workers": 0}
- Device: `mps`
- Best epoch: 23 of 35 run
- Code revision: `4bd15537642e5e9d39ad89a6e8b97a9ce13e014d`
- torch 2.11.0, numpy 2.3.5, Python 3.13.9

| Split | Wells | Images | Macro Dice | Micro Dice | Macro IoU |
|---|---|---|---|---|---|
| validation | B7 | 41 | 0.9492 | 0.9533 | 0.9039 |
| test | C7 | 60 | 0.9522 | 0.9572 | 0.9094 |

Split leakage check at the `well` level: **clean** — {'test': 1, 'train': 2, 'val': 1}.

## Acceptance gate: learned model versus the transparent baseline

Both scored on `data/livecell/manifests/test.jsonl` (60 images).

| Segmenter | Macro Dice | Micro Dice | Macro IoU | Instance matching 0.50:0.95 |
|---|---|---|---|---|
| Classical local contrast + Otsu | 0.4251 | 0.3949 | 0.2747 | 0.0072 |
| TriboVision U-Net | 0.9517 | 0.9568 | 0.9084 | 0.0486 |

- Mean per-image Dice difference: +0.5266 (SD 0.1030)
- The learned model wins on 60 of 60 images
- Exact sign test: p = 1.73e-18
- Verdict: **neural beats every reference predictor**

## Seed replicates

Independent runs differing only in random seed, same splits, same data.

| Run | Seed | Best epoch | Validation Dice | Test Dice |
|---|---|---|---|---|
| `seed_1` | 1 | 12 | 0.9464 | 0.9525 |
| `seed_2` | 2 | 20 | 0.9481 | 0.9548 |
| `seed_3` | 3 | 19 | 0.9461 | 0.9557 |
| `baseline` | 42 | 23 | 0.9492 | 0.9522 |

Test Dice across 4 seeds: **0.9538 ± 0.0017** (mean ± SD), range 0.9522–0.9557.

The spread is the honest uncertainty on the headline number. It is far smaller than the gap to the classical baseline, so the comparison does not depend on a lucky seed.

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
