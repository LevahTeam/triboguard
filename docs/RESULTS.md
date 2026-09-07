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

## Instance separation

Pixel accuracy and cell separation are close to orthogonal here. Every row
below is scored on the same held-out images.

| Method | Matching 0.50:0.95 | @0.50 | @0.75 | Dice | Objects found / true |
|---|---|---|---|---|---|
| Classical local contrast + Otsu | 0.0072 | 0.0206 | 0.0044 | 0.4249 | 490 / 229 |
| TriboVision U-Net (semantic + watershed) | 0.0455 | 0.1060 | 0.0386 | 0.9511 | 225 / 229 |
| *Ground-truth mask, same instance step (ceiling)* | 0.1677 | 0.2852 | 0.1508 | 0.9999 | 476 / 229 |
| Cellpose, zero-shot (no training on this data) | 0.3893 | 0.6895 | 0.3997 | 0.9239 | 188 / 229 |

Cellpose, with **no training on this data at all**, separates cells 8.6x better than the in-house model and 2.3x better than the ceiling a binary-mask representation allows — while scoring *lower* pixel Dice (0.9239 against 0.9511).

That is the whole finding. The in-house model is not worse at seeing cells; it is bound by predicting a binary foreground mask, and no amount of further training on that objective moves the instance number. Note also that the U-Net's object *count* is the closest to truth of any method here, while its matching score is the second worst — getting the count right by accident is not the same as getting the objects right, which is why counts alone are not reported as a result.

Cellpose configuration: cpsam v4.2.1.1, diameter 15.0 chosen on the training split.

## A three-class instance model, and its uncertainty

Same 2M-parameter U-Net body; only the output head and the target
change, from a binary mask to background / cell interior / touching-cell
boundary. Intervals resample acquisition groups, not crops.

| Method | Matching 0.50:0.95 | 95% CI |
|---|---|---|
| Same U-Net, binary target | 0.0455 | [0.0223, 0.0794] |
| *Perfect binary mask (ceiling)* | 0.1677 | [0.1230, 0.2286] |
| **Same U-Net, three-class target** | 0.2078 | [0.1840, 0.2340] |
| Cellpose, zero-shot | 0.3893 | [0.3533, 0.4259] |

- Against the binary target: **+0.1623** [+0.1466, +0.1772], which excludes zero. Changing the prediction target, and nothing else, is what produced this.
- Against a *perfect* binary mask: +0.0400 [-0.0031, +0.0781] — the model is indistinguishable from it. The point estimate is higher, and the interval includes zero, so the honest claim is a tie, not a win.

### Does more training data help?

Same network, same hyperparameters, and a held-out manifest that is
byte-identical across every point, so the curve measures data alone.

| Training images | Matching 0.50:0.95 | 95% CI | Boundary recall |
|---|---|---|---|
| 79 | 0.1417 | [0.1219, 0.1664] | 0.6464 |
| 152 | 0.1472 | [0.1295, 0.1691] | 0.6758 |
| 304 | 0.2078 | [0.1840, 0.2340] | 0.6926 |

Paired differences, same images and same groups:

- **152 vs 79 images**: +0.0055 [+0.0012, +0.0092]
- **304 vs 152 images**: +0.0606 [+0.0521, +0.0685]
- **304 vs 79 images**: +0.0660 [+0.0587, +0.0739]

- The endpoints have three seeds each; the middle point has one, so it is reported but not claimed.
- All training images come from the same two wells, so this measures more samples of the same conditions rather than more diversity.
- The paired differences below are over images only. See the seed section for what happens when training stochasticity is included.

### How much of that is the seed?

Three seeds at each endpoint, all scored on the same 60 images. The
differences above come from a bootstrap over *images*, which is silent on
which initialisation a model was trained from - a separate question with a
separate answer.

| Training images | Seed scores | Mean | Seed SD | CI images only | CI images + seed |
|---|---|---|---|---|---|
| 79 | 0.1250, 0.1453, 0.1417 | 0.1373 | 0.0108 | [0.1171, 0.1622] | [0.1153, 0.1654] |
| 304 | 0.2360, 0.2088, 0.2078 | 0.2175 | 0.0160 | [0.1914, 0.2465] | [0.1886, 0.2520] |

- **304 vs 79 images, accounting for both sources**: +0.0802 [+0.0623, +0.1013] - the headline scaling result survives.
- **The 152 vs 79 step does not.** It is +0.0055, about half the seed standard deviation of 0.0108 measured at that size. The image-level interval called it significant because it was never asked about training noise. Only one seed was run at 152, so that point is reported but not claimed.

Adding seed uncertainty widens these intervals by roughly 11-15%, which is modest. The lesson is not that image-level intervals are useless, but that a difference smaller than the seed spread cannot be established by them however many images are tested.

## Cross-cell-line transfer

Every model was trained on A172 (glioblastoma), wells A7+D7, 304 images and tested on lines it
never saw. **Raw scores are not comparable across cell lines** — the ceiling,
which is how hard the instance task is for that line, varies more than
fourfold here, so an average of raw scores measures which lines were picked
more than it measures the model.

| Cell line | Ceiling | Three-class | ÷ ceiling | Cellpose | ÷ ceiling |
|---|---|---|---|---|---|
| A172 *(seen in training)* | 0.1677 | 0.2078 | 1.24 | 0.3893 | 2.32 |
| MCF7 (breast) | 0.2499 | 0.1325 | 0.53 | 0.3834 | 1.53 |
| SHSY5Y (neuroblastoma) | 0.1215 | 0.0588 | 0.48 | 0.2497 | 2.06 |
| SkBr3 (breast) | 0.5332 | 0.4718 | 0.88 | 0.6378 | 1.20 |

- Specialist (A172-trained): 1.24 of ceiling on the line it saw, 0.63 on unseen lines (**-49%**)
- Generalist (Cellpose): 2.32 to 1.60 (**-31%**)

Cellpose beats the specialist on every line, seen or unseen, and the paired interval excludes zero every time. Specialising on one cell line costs about 1.6 times more generalisation than the generalist gives up.

## Tissue mechanics from cell outlines

The vertex model of a confluent monolayer predicts a rigidity transition at a dimensionless shape index q* = 3.81, where q = P/sqrt(A). Below it a tissue is jammed and solid-like; above it cells can exchange neighbours and the tissue flows. That number is a *prediction of the theory*, not a fit to this data, which is what makes it a test rather than a description.

Measured over **458,187 annotated cells** across 8 cell lines. For reference a circle sits at 3.5449 and a regular hexagon at 3.7224; nothing can fall below the circle, so a measurement that does is an artifact rather than a discovery.

| Cell line | Cells | Median q | State | Unjammed | Wells jamming as they crowd |
|---|---|---|---|---|---|
| A172 | 32,942 | 4.543 | unjammed (fluid-like) | 94% | 4 of 4 |
| BT474 | 32,467 | 4.149 | unjammed (fluid-like) | 83% | not crowded |
| BV2 | 87,383 | 3.733 | jammed (solid-like) | 35% | not crowded |
| Huh7 | 12,509 | 4.085 | unjammed (fluid-like) | 78% | not crowded |
| MCF7 | 96,629 | 4.195 | unjammed (fluid-like) | 91% | 1 of 2 |
| SHSY5Y | 75,515 | 5.252 | unjammed (fluid-like) | 94% | 2 of 2 |
| SKOV3 | 55,391 | 4.532 | unjammed (fluid-like) | 94% | 3 of 4 |
| SkBr3 | 65,351 | 3.776 | jammed (solid-like) | 46% | 0 of 2 |

2 of 8 lines sit below q* on the median cell, and **10 of 14** wells that reach confluence 0.5 move *toward* the jammed side as they crowd, which is the direction the theory predicts. Each well is one trajectory and one unit of analysis; fields imaged at the same timestamp are averaged before the well is scored, so crops of one field cannot count as independent measurements.

A caution that belongs next to the result rather than in a footnote: for A172 the density relationship **does not survive controlling for cell size**. Crowding and cell area move together, and a partial correlation cannot separate them here. The trajectory result is the stronger of the two, and the cross-sectional density correlation should not be quoted on its own.

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
| `seed_1` | 1 | 38 | 0.9481 | 0.9523 | `c60ef886` |
| `seed_2` | 2 | 40 | 0.9467 | 0.9510 | `9e380d17` |
| `seed_3` | 3 | 28 | 0.9489 | 0.9507 | `9e380d17` |
| `baseline` | 42 | 36 | 0.9482 | 0.9515 | `f428dd52` |

Test Dice across 4 seeds: **0.9514 ± 0.0007** (mean ± SD), range 0.9507–0.9523.

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
