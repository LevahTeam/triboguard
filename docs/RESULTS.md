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

- The endpoints have three seeds each; the middle point has one, so it is reported but not claimed.
- All training images come from the same two wells, so this measures more samples of the same conditions rather than more diversity.
- The paired differences below are over images only. See the seed section for what happens when training stochasticity is included.

Paired differences, same images and same groups:

- **152 vs 79 images**: +0.0055 [+0.0012, +0.0092]
- **304 vs 152 images**: +0.0606 [+0.0521, +0.0685]
- **304 vs 79 images**: +0.0660 [+0.0587, +0.0739]

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

## Diversity versus volume

Both arms train on exactly 304 images with identical hyperparameters and 3 seeds each. The mixed arm draws from A172, MCF7, SkBr3; the control sees A172 alone. Both are scored on SHSY5Y, which neither arm ever saw. Matching the size is the point: the scaling curve confounds volume with variety, and this does not.

Two things that could have made this unfair were checked rather than assumed. The validation sets that drive early stopping are comparable in size (152 images for the single-line arm, 162 for the mixed one), so neither arm gets a cleaner stopping signal. And the mixed training subset stays balanced after the cap is applied — 102 / 94 / 108 images across the three lines — so the "mixed" arm is genuinely mixed rather than one line with a garnish.

| Training set | Mean | Seed SD | 95% CI (seeds + images) | ÷ ceiling |
|---|---|---|---|---|
| A172 only | 0.0570 | 0.0021 | [0.0503, 0.0654] | 0.47 |
| Three lines | 0.0576 | 0.0027 | [0.0481, 0.0702] | 0.47 |

**A caveat that could explain a null result.** The three lines arm selected at epochs 5, 10, 20; the a172 only arm selected at epochs 22, 26, 30. Selection uses validation boundary recall alone, and a single noisy metric can spike early and never be beaten, ending training with an under-trained model. The mixed arm stopped consistently earlier, so its score may reflect less effective training rather than less useful data. The rule was identical for both arms and fixed before either ran, which makes this a fair protocol comparison — but not, on its own, a clean test of diversity. The cost is measurable from the training histories, with no re-training: decoding an instance needs interior probability to find cells and boundary to split them, and the saved checkpoint gives up 0.135 of the interior recall the mixed runs demonstrably reached, against 0.059 for the single-line runs. For the single-line arm, selecting on boundary recall lands within a few epochs of what valuing both recalls would have chosen; for the mixed arm it does not. So the rule is not neutral between the arms, and the comparison understates the mixed one by an amount this experiment cannot pin down without re-running both.

Mixed minus single-line: **+0.0006** [-0.0052, +0.0070], resampling seeds and acquisition groups together. The interval includes zero. At this scale there is **no detectable advantage** to training on three cell lines rather than one, which points at capacity or at the representation as the binding constraint rather than at the narrowness of the training data.

The held-out ceiling is 0.1215: the ground-truth mask itself scores that through the same decoder, so both arms should be read against it rather than against 1.0.

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

## Does the mechanics result survive a change of cell line?

The shape-index recovery was originally measured on A172 only. Re-measuring it on three further lines tests whether the segmenter is doing something general or something A172-shaped. The rasterised ground truth is a positive control: where it fails, the limit is the measurement chain, not the segmenter. The rho > 0.7 bar predates the three-class model and is unchanged.

| Cell line | Median true q | Truth (control) | Cellpose | Three-class |
|---|---|---|---|---|
| A172 *(developed on)* | 4.621 | 0.951 ✓ | 0.767 ✓ | 0.707 ✓ |
| MCF7 | 4.182 | 0.927 ✓ | 0.388 | 0.181 |
| SHSY5Y | 5.726 | 0.978 ✓ | 0.893 ✓ | 0.801 ✓ |
| SkBr3 | 3.746 | 0.855 ✓ | 0.817 ✓ | 0.791 ✓ |

A ✓ marks a method clearing the pre-set rho > 0.7 bar on that line. Lines are reported individually and never averaged: the instance ceiling varies more than fourfold across them, so an average would mostly record which lines were picked.

The A172 three-class figure is one checkpoint, the same one used on every other line here, so the rows are comparable. Across three seeds that same arm averages 0.679, which is below the bar — the single-seed 0.707 is the optimistic reading and is marked ✓ only because the table reports the checkpoint, not the mean.

## Is a low correlation a bad segmenter, or nothing to track?

A rank correlation between true and recovered shape index confounds two things: how accurately a method measures q on one image, and how much q actually varies between the images being ranked. When the second is small the correlation collapses even for a near-perfect method, so a low number is not on its own evidence that segmentation failed.

Measurement error comes from each method's limits of agreement (a span of 3.92 standard deviations). Biological variation comes from the polygon annotations alone, with no segmenter in the loop. Their ratio is a signal-to-noise ratio.

| Cell line | Method | Biological sd | Error sd | SNR | Observed ρ | Attenuation predicts |
|---|---|---|---|---|---|---|
| MCF7 | three-class | 0.1092 | 0.1044 | 1.05 | 0.181 | 0.723 |
| MCF7 | Cellpose | 0.1092 | 0.0959 | 1.14 | 0.388 | 0.751 |
| SHSY5Y | three-class | 0.6716 | 0.5516 | 1.22 | 0.801 | 0.773 |
| A172 | three-class | 0.3054 | 0.2024 | 1.51 | 0.707 | 0.834 |
| SkBr3 | three-class | 0.1014 | 0.0620 | 1.63 | 0.791 | 0.853 |
| SkBr3 | Cellpose | 0.1014 | 0.0598 | 1.69 | 0.817 | 0.861 |
| A172 | Cellpose | 0.3054 | 0.1519 | 2.01 | 0.767 | 0.895 |
| SkBr3 | ground truth (raster) | 0.1014 | 0.0461 | 2.20 | 0.855 | 0.910 |
| SHSY5Y | Cellpose | 0.6716 | 0.2689 | 2.50 | 0.893 | 0.928 |
| MCF7 | ground truth (raster) | 0.1092 | 0.0375 | 2.91 | 0.927 | 0.946 |
| A172 | ground truth (raster) | 0.3054 | 0.0659 | 4.63 | 0.951 | 0.977 |
| SHSY5Y | ground truth (raster) | 0.6716 | 0.1182 | 5.68 | 0.978 | 0.985 |

Rows are sorted by signal-to-noise, not by correlation. Observed ρ rises with it but not perfectly monotonically, so the relationship is a strong tendency rather than a law.

The extremes make the point: MCF7 with three-class has only 1.05 times more biological signal than measurement noise and scores ρ = 0.181, while SHSY5Y with ground truth (raster) has 5.68 and scores 0.978.

Across all 12 cell-line-by-method combinations, the ratio predicts the observed correlation at Spearman **+0.930** (p < 0.0001). Neither term alone does: biological spread reaches only +0.281 (p = 0.38) and measurement error -0.231 (p = 0.47). Two earlier explanations — cell size, then biological spread on its own — were each written down in advance and each refuted by the next cell line; see docs/PRE_SPECIFICATION.md.

**This changes how the cross-cell-line table should be read.** A low score is not automatically a failure of segmentation, but neither is a narrow dynamic range automatically an excuse: SkBr3 has the narrowest spread of the four lines and still recovers the ordering at 0.791, because the error there is smaller still. What has to be checked, line by line, is the ratio — which is why it is printed beside every correlation rather than left for a reader to infer.

The predicted column uses the classical attenuation formula, which assumes Pearson correlation and errors independent of the true value. Neither holds exactly here — the error grows with q — so it consistently overestimates, and it is used as a direction check rather than a fit. The claim is the ordering, not the numbers.

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
```

The mechanics and cross-cell-line results, in order — each reads the artifacts the one before it wrote:

```bash
# Biological spread of shape index, from polygons alone. No segmenter involved.
python scripts/measure_q_dynamic_range.py
# Cell size and perimeter discretisation, per line.
python scripts/measure_resolution_limit.py
# The recovery on three lines the method was not developed on.
python scripts/measure_across_cell_lines.py
# Signal-to-noise beside every correlation. Needs across_cell_lines.json.
python scripts/measure_attenuation.py
# Diversity versus volume at matched training size. Trains three models.
python scripts/run_diversity_experiment.py
python scripts/collect_results.py runs > docs/RESULTS.md
```

Exact reproduction of a number also requires the same seed, the same device, and the dependency versions recorded in each report's `environment` block. CPU and GPU accumulation orders differ, so cross-device runs agree to about three decimal places, not exactly.
