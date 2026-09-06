# TriboVision

TriboVision turns ordinary phase-contrast microscope images into quantitative,
auditable measurements of cell morphology, and provides the statistical machinery
to test whether those visible changes track an independent viability assay.

It has two halves, and the distinction between them matters:

1. **A working cell segmenter**, trained and evaluated on the public
   [LIVECell](https://sartorius-research.github.io/LIVECell/) dataset. This part
   is finished and measured.
2. **A Tribonema treatment-analysis pipeline** — manifest schema, segmentation,
   per-cell morphometry, dose-response fitting, and viability linkage. The code
   and its statistics are complete and tested against synthetic data with a known
   answer. **No Tribonema images have been collected yet**, so this repository
   contains no evidence about Tribonema's effect on cancer cells.

LIVECell has no Tribonema treatment, concentration, exposure-time, control, or
viability labels. Nothing here can, or claims to, demonstrate an anti-tumour
effect. There is likewise no nanoparticle or nanorobot component of any kind.

## Measured results

Trained on LIVECell A172 phase-contrast images, evaluated on a **held-out well
the model never saw** (`C7`, 60 images):

| Predictor | Test Dice (macro) | Test IoU (macro) | Instance matching 0.50:0.95 |
|---|---|---|---|
| Every pixel labelled cell — no learning at all | 0.710 | 0.593 | — |
| Classical local-contrast + Otsu | 0.425 | 0.275 | 0.007 |
| TriboVision U-Net | **0.952** | **0.909** | 0.049 |
| *The ground-truth mask itself, same instance step* | *1.000* | *1.000* | *0.118* |

**Read the first row before the last one.** These frames average 59% foreground,
so a predictor that labels every single pixel a cell already scores 0.710 Dice —
much better than the classical rule. That trivial predictor, not the classical
one, is the floor this model has to clear, and it clears it by +0.242 Dice. IoU
separates them far more sharply (+0.316), which is why both are reported.

**And read the last row before the instance column.** Putting the *perfect* mask
through the same instance step scores 0.118, because 382 annotated cells in one
crowded frame merge into one predicted region. So 0.049 sits against a ceiling of
0.118, not against 1.0. That is a limit of representing cells as a binary
foreground mask at 59% confluence — not something more training fixes.

Against the classical rule the model wins on every held-out image. The sign test
counts acquisition groups, not crops: LIVECell tiles one capture into several
crops and this manifest is a time-lapse of one well, so the 60 images are 33
independent fields of view (p = 2.3 × 10⁻¹⁰; the per-image figure of 1.7 × 10⁻¹⁸
is pseudoreplication and is reported only to show the size of the inflation). At
the level of *biological* replication this is one well on one plate — n = 1.

`tribovision compare` scores all of these and passes only if the model beats the
*strongest* reference. Reproduce it yourself:

```bash
tribovision compare --checkpoint runs/baseline/best_model.pt
```

Separating individual cells needs a model that predicts instances directly —
Cellpose specifically, not StarDist, whose star-convex polygons cannot represent
a ruffled adherent cell. See [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) for
the measured ceilings, the brittleness stress test, and what remains to be done.

Published run artifacts are in [results/](results/) so these numbers are available
without retraining. `docs/RESULTS.md` is generated from them, not typed by hand.

## What was wrong before, and what fixed it

This project's earlier revision reported a model with 0.68 training Dice and
0.001 validation Dice, which read as a total failure to generalise. It was not.

Loading that exact checkpoint and evaluating it twice — once with `model.eval()`
and once with `model.train()` — gave 0.001 and 0.739 on the same images. The
blocks used `BatchNorm2d`; at batch size 2–4 for one epoch the running mean and
variance never converged on the batch statistics the network was trained with, so
evaluation mode collapsed every prediction to background. Switching to
`GroupNorm`, whose statistics are per sample, makes training and evaluation
identical by construction. `tests/test_model.py` asserts that equality so the
regression cannot come back.

Six other corrections changed reported numbers:

- **Split leakage.** The official LIVECell `train` and `val` files share wells
  (A7, B7 and D7 all appeared in both, with 22 shared acquisition groups). Splits
  are now re-partitioned so no well crosses the train/validation boundary; the
  official test split is untouched.
- **Geometry.** 704×520 frames were stretched into squares, distorting every area,
  perimeter and roundness measurement anisotropically. Images are now letterboxed
  and padding is excluded from loss and metrics.
- **Rasterisation.** COCO polygons were drawn with Pillow, which averaged 0.94 IoU
  against `pycocotools` on 500 real instances and added ~45 pixels per instance.
  `pycocotools` is now a required dependency and the masks are bit-identical.
- **Metric naming and smoothing.** The reported "instance AP" had no confidence
  ranking and no precision-recall curve, so it was not average precision; it is
  now named for what it is. Reported Dice and IoU are exact, with no ε smoothing.
- **A silent data-corruption bug on the default device.** Batches were moved to
  the accelerator with `non_blocking=True` from unpinned host memory, letting the
  copy race the freeing of the augmented tensors. On Apple silicon — which
  `--device auto` selects — masks arrived as NaN while the logits from the same
  batch were still finite, so training optimised against garbage and the run
  finished reporting a plausible score. CPU and MPS now agree to four decimals.
- **A flattering comparison.** The headline quoted the classical rule (0.425),
  which loses to a predictor that labels every pixel a cell (0.710). The gate now
  measures against the stronger reference.

A point-by-point response to the full audit is in
[docs/AUDIT_RESPONSE.md](docs/AUDIT_RESPONSE.md).

## Install

Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

For an exactly reproducible environment, install from the lock file instead:

```bash
uv sync --locked --extra dev
```

LIVECell is **CC BY-NC 4.0**. The dataset is downloaded locally and excluded from
Git. Review the licence before any use beyond non-commercial research.

## Pipeline

### 1. Prepare a leakage-free subset

```bash
tribovision prepare-livecell
```

Downloads the official per-cell-type annotations and only the selected members of
the official image ZIP. Every manifest row records the image, source annotation
URL, DOI, licence, selection seed, SHA-256, annotation IDs, cell type, well, the
assigned split, the *original* official split, and the acquisition group.

Add `--micrometers-per-pixel` if you know your microscope's scale. Without it,
morphology stays in pixels and is never reported in micrometres.

### 2. Verify the splits before trusting any number

```bash
tribovision verify
```

Re-hashes every image, checks that each record's split matches the manifest it
lives in, that every referenced annotation belongs to the stated image, and that
no well or acquisition group is shared between splits. Exits non-zero on leakage.

### 3. Establish a transparent reference

```bash
tribovision baseline --max-images 60
```

A fixed local-contrast + Otsu rule with no learned parameters. A trained model is
only worth reporting if it beats this on images neither has seen.

### 4. Train

```bash
tribovision train --epochs 40 --image-size 512 --batch-size 4 --patience 12
```

Training refuses to start on leaking splits. `metrics.json` records the code
revision, dependency versions, seed, device, dataset content hashes and full
configuration, with repository-relative paths only.

### 5. Segment new images

```bash
tribovision predict --checkpoint runs/baseline/best_model.pt path/to/images/
```

Writes masks, overlays, per-object morphology (area, perimeter, equivalent
diameter, major/minor axis, aspect ratio, circularity, solidity, intensity
statistics) and a report. Predictions are mapped back to the native image grid
*before* anything is measured.

### 6. Prove the model earns its place

```bash
tribovision compare --checkpoint runs/baseline/best_model.pt
```

Exits 0 only if the trained model beats the classical baseline.

### 7. Analyse a treatment experiment

```bash
tribovision treatment-template --output data/treatment/manifest.csv
# fill in one row per image, then:
tribovision analyze-treatment --manifest data/treatment/manifest.csv \
  --checkpoint runs/baseline/best_model.pt
```

Produces per-well morphology, Spearman dose-response with Benjamini-Hochberg
correction, 4-parameter-logistic IC50 fits with bootstrap confidence intervals,
and a leave-one-experiment-day-out test of whether morphology predicts viability,
against a within-day label-permutation null.

The manifest schema and the experimental design it requires are described in
[docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md). Manifests that cannot
support a dose-response claim — no vehicle control, one concentration, or fewer
than two independent wells per condition — are refused rather than analysed.

## Developer commands

```bash
python -m pytest                                  # test suite
python -m pytest --cov=tribovision --cov-report=term-missing
python -m ruff check src tests                    # lint
python -m ruff format --check src tests           # formatting
python -m mypy                                    # type check
uv lock --check                                   # lock file is current
```

## Full source download

```bash
tribovision prepare-livecell --profile full --confirm-full-download
```

Downloads source files only; it does not extract the full archive or generate
full manifests. Full-scale instance segmentation needs substantially more compute
than this laptop-oriented baseline.

## Research design boundaries

LIVECell is appropriate for **pretraining a generic cell segmenter**. It cannot
by itself say whether treated cells are healthy, stressed, or dying. A defensible
Tribonema study still needs new, permissioned images; experimental metadata;
plate- and day-level separation; vehicle and positive controls; and an
independent viability measurement. Do not mix fields from the same well across
model development and final evaluation.

One correction to the source material this project started from: RAW 264.7 is
classified by ATCC as a macrophage/monocyte line derived from a tumour induced by
Abelson murine leukaemia virus. Describing it as a general "mouse leukaemia cell"
model overstates what a result in that line supports.

## Citation

> Edlund, C. et al. LIVECell — A large-scale dataset for label-free live cell
> segmentation. *Nature Methods* 18, 1038–1045 (2021).
> https://doi.org/10.1038/s41592-021-01249-6

TriboVision code is MIT-licensed. LIVECell images, annotations and original
models remain under their authors' CC BY-NC 4.0 licence.
