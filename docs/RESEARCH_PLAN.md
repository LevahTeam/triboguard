# Research plan and honest claim boundaries

## Where the project actually stands

TriboVision is a working, measured cell-segmentation system plus a complete,
tested analysis pipeline for a treatment experiment that has not yet been run.
The most defensible one-sentence description is:

> A reproducible, leakage-audited segmentation system that measures cell
> morphology from ordinary phase-contrast images, together with a pre-specified
> statistical pipeline for testing whether those measurements track viability —
> validated end to end on public data and on synthetic experiments with known
> answers, and awaiting its own Tribonema imaging data.

That is a real project. What it is not:

- It is **not** evidence that Tribonema kills cancer cells. No Tribonema image
  has been analysed here.
- It is **not** nanotechnology. There is no nanoparticle, no delivery model, no
  nanorobot, and no simulation of one. Using that vocabulary would be branding.
- It is **not** an instance segmenter yet. It labels cell pixels, and does not
  reliably separate cells that touch.
- It is **not** a viability assay. Morphology may predict viability; that is a
  hypothesis the pipeline is built to test, not an assumption it makes.

## Evidence currently in the repository

| Claim | Evidence | Where |
|---|---|---|
| The segmenter beats a transparent non-learned rule on unseen data | Test Dice 0.952 vs 0.425 on held-out well C7; wins 60/60 images, sign test p = 1.7e-18 | `runs/comparison/comparison.json` |
| The result is not from split leakage | Train = wells A7+D7, val = B7, test = C7; zero shared wells or acquisition groups, verified at load time | `tribovision verify`, `runs/baseline/metrics.json` → `split_check` |
| The result is not a lucky seed | Three independent seeds reported with spread | `runs/seed_*/metrics.json`, `docs/RESULTS.md` |
| Masks match the COCO reference exactly | Bit-identical to `pycocotools` on real LIVECell polygons | `tests/test_coco.py` |
| The learning pipeline can learn | Tiny-dataset overfitting test reaches >0.9 train Dice | `tests/test_training.py` |
| The statistics find real effects and reject noise | Synthetic dose response recovered; null experiment yields q > 0.05 | `tests/test_treatment.py` |
| Results are reproducible | Same seed reproduces metrics to 1e-6, including with worker processes | `tests/test_training.py` |

## The gap between here and a Tribonema result

In order, each step blocked by the one before it:

1. **Collect images.** The design in [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md):
   ≥5 concentrations, vehicle and positive controls, ≥2 wells per condition,
   ≥3 experiment days, fixed imaging protocol, recorded µm/pixel.
2. **Run the paired viability assay** on the same wells, with a
   reagent-interference control.
3. **Fine-tune the segmenter** on a small annotated subset of those images. The
   LIVECell model transfers to phase-contrast morphology, but not automatically
   to a different cell line, magnification and illumination; assume it needs
   ~20–50 annotated frames, held out by day.
4. **Analyse blind**, joining morphology to the treatment key only at the end.
5. **Report** the dose response, the IC50 with its confidence interval, and the
   leave-one-day-out prediction result — including if it is negative.

## Known limitations, stated rather than discovered

**Instance separation.** The model is semantic. Instance matching (0.50:0.95) is
0.049, which is better than the classical baseline's 0.007 but far from usable
for per-cell counting at confluence. Two consequences: cell *counts* from this
model are not trustworthy at high density, and per-object morphology is a
measurement of predicted *regions*. The watershed splitter improves this and is
labelled in every output row, but it is a heuristic, not a trained instance
model. The fix is a model that predicts instances directly — Cellpose or
StarDist fine-tuned on the project's own annotated frames.

**One cell line, one plate.** The LIVECell subset used here is A172 from a single
plate. Wells A7/B7/D7/C7 are separated, which controls for the strongest local
confound, but it does not demonstrate transfer to another plate, another day,
another microscope, or another cell line. This is the single largest external
validity gap in the segmentation half.

**Crack perimeter bias.** Circularity is computed from a pixel-boundary
perimeter, which overestimates the true circumference; a perfectly round
digitised object scores about 0.59, not 1.0. The constant is published in every
report. Circularity is therefore valid for comparing conditions measured the same
way, and invalid as an absolute shape constant.

**MTS is metabolic activity, not death.** Any correlation found between
morphology and MTS is a correlation with metabolic activity. Calling it cell
death requires a death-specific assay.

**Sample size.** With the design minimum (2 days, 2 wells per condition), the
leave-one-day-out test has very little power. A negative result at that size is
uninformative, not evidence of absence. Plan for 3+ days.

## An observation from running the pipeline on synthetic data

Running `analyze-treatment` over one synthetic experiment twice — once with the
classical baseline and once with the LIVECell-trained U-Net — gives the same
dose-response answer (area falls with concentration, rho = -0.93, q = 6e-9, IC50
fitted with R^2 = 0.93) but very different answers on the viability question. The
classical path reached R^2 > 0.5 with p < 0.05; the neural path returned
R^2 = -1.67 with p = 0.42, meaning worse than predicting the mean.

That is not a bug, and it is worth understanding before the real experiment. The
U-Net was trained on A172 phase-contrast microscopy; the synthetic frames look
nothing like it, so its features on them are poor, and the leave-one-day-out test
correctly declines to claim transfer it cannot support. Two lessons follow:

1. Step 3 of the plan — fine-tuning on the project's own annotated frames — is
   not optional polish. The segmenter's domain match materially changes the
   downstream statistical conclusion.
2. The pipeline fails in the safe direction. A domain-mismatched model produced a
   negative result rather than a confident wrong one.

## Statistical commitments made in advance

- The **well** is the unit of analysis. Fields within a well are averaged first.
- Every comparison is **blocked by experiment day**.
- Correlations are **Spearman**, because a dose response need not be linear.
- p-values across morphology features are corrected with **Benjamini-Hochberg**;
  raw p-values remain visible alongside q-values.
- Predictive skill is measured by **leave-one-experiment-day-out**
  cross-validation against a **within-day label-permutation null**.
- IC50 confidence intervals are **bootstrap** (400 resamples), and an IC50
  outside the tested concentration range is reported as an extrapolation.
- The primary endpoint is fixed in advance (mean per-cell area versus
  concentration). Everything else is exploratory and labelled as such.

## Competition-readiness checklist

| Item | Status |
|---|---|
| Reproducible code with version control | Done — commit history, `metrics.json` records revision |
| Quantitative result on held-out data | Done — 0.952 vs 0.425, p = 1.7e-18 |
| Comparison against a non-learned baseline | Done — `tribovision compare` gates on it |
| Repeated seeds with reported spread | Done — `docs/RESULTS.md` |
| Automated test suite | Done — 280+ tests, 91% coverage |
| Documented limitations | Done — this file, and every generated report |
| Pre-specified analysis plan | Done — this file, section above |
| Own experimental data | **Not started** — this is the critical path |
| Paired viability assay | **Not started** |
| Blinded analysis | Procedure defined; awaiting data |
| SRC / biosafety forms | Owner action — see EXPERIMENT_PROTOCOL.md §9 |
| Logbook | Owner action — keep dated records from the first culture session |

The last five rows are the honest reason this is not yet a finished science-fair
project: the engineering is done and measured, and the experiment has not been
run. Nothing in the code will paper over that.
