# Research plan and honest claim boundaries

## Where the project actually stands

TriboVision is a working, measured cell-segmentation system; a tissue-mechanics
measurement built on top of it and tested against a prediction the theory made in
advance; and a complete, tested analysis pipeline for a treatment experiment that
has not yet been run. The most defensible one-sentence description is:

> A reproducible, leakage-audited segmentation system that recovers a mechanical
> quantity — the shape index governing the vertex model's rigidity transition —
> from ordinary phase-contrast images, with the conditions under which that
> recovery succeeds or fails measured rather than assumed, together with a
> pre-specified statistical pipeline for testing whether those measurements track
> viability.

The mechanics half is the part that answers a question rather than building a
tool. It asks whether a vision model can measure something *physical* about a
tissue, not merely outline its cells, and it answers with the boundary: the
recovery works when the segmenter's measurement error is small relative to how
much the shape index actually varies between images, and fails when it is not.
That boundary is a property of the specimen as much as of the model, which is
why a low correlation on one cell line is not evidence that the method is bad.

That is a real project. What it is not:

- It is **not** evidence that Tribonema kills cancer cells. No Tribonema image
  has been analysed here.
- It is **not** nanotechnology. There is no nanoparticle, no delivery model, no
  nanorobot, and no simulation of one. Using that vocabulary would be branding.
- It is **not** an instance segmenter yet. It labels cell pixels, and does not
  reliably separate cells that touch. How much that costs is now measured rather
  than estimated: 0.046 against 0.389 for an off-the-shelf instance model.
- It is **not** a viability assay. Morphology may predict viability; that is a
  hypothesis the pipeline is built to test, not an assumption it makes.

## Evidence currently in the repository

| Claim | Evidence | Where |
|---|---|---|
| The segmenter beats every non-learned reference on unseen data | Test Dice 0.951, against 0.710 for an all-foreground predictor and 0.425 for the classical rule, on held-out well C7; wins in all 33 independent acquisition groups, sign test p = 2.3e-10 | `results/comparison__comparison.json` |
| The result is not from split leakage | Train = wells A7+D7, val = B7, test = C7; zero shared wells or acquisition groups, verified at load time | `tribovision verify`, `results/baseline__metrics.json` → `split_check` |
| The result is not a lucky seed | Four seeds: 0.9514 ± 0.0007 test Dice, range 0.9507-0.9523 | `results/seed_*__metrics.json`, `docs/RESULTS.md` |
| The instance gap is a representation limit, not a training limit | Zero-shot Cellpose reaches 0.389 where this model reaches 0.046 and the semantic ceiling is 0.168 — with lower pixel Dice | `results/instance_benchmark__instance_benchmark.json` |
| Rates of change are measurable, and refused when the design cannot support them | Planted half-time recovered exactly; fewer than three timepoints reports no rate | `tests/test_treatment.py` |
| Masks match the COCO reference exactly | Bit-identical to `pycocotools` on real LIVECell polygons | `tests/test_coco.py` |
| The learning pipeline can learn | Tiny-dataset overfitting test reaches >0.9 train Dice | `tests/test_training.py` |
| The statistics find real effects and reject noise | Synthetic dose response recovered; null experiment yields q > 0.05 | `tests/test_treatment.py` |
| Cell shape index can be measured at scale against a predicted threshold | 458,187 annotated cells across 8 cell lines; 2 lines sit below the vertex model's q* = 3.81 and 10 of 14 wells reaching confluence 0.5 move toward the jammed side as they crowd | `results/mechanics__mechanics.json` |
| The raster perimeter must be corrected or the physics is wrong | The uncorrected pixel perimeter inflates q by 24% and puts 100% of cells above q*, reversing the verdict for every line | `results/mechanics__segmenter_agreement.json` |
| A segmenter can recover the shape index, within limits | On A172 the ordering across images is recovered at rho 0.767 (Cellpose) and 0.707 (three-class, one checkpoint; 0.679 across seeds, below the pre-set 0.70 bar) | `results/mechanics__segmenter_agreement.json` |
| A low recovery correlation can be the cell line rather than the method | Correlation tracks the ratio of biological spread to measurement error, not segmentation quality; reported beside every correlation | `results/mechanics__attenuation.json` |
| Results are reproducible | Same seed reproduces metrics to 1e-6, including with worker processes | `tests/test_training.py` |

## The gap between here and a Tribonema result

In order, each step blocked by the one before it:

0. **Switch the segmenter to Cellpose.** This is now the first step, not a later
   one: it is measured, it needs no new data, and it is what makes per-cell
   morphology per-cell instead of per-region. Until it happens, the size-derived
   treatment features are confluency measurements.
1. **Collect images.** The design in [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md):
   ≥5 concentrations, vehicle and positive controls, ≥2 wells per condition,
   ≥3 experiment days, **≥4 exposure times with the same wells re-imaged**, fixed
   imaging protocol, recorded µm/pixel.
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

**The trivial-predictor floor.** These frames average 59% foreground. A predictor
that labels every pixel a cell scores 0.710 Dice, so quoting only the classical
baseline's 0.425 would flatter the result. The real margin is +0.242 Dice and
+0.316 IoU over the trivial predictor. IoU is the more honest headline number for
crowded frames, because Dice is generous to over-segmentation. Any future report
of this project should quote both, and should quote the trivial floor alongside.

**Instance separation, and its measured ceiling.** The model is semantic, and its
instance matching score (0.50:0.95) is about 0.05. That number is meaningless
without its ceiling, so `tribovision compare` now measures it: the *ground-truth
mask itself*, put through the same watershed step, scores about 0.12. Through
plain connected components it scores 0.002, because 382 annotated cells in one
frame collapse into a single component.

So roughly 0.05 against a ceiling of 0.12 — the model recovers much of what this
representation allows, and the representation allows very little. That is a
limit of predicting a binary foreground mask at 59% confluence, not a limit of
this model's training. No number of extra epochs moves it.

Three consequences: cell *counts* from this model are not trustworthy at high
density; per-object morphology is a measurement of predicted *regions*; and the
treatment pipeline's size-derived features are confluency measurements until this
is fixed (see the feature taxonomy in `treatment.py`).

The fix is a model that predicts instances directly. **Cellpose, specifically —
not StarDist.** StarDist represents each object as a star-convex polygon, which
cannot express a concave or ruffled adherent cell, and A172 and RAW 264.7 at
confluence are exactly that shape. StarDist is the right tool for round nuclei
and the wrong one here. The cheapest informative next experiment is to run stock
Cellpose `cyto3` on the C7 test split with no training at all and report its
instance score beside 0.05.

**A pipeline ceiling from resampling.** Letterboxing 704x520 into a 512-pixel
square is a 0.727x downscale, and the prediction is resampled back. Pushing the
*ground truth* through that transform and back, with no model at all, scores
0.986 Dice — so 0.986 is the hard ceiling at `image_size=512`, and about a
quarter of the model's residual error is resampling loss rather than model error.
At 768 the round trip is lossless (1.000), which is why that configuration is
reported alongside.

**A measured brittleness to imaging conditions.** Perturbing held-out images
without retraining: a 2x optical zoom costs almost nothing (0.935), gamma changes
nothing (0.952), but a 2-pixel blur drops it to 0.807 and moderate sensor noise
to 0.739 — at which point the model labels 100% of pixels foreground. Note what
0.739 is: exactly the all-foreground score on those frames. **The failure mode is
collapse to the trivial predictor, and Dice makes it look like a mediocre but
working model.** This is the strongest practical reason to quote IoU and the
trivial floor beside every Dice number. Photometric augmentation was added in
response; scale and illumination were never the risk, optics and noise are.

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

**Cell tracking is still absent.** The kinetics analysis measures how fast the
*population average* in a well changes. It cannot say which cells changed first,
or whether one cell rounded and recovered, because nothing matches cells between
frames. Tracking is the natural next step after Cellpose and is meaningless
before it: you cannot follow an object across frames if the segmenter merges it
with its neighbours in each one.

**Membrane damage is not measurable from these images at all.** It is one of the
project's stated goals and no image-analysis method substitutes for a dye assay
(LDH release, propidium iodide, trypan blue). Detachment is partly visible as
confluency loss, but confluency is also what contaminates the size features, so
it has to be reported as its own endpoint.

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

The full record of what was fixed in advance, what was tuned and on which data,
and every claim withdrawn after a stricter test, is in
[PRE_SPECIFICATION.md](PRE_SPECIFICATION.md).

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
| Reproducible code with version control | Done — commit history, `metrics.json` records revision and working-tree cleanliness |
| Quantitative result on held-out data | Done — 0.952 against a 0.710 trivial-predictor floor (see `results/comparison.json`) |
| Comparison against non-learned baselines | Done — `tribovision compare` gates on the *stronger* of the classical rule and the all-foreground predictor |
| Repeated seeds with reported spread | Done — `docs/RESULTS.md` |
| Automated test suite | Done — 658 tests, 91% coverage |
| Documented limitations | Done — this file, and every generated report |
| Pre-specified analysis plan | Done — this file, section above |
| A result that answers a question, not only a tool that works | Done — the mechanics half, with the conditions for its own failure measured |
| Own experimental data | **Not started** — this is the critical path for the *Tribonema* question specifically |
| Paired viability assay | **Not started** |
| Blinded analysis | Procedure defined; awaiting data |
| SRC / biosafety forms | Owner action — see EXPERIMENT_PROTOCOL.md §9 |
| Logbook | Owner action — keep dated records from the first culture session |

The last five rows are the honest boundary of this project. They are the reason
it cannot make any claim about Tribonema: that experiment has not been run, and
nothing in the code will paper over it.

What those rows do *not* mean is that the project has no result. The mechanics
half asks a question and answers it on public data with held-out cell lines, a
threshold the theory fixed in advance, and a measured account of when its own
method stops working. The missing rows bound which question is answered, not
whether one is.
