# Tribonema experiment protocol

This describes the experiment TriboVision's analysis half is built to consume.
It is written so that someone else could repeat it, and so that a reviewer can
tell exactly which claims the design can and cannot support.

## 1. What is being asked

**Question.** Do visible morphological changes in cultured cells, measured
automatically from phase-contrast images, track the loss of viability caused by
Tribonema extract?

**Primary hypothesis.** Mean cell circularity at a fixed exposure time changes
monotonically with Tribonema extract concentration.

This endpoint was deliberately changed from "mean per-cell area". While the
segmenter is semantic, touching cells merge into one predicted region, so an
object's *area* is largely a measure of how crowded the field is. A cytotoxic
extract kills cells, cells detach, confluency falls, and mean object area falls
with it — a clean dose response that says only "fewer cells at higher dose",
which the viability assay already said. Circularity and aspect ratio are
scale- and density-invariant, so they can answer the question actually asked.
`treatment.py` labels every feature by kind and refuses to let a density feature
be the confirmatory endpoint by accident.

**Secondary hypothesis.** A model fitted to morphology features on one set of
experiment days predicts MTS viability on a day it never saw, better than chance.

**Null hypotheses.** Morphology is unrelated to concentration; morphology carries
no information about viability beyond what day-to-day variation explains.

Both are stated before data collection, because a dose-response study that picks
its endpoint afterwards can find a trend in almost anything.

## 2. What the design must include

The analysis refuses manifests that cannot support the claim. Concretely:

| Requirement | Why | Enforced by |
|---|---|---|
| At least one zero-concentration (vehicle) control | Without it, "cells changed" cannot be separated from "cells were cultured" | `_check_design` |
| At least two distinct concentrations | One point is not a dose response | `_check_design` |
| At least two independent wells per condition | Otherwise within-condition variability is unestimated and every p-value is fiction | `_check_design` |
| At least two experiment days | Leave-one-day-out is the only honest test of transfer | `_check_design` |
| A well marked `control_type='vehicle'` | A zero-dose well is not a vehicle control unless it got the same solvent at the same final concentration | `_check_design` |
| Five distinct non-zero concentrations and eight wells, before any IC50 is fitted | Four parameters fitted to four points has zero residual degrees of freedom | `fit_dose_response` |

### Power: plan for more than the minimum

The enforced minimum lets the analysis run; it does not give it a real chance of
detecting an effect. Simulated power for the confirmatory leave-one-day-out test,
at a within-day correlation of rho between the morphology endpoint and viability,
with day-level batch shifts in both:

| Design | rho = 0.5 | rho = 0.7 | rho = 0.85 |
|---|---|---|---|
| 2 days, 20 wells | 0.53 | 0.85 | 0.99 |
| 3 days, 30 wells | 0.74 | 0.95 | 1.00 |
| **4 days, 60 wells** | **0.96** | **1.00** | **1.00** |

Measured type-I error under a pure null across the same designs: 0.033-0.067
against a nominal 0.05, i.e. correctly calibrated.

Those figures depend on the within-day centring the analysis now uses by default.
Without it — fitting across days to predict absolute viability — the same designs
give 0.33, 0.32 and 0.35 at rho = 0.5, and power stops improving as the study
grows, because the model keeps learning a slope that day-level confounding has
corrupted. That is a large enough difference to change what an experiment is
worth doing.

**Plan for 3-4 experiment days and 45-60 wells**, and commit to that number in the
logbook before collecting anything. A negative result below that size is
uninformative, and saying so afterwards is not the same as saying so in advance.

Recommended, beyond the enforced minimum:

- **Five or more concentrations** spanning at least two orders of magnitude, or
  an IC50 fit will be an extrapolation. The report flags this when it happens.
- **A vehicle control matched to the extract solvent**, at the same final solvent
  concentration as the highest dose.
- **A positive control** (a compound with a known cytotoxic effect in this line)
  on every plate, to show the assay could have detected an effect.
- **Three independent biological replicates** — separate passages on separate
  days, not three wells from one suspension. Wells from one suspension are
  technical replicates; the analysis treats the well as the unit and averages
  fields within it, so extra fields never inflate the sample size.
- **Randomised plate layout** so that concentration is not confounded with edge
  effects, and edge wells excluded or explicitly modelled.

## 3. Blinding

Blind the image analysis, not just the reader:

1. Acquire images and store the treatment key separately.
2. Rename image files to opaque identifiers before segmentation.
3. Run segmentation and morphometry on the renamed set.
4. Join morphology to the treatment key only at the analysis step.

The manifest schema supports this directly: `image_path` may point at renamed
files, and every experimental column is filled in from the key afterwards.

## 4. Imaging

- Record `micrometers_per_pixel` from the objective and camera. Without it,
  TriboVision reports pixels and refuses to print micrometres.
- Keep magnification, exposure, illumination and focus protocol identical across
  every condition on a given day. Any of these varying with concentration
  produces a "morphology change" that is really an imaging change.
- Image a fixed number of fields per well, at pre-declared positions.
- Record the time from treatment to imaging per plate; drift here becomes an
  exposure-time confound.

## 5. Viability assay

MTS is run on the same wells that were imaged, after imaging. Record both raw
absorbance and blank absorbance:

- `mts_absorbance` — the well reading
- `mts_blank_absorbance` — medium plus reagent, no cells

TriboVision computes `viability_fraction = mts_absorbance - mts_blank_absorbance`
when `viability_fraction` is left blank. If you normalise to vehicle control
yourself, put the normalised value in `viability_fraction` and say so in `notes`.

Note the known limitation: MTS reports metabolic activity, not cell death. A
reduced MTS signal is consistent with fewer cells, less metabolically active
cells, or direct interference by the extract with the reagent. Include a
reagent-only-plus-extract control to check for that interference.

## 6. Manifest columns

Generate the template with `tribovision treatment-template`.

**Required**

| Column | Meaning |
|---|---|
| `image_path` | Path relative to the manifest |
| `experiment_day` | ISO date; the blocking factor for every comparison |
| `plate_id` | Physical plate |
| `well_id` | Well, e.g. `B04` |
| `field` | Field of view within the well |
| `cell_line` | e.g. `RAW264.7` — see the note in the README about how this line should be described |
| `treatment` | Extract name and preparation batch |
| `concentration_ug_per_ml` | 0 for vehicle control |
| `exposure_hours` | Treatment to imaging |
| `control_type` | `vehicle`, `positive`, `untreated`, or `treated` |
| `replicate` | Biological replicate identifier |

**Optional but strongly recommended**

`viability_fraction`, `mts_absorbance`, `mts_blank_absorbance`,
`micrometers_per_pixel`, `operator`, `permission_status`, `notes`.

## 7. What the analysis reports

- Per-well morphology summaries (fields averaged first).
- Spearman correlation of each feature against concentration, with a p-value
  obtained by permuting concentration labels **within each experiment day** — so a
  day effect cannot masquerade as a dose effect — and Benjamini-Hochberg q-values
  across features. The pooled parametric p-value is reported alongside, as a
  descriptive only.
- Each feature labelled `shape`, `density`, `density-contaminated` or `intensity`,
  so a reader can tell which results are about morphology and which are about
  confluency.
- 4-parameter-logistic dose-response fits with bootstrap IC50 confidence
  intervals, flagged when the IC50 falls outside the tested range.
- Leave-one-day-out ridge prediction of viability from morphology, with a
  within-day label-permutation null and an exact permutation p-value. The model
  is fitted to each day's deviations from its own mean, so what it predicts is a
  well's viability **relative to its own day**, not an absolute value; the result
  says so explicitly. Two are
  reported: a **confirmatory** model using the single pre-specified endpoint, and
  an **exploratory** one using every available feature. Eight correlated
  predictors at 20-30 wells roughly halves the power, which is why the
  confirmatory test is the one that counts.
- R^2 both against the global mean and **within day**. Only the within-day figure
  is free of between-day batch variance; the plain R^2 can look excellent while
  explaining nothing about which well within a day is healthier.
- Bootstrap IC50 intervals with the resample convergence rate. Below 80%
  convergence no interval is reported, because one built from only the
  well-behaved resamples is biased narrow.

## 8. What the analysis will not report

- Any statement of mechanism. Morphology plus an MTS reading cannot distinguish
  apoptosis from necrosis from detachment. Establishing necrosis needs a specific
  assay (LDH release, membrane-permeability dyes, caspase activity).
- Any statement that a change in a cultured cell line predicts a clinical effect.
- Micrometre-denominated measurements without a supplied calibration.
- A causal claim from a correlation between morphology and viability. Morphology
  is evaluated as a *predictor*, never as a substitute for the assay.

## 9. Safety and compliance

Before collecting data, confirm and document:

- Institutional approval for the cell lines and biosafety level in use.
- The provenance and permission status of the Tribonema extract, recorded in
  `permission_status`.
- For a competition entry, whether the work requires a Scientific Review
  Committee sign-off before experimentation, and keep the signed forms with the
  logbook. Cultured cell lines are not human or vertebrate animal subjects, but
  the tissue-culture and BSL forms still apply at most fairs.
- Waste handling for treated cultures and MTS reagent.

## 10. Records to keep

- A dated logbook of every session, including failures and discarded plates.
- The treatment key, stored separately until analysis.
- Raw images, unmodified, alongside the renamed analysis set.
- The `treatment_report.json` produced by each analysis run — it carries the code
  revision, dependency versions and environment used to produce those numbers.
