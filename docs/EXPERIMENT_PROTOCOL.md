# Tribonema experiment protocol

This describes the experiment TriboVision's analysis half is built to consume.
It is written so that someone else could repeat it, and so that a reviewer can
tell exactly which claims the design can and cannot support.

## 1. What is being asked

**Question.** Do visible morphological changes in cultured cells, measured
automatically from phase-contrast images, track the loss of viability caused by
Tribonema extract?

**Primary hypothesis.** Mean per-cell area at a fixed exposure time decreases
monotonically with Tribonema extract concentration.

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
| At least two experiment days | Leave-one-day-out is the only honest test of transfer | `leave_one_day_out` |

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
- Spearman correlation of each feature against concentration, overall and
  per experiment day, with Benjamini-Hochberg q-values across features.
- 4-parameter-logistic dose-response fits with bootstrap IC50 confidence
  intervals, flagged when the IC50 falls outside the tested range.
- Leave-one-day-out ridge prediction of viability from morphology, with a
  within-day label-permutation null and an exact permutation p-value.

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
