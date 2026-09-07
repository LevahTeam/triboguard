# What was decided in advance, and what was not

A result is only as trustworthy as the decisions that preceded it. This file
separates the choices fixed before the data could influence them from the ones
made afterwards, and lists every claim that had to be withdrawn once it was
tested properly. Both halves matter: the first is what makes the numbers
credible, and the second is the evidence that the first was actually enforced
rather than merely intended.

Every item here is checkable against the git history and the test suite.

## Fixed in advance

**The held-out split.** The official LIVECell test split (well C7, 60 images) was
chosen before any model was trained and never changed. The scaling experiment
grew the *training* pool from 79 to 304 images while the test manifest stayed
byte-identical, which `test_per_split_caps_hold_the_test_set_fixed_while_training_grows`
asserts directly. No number in this project was ever produced by choosing a
different test set.

**Group-level separation.** Splits are separated by well, and the leakage check
compares acquisition groups rather than file names, because LIVECell stores
several crops of one field of view under different names. Training refuses to
start on leaking splits; it is not a warning.

**The trend threshold.** `agreement()` has judged a segmenter "usable for trends"
at Spearman rho > 0.7 since before the three-class model existed. The first
version scored 0.65. The improved version scored 0.707 on one seed and 0.679 on
average across three, so it still does not reliably clear the bar. The threshold
was not moved either time, and the single passing seed was not reported alone.

**The confirmatory endpoint.** `PRIMARY_FEATURE` in the treatment analysis is a
single pre-specified shape feature. The multi-feature model is reported beside it
and labelled exploratory, because eight correlated predictors at 20-60 wells
roughly halves the power.

**Statistical commitments.** The well is the unit of analysis; fields within a
well are averaged first. Every comparison is blocked by experiment day.
Correlations are Spearman. Multiplicity is corrected with Benjamini-Hochberg and
uncorrected p-values stay visible. These are stated in
[RESEARCH_PLAN.md](RESEARCH_PLAN.md) and enforced in code.

## Tuned, and on what

Two free parameters were chosen, both on training data and never on the split
they were scored against:

| Parameter | Value | Chosen on |
|---|---|---|
| Cellpose cell diameter | 15 px | 4 training images |
| Three-class interior threshold | 0.7 | 6 training images |

The Cellpose sweep spanned 0.34 to 0.37 across every value tried, so that
parameter is nearly inert and is reported as such rather than presented as
careful tuning.

Model hyperparameters — learning rate, depth, channel width, epochs — were not
tuned at all. They were set once by convention and left. That is a limitation,
not a virtue: a tuned baseline might close some of the gap to Cellpose, and this
project cannot say by how much.

## Exploratory, and labelled so

The jamming analysis was not pre-registered. The two-regime split at confluence
0.5 was found by looking at a trajectory that did not behave as expected, and
then applied uniformly. The cross-cell-line comparison of shape index was not
predicted in advance. These are hypothesis-generating, and the honest next step
for any of them is a fresh dataset rather than more analysis of this one.

**The resolution-limit analysis is post-hoc.** It was written after the MCF7
shape-index recovery came back far below A172's, to rule out the explanation
that had to be excluded first: that the Cellpose diameter, tuned on A172, simply
did not fit the other lines. That explanation failed in an informative
direction — the tuned diameter is a *poor* match for A172, where recovery works,
and a *good* match for the lines where it fails — so cell size in pixels
survives as the candidate. `resolvable_near_threshold` and
`discretisation_residual`, which do the measuring, were written for the original
A172 analysis and are applied here unchanged; that is what makes this a test of
an existing prediction rather than a curve fitted to four points. It is still
post-hoc, it is still four cell lines, and it is reported as a hypothesis.

**The selection-cost analysis is post-hoc too.** It was written after noticing
that one arm of the diversity experiment saved its checkpoint at epoch 5 while
its comparator saved at 22-30. Nothing about it was planned; it exists because a
number looked wrong. It reads only the training histories that every run already
recorded, so it costs no re-training and can invent no data — but it also cannot
show that a better checkpoint would have scored better, because no checkpoint was
saved at the epoch it identifies. It bounds the cost of the selection rule. It
does not measure a model.

**So the diversity experiment is being run twice, and both results will be
reported.** The first run follows the protocol registered above, unchanged, and
its answer stands as the answer to the question that was actually pre-registered.
The second re-trains *both* arms with a checkpoint rule that averages interior and
boundary recall, because reusing the existing single-line checkpoints would
compare a model chosen by one rule against a model chosen by another and
reintroduce the confound it exists to remove.

The second run is not pre-registered and must not be presented as though it were.
It was designed after seeing one mixed run peak on boundary recall 29% of the way
through training, with interior recall still owing 0.18, where the single-line
runs peaked 65-75% of the way through with 0.03-0.05 left.

**One run, not the arm.** An earlier draft of this paragraph called the pattern
systematic on the strength of two seeds. That was a misreading: the second seed's
running maximum was inspected while it was still training, and it went on to
improve twice more and to select an epoch in the same range as the single-line
runs. The rule is capable of selecting a badly under-trained checkpoint, and did
so once; it does not do so every time. That is a weaker problem than the one
originally described, and it is still worth removing, because a three-seed mean
that contains one badly selected checkpoint is dragged down by it.

What protects the second run from being a fishing expedition is that the decision
rule is inherited unchanged from the first — diversity wins only if the
seed-and-image interval on the difference excludes zero — and that the rule was
fixed before either arm of the second run had produced a number.

**And the cell-size hypothesis died too, which is why the next one is a
prediction.** SHSY5Y came back at rho 0.801 for the three-class model — better
than A172's 0.707 — despite cells barely larger than MCF7's, where recovery
failed at 0.181. Size cannot be the explanation. What separates the two is how
much the shape index *varies between images*, measured from the polygons alone
with no segmenter involved: SHSY5Y spans sd 0.672 and A172 sd 0.305, while MCF7
manages 0.109. A rank correlation cannot detect tracking ability when there is
almost nothing to track; that is range restriction, and it is a property of the
cell line, not of the method.

This was written while SkBr3 was still running, so it stood as a genuine
out-of-sample prediction: **SkBr3 has sd 0.101, the narrowest of the four, so its
three-class recovery should fail, and it should fail for reasons that have
nothing to do with segmentation quality.** If instead SkBr3 clears the bar, the
range-restriction account is wrong and the honest move is to say so. The commit
that records this precedes the commit that reports SkBr3.

### The prediction was wrong

SkBr3 scored **0.791** and cleared the bar comfortably — the second-best of the
four lines, on the narrowest dynamic range of the four. Biological spread alone
does not predict recovery, and the account in the paragraph above is withdrawn.

Across all twelve cell-line-by-method combinations, spread alone correlates with
the observed rho at only +0.281 (p = 0.38), and measurement error alone at −0.231
(p = 0.47). Neither term predicts anything on its own. Their **ratio** correlates
at **+0.930 (p < 0.0001)**.

That is why SkBr3 passed. Its shape index barely varies between images, but the
segmenter measures it unusually precisely there — error sd 0.062, the smallest of
any line — so the signal-to-noise ratio is 1.63. MCF7 has almost the same
biological spread and nearly double the measurement error, giving 1.05, and it is
MCF7 that fails. Two lines that look alike on the quantity I predicted from
behave oppositely, and the quantity that separates them is the one I had not
isolated.

The method that produced this was committed *before* SkBr3 finished, so it was
not built to rescue the failed prediction. That is the only reason the corrected
account is worth more than the one it replaces — and it is still a fit to twelve
points from four cell lines, so it is a hypothesis, not a law.

## Registered before the run, not yet answered

Two experiments were specified, committed, and only then executed. This section
was written while both were still training, so the decision rules below could not
have been shaped by their results. The commit that introduced this section is the
proof: it precedes the commit that reports the numbers.

**Diversity versus volume.** Does a training set spanning three cell lines
transfer better than a single-line set of the same size? Both arms train on
exactly 304 images with identical hyperparameters and three seeds each (42, 1,
2); the mixed arm draws from A172, MCF7 and SkBr3, the control from A172 alone.
Both are scored on SHSY5Y, which neither arm ever sees.

- *Decision rule.* Diversity wins only if the seed-and-image bootstrap interval
  on the difference excludes zero. A gap smaller than the seed spread is not a
  finding, and will be reported as "no detectable difference" rather than
  quietly dropped.
- *Why it is worth running.* Volume is confounded with diversity in the scaling
  curve already reported: the 304-image arm saw both more images and more fields
  of view. Matching the size isolates the part that is about variety.
- *What would falsify the appealing answer.* If the mixed arm matches or loses,
  the honest conclusion is that at this scale the model is limited by capacity or
  by the representation, not by the narrowness of its training data.

**Does the mechanics result survive a change of cell line?** The claim that a
segmenter can recover tissue shape index well enough to track trends was measured
on A172 only. It is now being re-measured on MCF7, SHSY5Y and SkBr3, against
polygon-derived truth, with the rasterised ground truth included as a positive
control.

- *Decision rule.* The pre-existing rho > 0.7 bar is unchanged. A method counts
  as usable for trends on a line only if it clears 0.7 on that line. Lines are
  reported individually; no average across lines will be quoted as a headline,
  because the instance ceiling varies more than fourfold between them and an
  average would mostly measure which lines were chosen.
- *The control is load-bearing.* If the rasterised ground truth itself fails to
  clear the bar on a line, that line's failure is a limit of the measurement
  chain, not of the segmenter, and will be reported that way.

## Claims withdrawn after testing

This is the part worth reading. Each of these was stated, then retracted when a
better test was run. All are in the git history.

| Claim | What it became | What caught it |
|---|---|---|
| "0.952 vs 0.425 for the classical rule" | The real floor is a predictor that labels every pixel a cell, at 0.710 | Measuring the trivial baseline |
| A sign test at 1.7e-18 | 2.3e-10 over 33 acquisition groups | pseudoreplication: 60 crops are not 60 independent trials |
| "5 of 5 wells jam as they crowd" | 10 of 14 | Replicating across wells instead of one per line |
| "The three-class model beats a perfect binary mask" | Indistinguishable from it: +0.040 [-0.003, +0.078] | Paired bootstrap intervals |
| "Every step of the scaling curve is significant" | The 152 vs 79 step is half the seed spread | Three seeds per endpoint |
| "The bottleneck moved to boundary prediction" | It moved there and is data-limited, not architecture-limited | The scaling curve |
| Transfer looked *better* on unseen cell lines | Raw scores are not comparable; the ceiling varies 4.4x | Normalising by each line's ceiling |
| A power table for the experiment protocol | Did not survive simulation; within-day centring doubled the real power | Re-deriving it instead of trusting it |
| "The improved model now clears the 0.70 trend bar at 0.707" | Mean 0.679 ± 0.030 across seeds; 1 of 3 passes | Checking the other two seeds before claiming it |
| "Recovery fails on MCF7 because its cells are too small to resolve a perimeter" | Withdrawn: SHSY5Y has cells nearly as small and recovers better than A172 | Extending to a third cell line |
| "Recovery fails where the shape index barely varies between images" | Withdrawn: SkBr3 has the narrowest spread of all four and scores 0.791 | A prediction registered before the run, and falsified by it |

Eleven retractions is not a sign the work is unreliable. Every one came from
applying a stricter test to a number that had already been written down, and the
stricter test is the one reported. A project with no retractions has usually not
looked hard enough.

The last two are the ones to read together. Both were explanations for the same
observation, both were written down before the data that could refute them
arrived, and both were refuted. What replaced them — that recovery tracks the
*ratio* of biological variation to measurement error, at rank correlation +0.930
across twelve combinations, where neither term alone reaches significance — was
computed by a script committed before the second refutation landed. Getting an
explanation wrong twice in public and having the third one hold is a better
outcome than getting it right once by luck, and only the git history can tell
those two apart.

## Still unverified

- Whether the split-leakage observation about LIVECell's official train/val files
  is novel. That is a literature check, not a computation, and it has not been
  done.
- Whether the instance results hold on a second plate or a second microscope.
- Whether tuned hyperparameters would narrow the gap to Cellpose.
- Everything about Tribonema. No treatment image has been collected or analysed.
