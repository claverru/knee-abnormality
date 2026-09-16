# 2026-09-14 meniscus soft-coupling result

**Retain reflection/sigma 40 as champion.** The completed
`meniscus-soft-coupling` run selected epoch 7 (index 6), scoring
**0.890868573**, down **0.001491595** from the champion. Both meniscus targets
and validation BCE worsened. This candidate has no measured public score;
the champion's public score remains **0.936**.

| Selected model | Macro AUC | Medial meniscus | Lateral meniscus | Meniscus mean | Soft-target BCE |
|---|---:|---:|---:|---:|---:|
| Reflection/sigma 40 | **0.892360** | **0.915115** | **0.887095** | **0.901105** | **0.451855** |
| Meniscus coupling, coefficient 0.1 | 0.890869 | 0.908600 | 0.885439 | 0.897020 | 0.453034 |
| Candidate minus champion | −0.001492 | −0.006515 | −0.001656 | −0.004085 | +0.001179 |

Eight of twelve targets declined. The other changes were ACL −0.007156,
MCL +0.003024, medial OA −0.000846, lateral OA +0.002660, PF OA −0.005579,
Effusion −0.001176, Synovitis −0.001532, Baker's +0.003957,
Contusion −0.004987 and Fracture +0.001906. The medial/lateral OA pair improved
by 0.000907; the other eight targets averaged −0.001443. Unpenalized targets
can change because all tasks still train a shared representation.

Both runs completed ten epochs and selected epoch 7. Coupling trailed the
champion's macro AUC at nine of ten matched epochs, with epoch 2 the exception.
Final-epoch AUC was 0.889812 versus 0.891347; this is not solely a difference
in checkpoint selection. Training took 1,694.22 versus 1,699.52 seconds,
with 4,380 optimizer steps each. These recorded times do not measure inference
efficiency or establish a repeatable speed difference.

## Did the penalty affect the classifiers?

Yes. At the selected checkpoints, Euclidean distance between the meniscus
weight vectors fell from the champion's **0.734266 to 0.583997**, a **20.5%**
reduction. Weight MSE fell from 0.00210604 to 0.00133223, or **36.7%**;
cosine similarity rose from **0.202783 to 0.406304**. Medial/lateral vector
norms changed from 0.525655/0.630231 to 0.488421/0.575115.
The candidate's distance was 0.784023 before training and 0.586924 at epoch 10.
The penalty was active; this does not establish that the resulting alignment
is anatomically useful or that 0.1 is the optimal coefficient.

At selected epoch 7, average training BCE was 0.452911 and the separately
logged penalty was 0.00013715. Its small contribution to the scalar loss
did not imply negligible influence on classifier weights under AdamW.
Neither gradient attribution nor a same-runtime zero-coefficient control
was run. Independent attention means that the two vectors operate on
different pooled feature distributions; encouraging closer readouts can still
harm discrimination even though exact equality is never required.

## Predefined subgroup diagnostics

Among 344 studies with exactly one positive meniscus target (281 medial-only,
63 lateral-only), AUC for the medial-minus-lateral logit margin declined from
**0.969045 to 0.965712**. The diseased side received the higher prediction in
92.15% versus 90.70% of cases. This zero-margin ranking diagnostic is
sensitive to the unequal class counts and independent biases; it is not a
separately calibrated decision rule.

The lateral target improved when the medial target was negative
(548 studies, 63 lateral positives: **+0.004287 AUC**), but deteriorated when
medial disease was present (356 studies, 75 lateral positives: **−0.011910**).
It also improved without OA (+0.003001), while declining with OA (−0.006803).
Medial meniscus deteriorated in every predefined disease subgroup: −0.005518
without lateral disease, −0.012063 with lateral disease, −0.007312 without OA
and −0.010622 with OA. These overlapping strata do not isolate a causal
co-occurrence mechanism, and their AUCs do not simply average to full-fold AUC.

For the 198 exclusive-positive OA cases, margin AUC declined from 0.879085
to 0.847171 despite the full-fold OA pair gain. Aggregate target AUC and
within-study compartment discrimination remain distinct diagnostics.

| Acquisition vendor | Studies | Macro AUC delta | Meniscus-pair AUC delta |
|---|---:|---:|---:|
| GE | 201 | +0.000601 | +0.000582 |
| Philips | 255 | −0.004957 | −0.009348 |
| Siemens | 386 | −0.000269 | −0.003537 |
| Toshiba | 48 | −0.005294 | −0.021412 |

The remaining 14 studies comprise Canon 10, Fujifilm 3 and Hitachi 1; complete
macro AUC is undefined in each small group. The artifact retains all supports
and defined target results. Vendor is an acquisition proxy, not a site label.
These findings show heterogeneous effects but do not establish hidden test
composition, scanner causality or candidate leaderboard performance.

## Interpretation and reproducibility

This setting fails the predefined selection criteria: meaningful classifier
coupling was accompanied by lower intended-family and full-fold performance.
Keep the champion; do not automatically add OA or begin a coefficient sweep.
If continuing architectural transfer, the next distinct proposal is
[shared nonlinear disease features with independent final classifiers](../../research/medial-lateral-transfer.md#candidate-2-shared-disease-features-independent-decisions).
This is a negative result for coefficient 0.1 in this run, not a rejection of
all possible soft-sharing strengths or proof of worse public performance.

The 2,000 paired report-group bootstrap draws use 852 groups, seed 20260914,
with all draws defined. Candidate-minus-champion 95% percentile intervals
are [−0.003843, +0.000949] for macro AUC and [−0.009060, +0.000615] for the
meniscus pair. They quantify uncertainty conditional on these selected
predictions; they do not include training-seed or checkpoint-selection
variability. Crossing zero does not erase the measured competitive regression.
Gold-only AUC rose from 0.922156 to 0.924206 on just eleven studies, which
cannot resolve this decision; report-derived AUC fell from 0.891725 to 0.890082.

Saved configurations differ only in run name and the added loss section.
Both use seed 42, Steven v4 with gold overrides, the same `all_5fold_v3` fold 0,
3,503/904 membership, ReXNet initialization, reflection 0.5, sigma 40,
224-pixel/140-mm inputs, optimizer, batch 8 and 2× validation images.
Label, split, membership, training/validation-row and series identities match.
Preprocessing source and mapping metadata differ because the champion predates
the [CoAtNet adapter refactor](../../modeling/coatnet-224-spatial-backbone.md).
That refactor was statically reviewed for unchanged ReXNet behavior; there is
no contemporaneous unregularized rerun or bitwise training-parity claim.
The candidate records parent commit `ea19772` and its uncommitted source snapshot.

The [comparison artifact](../../../outputs/analysis/meniscus-coupling-20260914/comparison.json)
contains all target, conditional and vendor results, epoch curves, CPU checkpoint
weight diagnostics, source changes, configurations and file hashes. The
[CPU analysis](../../history/meniscus-coupling-sources/README.md) left the script catalog
on 2026-09-16, when the candidate's checkpoints were deleted, so it can no longer be rerun.
Saved predictions reproduced both recorded selected AUCs. Analysis performed no model execution, GPU access, tests,
new training or submission.

The experimental loss, configuration field and training/profiling integration
were removed on 2026-09-14, restoring the champion code at `ea19772`.
The run's original source archive, effective configuration, checkpoints and
analysis artifacts remain intact; the historical CPU analysis owns its weight
diagnostics and does not depend on the removed training loss. See the
[archived objective and reproduction requirements](../../modeling/retired-side-and-compartment-variants.md#meniscus-classifier-coupling).
