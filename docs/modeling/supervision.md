# Supervision

Evidence: unverified working notes, explicitly identified official/public
sources, and the local CSV audit completed on 2026-09-09;
see [provenance](../research/provenance.md).

The supplied reference labels cover only 58 studies. Keep missing labels
distinct from negatives when preparing supervision for the remaining studies.
Public report-derived labels are a starting point to investigate; reported
extraction improvements have not consistently improved downstream models.

**Selected for the first baseline, 2026-09-10:**
[Steven Lee's v4 blend](../public-labels/source-consistency-and-first-baseline-choice.md#implication-for-the-first-baseline),
using the pinned file from Kaggle dataset version 6. Keep official reference
labels separate and apply them only to eligible training rows under the fixed folds.

The current [all-study protocol](../validation/project-validation-protocol.md)
assigns all 4,407 studies to five folds, stratifying effective labels, gold
labels and gold status. Training uses gold where available and the selected
report-derived probabilities otherwise. All nonmissing targets contribute to training;
the [training evidence mask](#training-label-masking) was tried and rejected.
The split's frozen effective label
table supplies validation targets, with a fixed 0.5 cutoff for AUC and separate
gold/report-derived diagnostics. Label-source experiments change training
supervision while preserving the evaluation table.

The [2026-09-10 public-label audit](../public-labels/INDEX.md) now pins and measures
Steven Lee's v1/v2/v4, Pilkwang v2, and the lixin73 component of Steven v4.
All cover the 4,349 report-only studies. Pilkwang omits one reference study,
so its extraction score uses 57 rather than 58 studies. Their uncertainty
encodings differ; consult the audit before constructing training targets.

The [report-state audit](../public-labels/uncertainty-and-confidence-encoding.md#report-state-audit-and-implications-for-supervision)
measures disease despite report silence, information lost by collapsing soft scores and
reporting-status associations in saved image predictions. Auxiliary report-state supervision
is proposed but unimplemented; the current disease targets and loss remain in force.

## Fixed class-balanced BCE

**Measured 2026-09-19:** `balanced-bce` reached 0.895282, +0.008796 over the closest same-seed
historical reference ([result and attribution limits](../validation/results/2026-09-19-balanced-bce-result.md)).
It is always on; `trainer.class_balance_exponent` (required) damps it, and 0 gives unit weights,
ordinary masked soft-target BCE. The
[research note](../research/auc_loss_approaches.md#weighting-and-sampling-design-space) explains
the hypothesis and alternative weighting/sampling methods.

For each label, compute its positive fraction `r_c` once from observed targets in the actual
training subset, after official-label overrides and any training-study limit. Count `y >= 0.5`
as positive and `y < 0.5` as negative; require both groups. A damping exponent
`trainer.class_balance_exponent` (default `1.0`, range `(0, 1]`) controls weighting strength:
`rho_c = ((1 - r_c) / r_c) ** k`, `w_neg_c = 1 / (r_c * rho_c + 1 - r_c)`, `w_pos_c = rho_c * w_neg_c`.
`k = 1.0` reproduces the previous fixed weights (`w_pos_c = 1 / (2 * r_c)`,
`w_neg_c = 1 / (2 * (1 - r_c))`); smaller `k` damps the pos:neg ratio toward 1:1 (e.g. `k = 0.5`
gives ratio `sqrt((1 - r_c) / r_c)`). Only training rows enter these counts, and the weights
remain fixed throughout the run. For any exponent, each label's mean weight over the training
subset is one. **Measured 2026-09-21:** on the public leaderboard, k=0.5 scored 0.945, unweighted
0.943 and k=1 0.940. Fold-0 validation ranked them in the opposite order
([result](../validation/results/2026-09-21-class-balance-exponent.md)).

Multiply the entire unreduced BCE cell by its group's weight and retain the mean over observed
cells. Targets remain soft and every observed cell remains active. There is no per-batch weight
estimation or normalization by the batch's sum of weights. Over the training subset, each label
has mean weight one and equal total weight in the thresholded positive and negative groups.
This differs from `pos_weight`, which scales only the positive term inside BCE, including the
positive component of soft negative targets such as 0.25.

Enabled runs save `fold_<N>/class_balance.json` alongside their effective run configuration.
Its version-1 table also records a top-level `exponent`, alongside each target's observed,
positive and negative counts, positive fraction and the two fixed weights, in model target order.
Checkpoints retain the same table in `hyper_parameters.class_balance`; resume compares the full
table against the current configuration and the selected training-input identity before writing
the new weighting artifact, so checkpoints saved before the exponent existed cannot resume.
Checkpoints without that metadata require their archived training implementation for resume.
Device weights are nonpersistent buffers, so they add no inference
model-state entries.

The option changes training loss only. Study exposure, validation BCE/AUC and inference scoring
retain their existing behavior. Weighting can amplify rare-label errors; the first run improved
ten of twelve label AUCs against its historical reference. The separate [cell-masking proposal](../backlog/small-changes.md#report-label-supervision)
drops selected majority cells and remains unimplemented.

## Training-label masking

**Rejected on 2026-09-13:** the completed `masked-labels` run reduced full-fold
AUC by 0.011455 against reflection. The baseline again trains on all nonmissing
soft targets, with official gold overrides. See the canonical
[masking result](../history/sharing/2026-09-13-label-masking-result.md).

The experiment excluded report-derived training targets whose original Steven v1
value was 0.5, preserving gold labels and the stored Steven v4 probabilities.
It used pinned v1/v2 evidence tables and averaged BCE over retained entries;
batches with no observed targets skipped their training step. Fold membership
and validation targets were unchanged.

Of 3,503 training studies, including 47 gold, masking excluded **10,548** target
entries and retained **31,488**, including all **564** gold targets. Forty-six
studies had no supervised targets left. All 904 validation studies and 10,848
validation entries were retained, with the same complete label-row hash as
reflection. The [label-loading summary](../../outputs/analysis/label-masking-20260913/label-summary.json)
records source hashes, identities and per-target counts.

The policy field, evidence requirement, training-mask application, empty-batch
skip and additional checkpoint identity field have been removed. Reproducing the
rejected implementation requires its archived
[source](../../outputs/training/masked-labels/source_snapshot.zip) and
[configuration](../../outputs/training/masked-labels/effective_config.json); its checkpoints
and recorded identities remain unchanged.

## Training with Tonylin soft labels

**Historical reproduction:** prepared on 2026-09-13 for the
[completed fold-0 comparison](../validation/results/2026-09-13-tonylin-label-comparison.md),
which scored 0.921 publicly versus Steven v4's 0.929. The active labels are now the
[combined labels](../public-labels/combined-labels.md); the procedure below preserves the Tonylin
experiment.
The exact historical run needs its archived source; the procedure below trains the same label
setup with the current code.
The [label audit](../public-labels/2026-09-13-new-public-label-comparison.md)
owns the source-quality evidence and its limitations.

`data/labels/prepared/tonylin-soft-v3/labels.csv` contains all 4,407 study IDs
and the twelve competition target columns. Preparation selects the audited
release-3 `*_soft` values, renames the columns and sorts by study ID. Values
are preserved; the publisher's hard labels and fold assignments are not used.
The [training manifest](../../configs/labels/tonylin-label-sources.json) pins the prepared file,
original CSV, metadata, column mapping and preparation script. The prepared
CSV SHA-256 is
`aa852eebb32e6ef95bbf1736e4f73b40699bb3dd247049606e8f1696bca4e216`.

In a copy of `configs/training.yaml` saved as `configs/tonylin-soft-v3.yaml`, change these
two values, retaining the other path settings:

```yaml
paths:
  label_manifest: labels/tonylin-label-sources.json

labels:
  source: tonylin_soft
```

Keep `validation.split: all_5fold_v3` and `validation.folds: [0]`. Training
loads our existing `studies.csv` membership and frozen `targets.csv`;
official labels still override source values on eligible gold training rows.
Preparation successfully loaded **3,503 training studies (47 gold)** and
**904 validation studies (11 gold)** through `load_fold_labels`. The local
`data/labels/prepared/tonylin-soft-v3/preparation.json` records their source,
membership, protocol and evaluation-target hashes.

After editing the YAML, launch a fresh run with:

```bash
rtk proxy uv run --locked just train tonylin-soft-v3 configs/tonylin-soft-v3.yaml 0
```

`just train` does not generate folds. Do not run `just prepare-folds` for this
label-source comparison: stratification and frozen evaluation targets depend
on the selected labels, so rebuilding them with Tonylin would propose a
different protocol. The preparation command refuses to overwrite conflicting
existing split artifacts, but a new split directory would also change the
comparison endpoint. Reuse the saved split throughout the experiment.

The [preparation utility](../../scripts/tools/prepare_tonylin_labels.py) uses only
tabular metadata and the existing fold-label loader. To reproduce into fresh
locations, run from the repository root:

```bash
rtk proxy uv run --locked python -m scripts.tools.prepare_tonylin_labels --output-dir data/labels/prepared/tonylin-soft-v3-reproduction --output-manifest configs/labels/tonylin-label-sources-reproduction.json
```

The utility leaves the input YAML and split files unchanged. Preparation ran
no training, image decoding or tests. The subsequent user-trained model and its
submission are covered by the completed result linked above.

## Official labeling context

The [official label criteria](../metadata/official-label-criteria.md) hold the
host's verbatim rubric: examination-level reference labels from two MSK
radiologists with a third adjudicating, borderline findings graded negative,
high-grade ACL/MCL tears distinguished from mild degeneration or sprain,
meniscal surface-reaching tears from intrasubstance degeneration, moderate/large
effusions and Baker's cysts from minor findings, and OA requiring a substantial
area of high-grade cartilage loss. Use the complete rubric when designing
extraction, rather than marking every abnormality mention positive.

That document also records two host statements that constrain supervision: the
image-derived label is authoritative wherever report text disagrees, and a
negative means the finding was annotated absent rather than unannotated, so
report silence is an unobserved label and not a missing one.

The [official data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)
confirms that reports are training-only and are not provided for test studies.
Keep report extraction offline and train the submission model to predict from
available image inputs.

The host explicitly permits hosted LLM/API processing of competition reports
for label extraction, subject to accessibility, cost, competition rules, and
service terms. Such processing is not by itself prohibited private sharing.
The clarification also distinguishes straightforward dataset registration from
substantial institutional approval barriers; non-commercial restrictions alone
do not disqualify a dataset under the competition's rules.
[Host clarification, including 2026-08-27 replies](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965).

**KneeCoT is specifically disallowed** under the host's 2026-08-25 ruling because
of its institutional agreement requirement. The host allows for reconsideration
if access terms change; no reversal appeared in the retrieved thread.
[KneeCoT ruling](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109).

For actual public label-table comparisons, missing-label experiments and
negative results, read [public supervision evidence](../research/public-research.md#report-derived-supervision).

## Available supervision

The [local metadata audit](../metadata/official-csv-audit.md#inventory-and-acquisition) confirms
4,407 training studies, each with a nonblank `Report`:

- 58 studies have all 12 reference labels, each binary `0` or `1`.
- 4,349 studies have all 12 labels missing; no studies are partially labeled.
- Missing cells mean absent supervision, not a negative label. Preserve a
  label-observation mask; keep any inferred label source/version separate
  from the reference labels.
- Exact report text repeats in 46 groups covering 177 studies. Identical
  wording does not establish the same patient or duplicated images. No
  conflicting reference labels were found among identical reports within
  the small reference-labeled subset; this is not evidence of label correctness.

In these notes, **gold** means supplied reference labels and **silver** means
labels inferred from reports or other model predictions. The downloaded
example `test.csv` contains only study IDs, with no reports or target labels.

Unverified context and research leads:

- Reports were said to be unavailable at inference, so text extraction is an
  offline supervision step for the image model.
- This [public-label discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932)
  is a relevant public resource. The [public review](../research/public-research.md#report-derived-supervision)
  identifies concrete sources and author-reported version comparisons.

## Reference label distribution

Observed in `train.csv`, in the column order of `sample_submission.csv`:

| Target | Positive | Negative |
|---|---:|---:|
| ACL | 24 | 34 |
| MCL | 9 | 49 |
| Medial Meniscus | 26 | 32 |
| Lateral Meniscus | 23 | 35 |
| Medial OA | 15 | 43 |
| Lateral OA | 11 | 47 |
| PF OA | 21 | 37 |
| Effusion | 35 | 23 |
| Synovitis | 27 | 31 |
| Baker's | 12 | 46 |
| Contusion | 19 | 39 |
| Fracture | 18 | 40 |

Each target has 58 observed labels and 4,349 missing labels, with no values
outside `{0, 1}` among observed cells. These counts describe the supplied
reference subset, not prevalence across all training studies. MCL has only
nine positive examples, so check target coverage when designing
[validation splits](../validation/INDEX.md). External silver-label files and the official
evaluator were not part of this CSV audit.

## Label extraction results

These unverified scores were recorded as AUC on the 58 gold-labeled studies,
not accuracy or image-model public leaderboard scores. Exact checkpoints and
evaluation code are unavailable, and model names may be imprecise.

| Label source | Reported gold-label AUC on 2026-08-28 |
|---|---|
| Public regex extraction | 0.86 |
| Qwen3.8 27B | 0.89 |
| Each of two `ox alpha (glm 5.3 flash)` runs | About 0.90 |
| Combined extraction ensemble | 0.91 |
| Kimi K3 alone | 0.91; considered too expensive to run broadly |

One unnamed target was around 0.75 AUC; the rest were described as relatively
uniform. Per-target tables are unavailable.

An earlier local setup used the name `EschaLabs/Qwen3.8-27B-Escha-W2`, described
as a 2-bit model occupying about 10 GB, with thinking disabled and approximately
batch size 8 on an RTX 4090. These are reported setup details, not verified model
specifications. The exact checkpoint has not been identified.

## Downstream benefit changed over time

- **Early positive result, 2026-08-28:** adding two GLM label sources to regex
  plus Qwen reportedly moved the image-model public score from 0.918 to 0.923.
  This was associated with improved extraction AUC on gold labels.
- **Later qualification, 2026-09-08–09:** further label improvements reportedly
  did little for image-model performance. Models trained on improved labels
  also seemed more correlated and contributed less to ensembles.

These describe different stages and should both be retained. The available
evidence does not establish that label work always helps or never matters. Preserve
the label version when comparing training runs and distinguish label-extraction
quality from downstream model and ensemble quality.

## Attempts with little or no reported benefit

- DSPy/GEPA prompt optimization, more thinking, English translation, and
  bilingual prompting did not improve extraction in the reported attempts.
  A simple prompt with thinking disabled was preferred.
- Training a separate NLP label predictor performed worse than regex/LLM
  extraction.
- Smoothing labels where extractors disagreed left the public score unchanged.
- An LLM judge was tried, but no isolated result was provided.
- Adding textbook material was considered but not demonstrated. Matching the
  report-labeling methodology using the report text remains the stronger lead.

## Refinement and earlier eligibility uncertainty

Using out-of-fold image-model predictions reportedly improved inferred labels
against gold labels. Later analysis identified circular reasoning in attempts
to create more strong labels. See [validation](../validation/local-to-public-score-relationship.md)
before treating these improvements as independent evidence.

Hosted LLM eligibility was initially uncertain. The
[official clarification above](#official-labeling-context) resolves that
uncertainty, subject to its conditions. One unidentified external dataset was
considered potentially usable but probably not worthwhile.
