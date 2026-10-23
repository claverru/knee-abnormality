# 2026-09-27 public label investigation

Investigation checked on **2026-09-27**. Production labels remain `combined_v1`.
The investigation left production labels, code and training configurations unchanged and ran
no image-model training or tests. This document records the completed investigation and its
recommended follow-up.
The [isolated pilot](../../history/qwen-lateral-labels.md) now implements the fixed blend;
its prepared table preserves gold source rows, and image-model training remains unstarted.

**New community labels exist, but no whole-table replacement is justified.** The strongest
new-source candidate for a controlled experiment is a fixed 50/50 blend of Qwen and the current
**Lateral OA** target, preserving the other eleven targets. Its gold label-table macro AUC rises
from 0.907537 to 0.910439, but the paired 95% interval includes zero. Its ranking also improves
on two separate report-review panels. These findings justify a bounded experiment, not adoption. The gold score evaluates a
hypothetical blend before overrides; the implemented silver-only table preserves gold source rows.

## How our current labels were created

The canonical recipe and original selection evidence are in
[Combined labels](../combined-labels.md). The investigation read that recipe,
its source manifests and preparation/analysis implementation before comparing alternatives.

- Steven v4 supplies eight targets: both menisci, Medial and Lateral OA, Effusion, Synovitis,
  Baker's cyst and Contusion.
- ACL, MCL, PF OA and Fracture use the per-study arithmetic mean of available values from nine
  public sources: Pilkwang v2, Laymond v2 filled, Zhen v7, Yunus, JEV text, Josh, Ramit,
  Dreaddevelopment and Vmohit. At least five cover every study.
- Encodings matter: Laymond uses its positive-probability `__confidence`; Josh becomes
  `0.5 ± confidence / 2`; Vmohit positive is 1, negative/below-threshold is 0, and
  uncertain/unmentioned is omitted. These values are not interchangeable confidence scales.
- The original investigation removed obvious duplicate families. A target switched from Steven
  only if the consensus beat it in over 90% of gold-study bootstrap draws, under every
  leave-one-source-out check, and after excluding gold-tuned JEV and Josh.
- All 4,407 studies are covered: 58 official gold and 4,349 silver. Official gold overrides the
  prepared soft targets when consumed by training. All scores here compare **raw source
  predictions before that override**, so copied gold cannot manufacture a perfect score.

The prepared table scores **0.907537** gold macro AUC versus **0.892707** for Steven v4.
Gold already helped select this recipe, and some extraction prompts used gold. It is not an
untouched validation set. The [source manifest](../../../configs/labels/combined-label-sources.json)
pins the current inputs; the [first image-model result](../../validation/results/labels-and-losses/2026-09-21-combined-labels-first-run.md)
records its subsequent training result.

## What was searched and what is new

Authenticated Kaggle API 2.0.2 searches covered labels, reports, pseudo-labels, soft/weak labels,
LLM extraction, annotation, probabilities, calibration and corrected targets. Updated searches
were paginated back to 2026-09-22, and relevant public notebooks were inspected for provenance.
The initial search found 75 dataset references and the expanded search 100; these are discovery
counts, not 100 distinct label tables. Downloaded notebook code was not executed.

The live official competition `train.csv` is byte-identical to our local copy. Its SHA-256 is
`8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33`.
**No new official gold labels were found.** Competition context remains the
[RSNA Knee Abnormality Detection competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

The following are community releases or newly inspectable files as of the check date. Dates
refer to dataset updates, which do not establish that every contained file is new.

| Source | Updated | Coverage and measured result | Assessment |
|---|---|---|---|
| [Qwen3-8B labels](https://www.kaggle.com/datasets/xhaitanya/qwen38b-labels-rsnaknee) | Sep 26 | 4,406 studies, all 58 gold; raw macro 0.879749 | One silver study missing. Model named in title; extraction prompt undocumented. Useful Lateral OA candidate. |
| [Bhavesh LLM labels](https://www.kaggle.com/datasets/bhaveshkolhe/knee-llm-labels) | Sep 24 | Two shards covering 4,407 studies; raw macro 0.878902 | Model/prompt undocumented. Strong gold Lateral OA gain, weaker report-panel and multilingual evidence. |
| [Nartaa HPO assets](https://www.kaggle.com/datasets/nartaa/rsna-knee-hpo-assets) | Sep 25 | Raw `train_labels_v1.parquet` covers 4,407; Gemini 3 Flash macro 0.887624 | Four previously downloaded silver files are byte-identical to Sep 22. New raw predictions agree 98.94% with Vmohit binary calls on comparable silver cells. |
| [Saisahana radiologist-report labels](https://www.kaggle.com/datasets/saisahanas/rsna-knee-radiologist-report-labels) | Sep 24 | Available-cell macro 0.833304; 1,945 missing cells | Notebook reuses Lixin, Steven v1 and Pilkwang, mostly hard Lixin targets. Per-target gold coverage 49–58; score is not directly comparable with full-coverage scores. |
| [Opus/Sonnet labelled train](https://www.kaggle.com/datasets/dhiyaneshbabu/rsna-knee-labelled-train) | Sep 26 | All official gold copied; 84 missing silver cells | Gold score cannot measure extraction quality. |
| [Posterior-calibrated labels](https://www.kaggle.com/datasets/jaygautam007/knee-labels-posterior-calibrated) | Sep 24 | All official gold copied | Calibration provenance unresolved; cannot benchmark against gold. |
| [Yehezkiel pseudo-labels](https://www.kaggle.com/datasets/yehezkielhaganta/psuedo-labeling-knee) | Sep 24 | Gold copied; 20,557 silver cells use `-1` | `-1` is unmentioned, not a negative probability. Cannot benchmark extraction on copied gold. |
| [Manual Label Boost](https://www.kaggle.com/datasets/docdsandeep/rsna-knee-manual-label-boost) | Sep 27 | 200 selected studies: 58 gold, 47 reviewed silver, 95 normal silver | Not new official gold; reviewed rows contain explicit report conflicts. See below. |
| [Hosseini pairwise JEV](https://www.kaggle.com/datasets/hosseini7798/rsna-labeling-pairwise-comparison-typesafe-jev) | Sep 22 | 1,653 pairs, exactly 58 choose 2, covering gold only | Borda macro 0.898574; normalized and graph least-squares methods 0.889763/0.894693. No new silver targets. |

Qwen and Bhavesh retain genuine predictions on gold rather than copying the annotations.
Their lower whole-table scores do not preclude a useful individual target. Nartaa's high
agreement is evidence of redundancy, not proof of a shared extractor or identical prompts.
Other discovered releases included model weights and a regex extraction notebook, without a
strong new full-coverage label table.

### Why the new manual labels should not be applied automatically

Manual Label Boost was updated on Sep 27 at 12:48:55 UTC. All 95 selected normal studies already
have all twelve current targets below 0.5; the largest current value is 0.4 and their mean is
0.042959. They add no new binary negatives, although hard-zeroing would alter soft supervision.
The release does not document image adjudication, and its sparse CSVs are not full replacements.

Among its 47 reviewed silver studies, the following report conflicts were inspected:

| StudyInstanceUID suffix | Proposed annotation | Report conflict |
|---|---|---|
| `11347466940974045182979684641300336177` | Effusion 0 | Report explicitly describes a large joint effusion. |
| `10273861326753707436672556008991334542` | Medial Meniscus 1; Contusion 1 | Spanish report says the finding does not reach the articular surface and has no tear features; bone marrow is normal. Soft-tissue contusion is not bone contusion. |
| `10422130685453254363700365908336246863` | Lateral OA 1 | Femorotibial cartilage is normal; described superficial findings concern the lateral patella/trochlea, a different compartment. |

These examples establish a reason to reject automatic application. They do not estimate the
error rate of all 47 studies or establish image truth. “Manual” is not a replacement for a
documented adjudication protocol.

## Experiments beyond whole-table comparisons

The CPU investigation screened **403 variants**. Seven source pools combined the original nine
with Qwen, Bhavesh and Nartaa, included Steven, or excluded gold-tuned JEV/Josh. Six aggregation
methods were evaluated: mean, median, 20% trimmed mean, mean logits, mean empirical ranks
remapped to current silver quantiles, and equal weighting of correlated source families.
Family clustering used per-target silver binary kappa ≥ 0.90 with complete linkage.

The screen varied target coverage (all twelve, current four, Steven eight), blend weights
0.25/0.5/1, and gates for ambiguity, strong disagreement and Steven's quarter-valued cells.
Additional focused checks used Steven v1's actual unmentioned cells, clipping/source-removal
sensitivity, held-pair selection, shared reliability calibration and existing image predictions.
No claim of source independence follows from the clustering.

| Comparison | Gold macro AUC | What it establishes |
|---|---:|---|
| Current `combined_v1` | 0.907537 | Reference raw label table |
| Best whole recipe in the 403 screen | 0.911897 | Bhavesh weight 0.25 on Steven v4 quarter-valued cells; zero silver threshold flips |
| Best independently chosen target from each of 403 variants | 0.928854 | Optimistic same-gold cherry-picking |
| Per-target variant selection with each tested gold pair withheld | 0.910014 | Much smaller gain after partial protection against selection |
| Whole-recipe selection with each tested gold pair withheld | 0.902446 | Broad recipe selection generalizes worse than the current table in this diagnostic |
| Same-original-source selection, current four targets, pair held out | 0.908724 | Small residual gain from changing aggregation |
| Expanded-source selection, current four targets, pair held out | 0.908446 | Adding sources does not improve this selection diagnostic |
| Shared source-reliability logistic calibration, pair held out | 0.901445–0.906260 | Learned reliability weights do not beat the current table |

For held-pair diagnostics, both the positive and negative study were removed across all targets
before selecting a variant, then their ordering was evaluated. These scores assess selection
procedures; they are not scores of a single deployable table. The candidate family was designed
after viewing gold, so this is not pristine nested validation. Reliability models used shared
source coefficients, target intercepts and fixed regularization `C=0.1`, with probability or
logit features; there was no image-model training.

Bootstrap comparisons used 10,000 paired gold-study resamples, seed **20260927**, requiring
both classes for each target in a valid macro draw. The intervals describe sampling uncertainty
conditional on the examined recipe, not search uncertainty or future training performance.

## Focused candidates and their failure modes

| Candidate | Gold macro | Delta | Paired 95% delta interval | Silver calls changed at 0.5 |
|---|---:|---:|---|---:|
| Qwen 50/50, Lateral OA only | 0.910439 | +0.002901 | [−0.000378, +0.007212] | 181 |
| Bhavesh 50/50, Lateral OA only | 0.914066 | +0.006528 | [−0.000117, +0.015212] | Not retained in the focused effect table |
| Existing nine-source logit mean, Fracture only | 0.909737 | +0.002199 | [0.000000, +0.006449] | 8 |
| Bhavesh 25% on actual Steven v1 unmentioned cells | 0.913291 | +0.005753 | [−0.002135, +0.014393] | 17 |
| Qwen Lateral OA plus Fracture logit mean | 0.912638 | +0.005100 | [+0.000690, +0.011062] | 189 |

### Qwen Lateral OA: the best bounded new-source experiment

Lateral OA gold AUC rises **0.832689 → 0.867505**. The gain is positive after removing any one
gold study; removing its top two contributing studies still leaves a target gain of +0.013002,
and removing the top three leaves +0.004831. Bootstrap probability of a positive macro delta
is 0.9503. This is a ranking gain: at 0.5, gold false positives increase 8 → 9 and false
negatives remain 2.

On silver, the positive rate changes **27.386% → 26.167%**, 181 calls cross 0.5, and mean
absolute Lateral OA probability change is 0.084360. Qwen's one missing silver study must retain
the current value. This changes supervision and potentially class weights; unchanged hard
calls elsewhere do not imply unchanged soft losses.

Two additional report-review panels favor this candidate over the larger Bhavesh gold gain:

| Lateral OA report panel, excluding official gold | Observed cells / positives | Current AUC | Qwen half-blend | Bhavesh half-blend |
|---|---:|---:|---:|---:|
| Davidadekanmi human review | 61 / 6 | 0.993939 | 1.000000 | 0.981818 |
| JEV additional report review | 13 / 3 | 0.933333 | 0.966667 | 0.900000 |

The underlying panels contain 98 and 16 non-gold studies respectively, but not every target
was annotated. They are small, selected, report-based challenges; displayed extractor labels,
author tuning and overlap with existing source development prevent calling them blind,
independent image validation. Their near-perfect report AUC does not resolve report/image gaps.
Six corroboration or Nartaa-veto gates did not consistently improve on the plain Qwen blend.

### Why the larger Bhavesh gain is less convincing

Its Lateral OA gold AUC is **0.911025**, with bootstrap probability 0.9714 of a positive macro
delta. But one influential gold-positive case describes normal cartilage, and both report
panels worsen. Among 51 Greek silver reports with high current Lateral OA probabilities,
Bhavesh makes 17 very low calls below 0.1; Qwen makes none. These are disagreements needing
review, not 17 proven errors. Only three gold reports are Greek and three Cyrillic, leaving
poor protection against multilingual failures.

Exploratory language/format stratification also found weaker agreement with the old-source
median in Greek and Cyrillic narratives. Agreement is not correctness, and Latin-script
grouping is not language identification. An undocumented extraction pipeline plus these
blind spots makes the larger small-gold gain insufficient to prioritize Bhavesh.

### Fracture logit pooling: interesting but fragile

This replaces the arithmetic mean with `sigmoid(mean(logit(clip(p, 0.01, 0.99))))`, using
available original-source values. Fracture AUC rises **0.906944 → 0.933333**; gold threshold
errors stay at five false positives and four false negatives. Silver positive rate changes
6.070% → 6.208%, with mean absolute probability change 0.034499.

However, one gold-positive study accounts for **14 of 19 net improved pair orderings**. Its
Greek report describes contusion and ligament injuries without mentioning a fracture. Its
target actually falls 0.136609 → 0.107141; negatives fall further, creating the AUC gain.
An older out-of-fold image teacher predicts 0.840823 for this study, illustrating the distinction
between report extraction and image truth.

Clip bounds 0.001/0.01/0.02/0.05/0.1 yield Fracture AUC
0.930556/0.933333/0.925000/0.906944/0.883333. Removing JEV alone gives 0.901389,
below the current target; removing both JEV and Josh gives 0.923611. This does not satisfy
the original all-source-removal robustness standard against the current target.
The combined Qwen/Fracture interval above excludes zero, but both components were selected
on the same gold. It is not independent confirmation and should not be the first experiment.

### Unmentioned-cell blending and calibrated-source lineage

Quarter-valued Steven v4 cells are an imperfect proxy for Steven v1 silence: eight v4 quarter
cells were not unmentioned in v1, while 3,353 v1-unmentioned cells were not v4 quarters.
The corrected Bhavesh gate scores 0.913291 with 17 hard flips, but its interval includes zero.
It suggests that changing missing-finding priors can improve ranking without many threshold
changes; it does not reveal new ground truth or justify a broad rewrite.

For the posterior-calibrated release, monotone, linear and logit reconstructions from existing
sources did not fully explain its scores. Example cross-validated reconstruction R² values were
0.406/0.339 for Lateral OA and 0.750/0.700 for MCL with raw/logit features. This does not prove
independent extraction or image-derived signal. Copied gold still blocks a trustworthy benchmark.

Existing image-teacher blends were revisited without training. Restricting a half-blend to
Steven's eight targets scores 0.927529 on gold, but this teacher already failed to establish
a public improvement in the [image-blend result](../../validation/results/labels-and-losses/2026-09-23-image-blend-labels-result.md).
Teacher regeneration remains deferred; its high label-table AUC does not reopen a rejected
adoption. Prior blanket low-grade consensus suppression also reduced gold macro
0.9075 → 0.9044 ([report attributes](../report-attributes.md)).

## Recommended follow-up

**Keep production `combined_v1`.** If the user starts a label experiment, use one fixed
intervention: `new_Lateral_OA = 0.5 * current_Lateral_OA + 0.5 * Qwen_Lateral_OA` on silver,
falling back to current labels for the missing study. Preserve the other eleven targets and
official gold. This is a proposed training-label change, not a rewrite of evaluation targets.

Before training, freeze the downloaded Qwen revision/hash and review a bounded untouched panel
of disagreements, including high-confidence OA positives, explicit negatives, compartment
confusion and Greek/Cyrillic reports. Keep a random panel separate from enriched hard cases;
do not tune the blend weight or gates to the known gold examples.

Then compare with a fresh control under the current adopted recipe, identical fold, seed,
sampling and training budget, with frozen validation labels and submission procedure. Record
per-target hard/soft AUC, the label manifest, changed positive rates and class weights. If
weights move materially, a fixed-original-weight sensitivity can distinguish target changes
from reweighting. Apply the existing controlled-comparison rule; label-table AUC alone cannot
authorize adoption. Training remains user-launched and no run is queued by this document.

This continuation became the [Qwen Lateral OA pilot](../../history/qwen-lateral-labels.md) and was
superseded by the [label rebuild](../label-rebuild/INDEX.md).
Fracture pooling, missing-prior rewrites, automatic manual-label application and renewed teacher
blends are diagnostic findings, not additional scheduled experiments.

## Evidence and reproducibility

The repository HEAD during investigation was `c19c64ddf8733ac368d2ae8fe0140c1f3450b4ff`, with
unrelated local changes already present. Production input fingerprints, rather than HEAD alone,
identify the compared tables. Evidence is preserved locally under the Git-ignored
`outputs/analysis/p/public-label-investigation-20260927/`:

- Investigation summary (`outputs/analysis/p/public-label-investigation-20260927/deep/investigation_summary.json`, deleted),
  input hashes (`outputs/analysis/p/public-label-investigation-20260927/deep/input_fingerprints.json`, deleted)
  and dataset-discovery metadata retain measured scores and source identities.
- 403-variant screen (`outputs/analysis/p/public-label-investigation-20260927/deep/aggregation_screen.csv`, deleted),
  candidate definitions, source families/kappa, held-pair selections and reliability outputs
  record the aggregation comparisons.
- Focused candidates (`outputs/analysis/p/public-label-investigation-20260927/deep/focused_candidates.json`, deleted),
  bootstrap intervals, training-target effects, driving cases, report challenges, manual cases,
  language diagnostics and guarded-OA results retain the follow-up evidence.

Original downloaded inputs, matrices and scratch analysis remain in
`/tmp/knee-label-audit-20260927/`. The summary records SHA-256 hashes for `matrices.npz`,
`deep_analysis.py`, `deep_validation.py` and `shared_reliability.py`; the latter three implement
the broad screen, bootstrap/held-pair diagnostics and reliability fitting. They are scratch
analysis, not maintained repository commands. The retained JSON/CSV evidence does not include
all raw inputs or scripts, so complete replay requires preserving the original scratch workspace.
Local evidence is not version-controlled; the tables and qualifications in this document are
the durable record. Recheck Kaggle revisions before implementation.
