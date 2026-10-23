# Gold versus silver labels

**Status, 2026-09-27:** the training labels describe what the reports say, the hidden-test labels
what the images show. This document collects the measured gaps, how they affect training and
validation, and what to try. The site-bias part has its own document, [site bias](../modeling/site-bias.md).

## Where the labels come from

| Labels | How they were made | Studies |
|---|---|---|
| **Gold** (official labels in `data/raw/train.csv`) | From the images under standardized criteria; the host confirmed they were "assigned independently from the MRI images, rather than extracted from the reports" ([criteria](../metadata/official-label-criteria.md#labels-versus-report-text-discussion-733826-host-reply-2026-08-08)) | 58 |
| **Silver** (`combined_v1`, our training targets for everything else) | The competition gives reports only; the targets are public LLM extractions of those reports ([combined labels](combined-labels.md)) | 4,349 |
| **Hidden test** | By the host's description, the same process as gold: from the images | unknown |

The 58 gold studies are the only training data labeled the way the test is labeled. The host
states they are "intended to help participants surface this conclusion": that image-derived labels
are authoritative where they disagree with report text. The host also states there is no
missingness in the target: every finding was adjudicated for every reference study, so a report's
silence is an unobserved label, not an absent one.

## Measured gaps

From the [v3 answer analysis](jevk5-report-attribute-extraction.md)
(58 gold studies, 2026-09-25):

| Report says | Gold positive | Silver targets positive | Official text |
|---|---|---|---|
| Small effusion | 15 of 28 (54%) | 95% | negative |
| Small Baker's cyst | 3 of 9 (33%) | 61% | negative |
| Low-grade or softened medial cartilage | 1 of 5 | 67–75% | negative |
| Low-grade or softened PF cartilage | 3 of 11 | — | negative |

- **Low grades are over-called** by the silver targets, but gold is not simply the official cut:
  small effusions are gold-positive about half the time.
- **Silence becomes a negative label** in silver, while gold adjudicated every finding.
- **Site conventions:** one minor-vendor site's labels run ×1.1–×3.6 higher on ten of twelve
  targets from reporting completeness ([vendor associations](../metadata/vendor-label-associations.md)).

Scoring the same v3 answers against gold, a rule at the official threshold against "any finding":

| Label | Official-threshold rule | Any finding |
|---|---|---|
| PF OA | **0.947** | 0.842 |
| Effusion | **0.755** | 0.614 |
| Baker's | 0.832 | **0.871** |
| Medial OA | 0.848 | **0.913** |
| Lateral OA | 0.832 | **0.841** |

A hard official cut is too strict for some labels and right for others; samples are small
throughout.

## How it affects us

1. **The training targets are shifted, not just noisy.** The model learns report habits (low-grade
   positives, silence as negative, site conventions) that the test labels do not contain.
2. **Local validation partly measures the wrong thing.** Folds are scored on silver labels except
   for the gold studies, so the local metric rewards matching report habits. Local and public have
   disagreed in sign repeatedly: on fold 0, S58 (a local reject) scored 0.944 and S59 (a local
   adoption) 0.939; on fold 3, the scanner confusion head lost 0.0008 locally and gained 0.001
   publicly (S67), while the scanner-model target lost 0.0013 locally and 0.004 publicly (S70).
3. **AUC is a ranking metric,** so borderline cases matter most, and the borderline cases are the
   low grades the reports over-call.

## Already addressed

- Gold labels override silver on the 58 studies, in training and evaluation.
- `combined_v1` sources were chosen by gold AUC (0.9075 macro against 0.8927 for Steven v4; gold
  helped choose, so the figure is optimistic). `rebuild_state_v2` and then `rebuild_state_v3`
  (report-reading rules aimed at the same over-calls) replaced it ([label tables](label-tables.md)).
- Report-attribute heads add structured report supervision
  ([heads](../modeling/report-attribute-heads.md)).
- Site debiasing ([site bias](../modeling/site-bias.md)): the light scanner confusion head is adopted
  (0.948 against 0.947); stronger site removal lost publicly (S70 0.944, S71 0.943), in the
  order of each variant's disease cost on the major vendors.

## Ideas considered

1. **Target repair toward the image criteria** (evidence-backed). The v3 answers of the
   [extraction work](jevk5-report-attribute-extraction.md) encode severity at the official
   thresholds; the proposal was to blend each study's silver target toward "the finding is at or
   above the official threshold" with one label-agnostic rule instead of a hard cut, since the gold
   check favors the threshold for PF OA and Effusion and "any finding" for Baker's and Medial OA. The
   OA-only repair plan it relates to was superseded by the [label rebuild](label-rebuild/INDEX.md),
   and the report-reading rules of `rebuild_state_v3` addressed the same mismatch with per-label
   rules.
2. **Decide with pooled gold, not fold gold.** Fold gold is about 11 studies, which is noise;
   out-of-fold predictions across all five folds give the full 58. It was reported as a standing
   diagnostic beside the fold metric, not as an adoption rule.
3. **Read local losses on over-called labels as a possible good sign.** A candidate that loses
   locally by lowering low-grade positives may be moving toward the test labels; check with a
   paired public submission.

## Tried and not worth repeating

- Correcting labels with the model's own predictions (noisy-labels campaign, nothing adopted).
- Upweighting the gold studies in the loss (rejected by the user).
- Stronger site removal: finer site identity and harder confusion pressure lost publicly
  ([site bias](../modeling/site-bias.md)).
- Post-hoc centering of predictions per scanner: lost on gold (−0.0096 pooled).
