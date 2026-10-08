# Report-attribute auxiliary heads

**Part of the recipe since 2026-09-25** (user decision): on target set 3.1 the heads scored
+0.00105 ×4 macro and 0.940 public against 0.938
([result](../validation/results/2026-09-25-report-attribute-heads.md)). Training only: one
auxiliary prediction per report-attribute question beside the unchanged disease loss. The
attributes come from the [JevK5 extraction](../ongoing/report-attribute-extraction.md). On the
ResNet-18 recipe, removing them cost −0.0006 ×4 (80% [−0.0011, −0.0001]) and halving the
coefficient to 0.05 cost −0.0004, so they stay
([revisits campaign](../validation/results/2026-10-01-revisits-campaign.md#other-runs)).

## Configuration

Both fields are required in every experiment config:

```yaml
paths:
  report_attribute_manifest: labels/jevk5-report-attribute-targets-v3.1.json
trainer:
  report_attribute_loss_weight: 0.1
```

The manifest path resolves relative to the config file, like `paths.label_manifest`.
A weight of 0 switches the loss and its validation fit off; the heads stay in the model untrained.

## Targets

[`prepare_report_attribute_targets`](../../scripts/tools/prepare_report_attribute_targets.py)
turns a finished extraction run into
`data/labels/prepared/<name>/targets.parquet`, one row per training study, one column per
question option and one weight per question, and a manifest in `configs/labels/` pinning the
targets, the question set and the source run by SHA-256. The current set is 3.1:
`configs/labels/jevk5-report-attribute-targets-v3.1.json`, 35 questions and 181 options.

- **Values are JevK5's option probabilities,** averaged over both option orders, not its top
  answer: ambiguous reports give spread targets.
- **Dropped questions:** knee side, because reflection augmentation mirrors the knee and a
  left/right target would contradict the image (medial and lateral questions stay; they are
  anatomical, like the meniscus labels); the three osteophyte questions, mostly position noise
  (3,778 reports with no osteophyte word carried 0.32 mean osteophyte probability); and Hoffa,
  whose answers were mostly other edema and whose head did not learn in the first run.
- **Merged options** (summed in both orders): overall OA "not mentioned" and "none"; cartilage
  depth "not mentioned", "normal" and "only elsewhere"; cartilage extent "no loss" and "only
  elsewhere"; bone-on-bone into full thickness. The merged pairs mean the same for the disease and
  flipped between option orders on 600–1,369 reports each
  ([analysis](../ongoing/report-attribute-extraction.md#v3-full-run-analysis-2026-09-25)).
- **Reliability weight per question and study:** one minus the total-variation distance between
  the forward-order and reverse-order distributions after merging. Mean 0.86; 4% of cells fall
  below 0.5. Lowest by question: cartilage extent (0.73–0.78), Baker's size (0.76), cartilage
  depth (0.79–0.80).
- **Child questions are conditioned on their parent.** With `p` the parent's probability of a
  finding, the child becomes `1 - p` on its "none" option plus `p` times its own distribution,
  whose "none" mass moves to its "not stated" option, and its weight becomes the parent's weight
  where the parent says no and its own where the parent says yes. Gates: meniscus tear location
  and type on meniscus status; MCL timing on MCL status; contusion location on contusion status;
  Baker's rupture on Baker's size; cartilage extent and PF surface on cartilage depth. Ungated
  contusion location placed a contusion on 190 reports whose Contusion target was near zero, and
  ungated PF surface reported damage on 1,267 reports whose PF depth said none.

## Heads and loss

`ReportAttributeHeads` (`training/report_attributes.py`) lives in the Lightning module, not in
`KneeAbnormalityModel`, so prediction, export and checkpoint loading are unchanged and the hashed
preprocessing identity is untouched.

- **Pooling:** each question has its own attention query (256→128→questions) over the study's
  visually valid normalized tokens, the tokens the disease head pools, without the anatomical
  bias or meniscus side mask.
- **Output:** one logit per option from the question's pooled feature; softmax within each
  question.
- **Loss:** soft cross-entropy against the target distribution, a reliability-weighted mean over
  questions and training studies with targets (divided by the sum of weights, so the scale does
  not depend on the weights), times `trainer.report_attribute_loss_weight`, added to the
  class-balanced BCE and the half-knee loss. Validation and the adoption metric are the twelve
  disease labels only.
- **Optimizer:** head learning rate and the usual decay rule. Weight averaging copies only the
  disease model.
- **Initialization** runs under a forked RNG, so the disease model's random streams match a run
  without the option.

## Question weighting

Target set 3.1 gives 35 heads. Apart from the reliability weights, the attribute loss has no
per-question, per-label or class weighting, so the share per area follows the question count:
cartilage/OA 8 heads, menisci 7, other findings 9, fluid and lining 4, ligaments 3, bone 3, study
context 1. Cartilage/OA and the menisci together carry about 43% of the attribute signal. (The
first run's v3 targets had 39 heads, cartilage/OA 11.)

Untested alternative (noted 2026-09-25): average within each finding area first, then across
areas, so each area weighs the same. It is still label-agnostic; keep it in mind if the first
run trains the term but helps only the heavily represented areas.

## Logged diagnostics

`report_attribute_loss` (epoch mean cross-entropy), `report_attribute_studies` (training studies
with targets) and `report_attribute_kl_<question>`, the per-question divergence from the target
distribution, which is zero at a perfect fit. They show whether the term trains; attribute
accuracy is diagnostic, not the adoption metric.

## Identity and resume

When enabled, the input identity records the manifest SHA-256, and the checkpoint
hyperparameters record the manifest, target and question-set hashes and the question list. A
resume with different targets is refused.

## Coefficient

0.1, fixed before launch. On target set 3.1 the attribute cross-entropy starts near the mean log
option count, 1.57, and cannot fall below the targets' weighted mean entropy, 0.55, which carries
no gradient. The learnable part is therefore about 1.0 at the start, 0.1 after weighting, against
a disease loss of about 0.47: roughly a fifth at the start, shrinking as the heads fit. The
half-knee loss uses 0.075 on a similar scale. The first run, on v3 targets, fell from 1.14 in its
first epoch to 0.95 with a floor near 0.63.

- **Healthy run:** `report_attribute_loss` falls from about 1.55 toward 0.7.
- **Too weak:** the attribute loss falls but the disease curves track the control step for step.
- **Too strong:** disease validation AUC trails the control from the first epochs.
