# 2026-09-27 auxiliary-signals campaign

The [auxiliary-signals campaign](../../../../autoresearch/campaigns/auxiliary-signals/INDEX.md)
searched for training-time auxiliary signals built from existing data: the JevK5 v3 report
attributes and the DICOM headers. It ran 21 experiments; the user adopted the **scanner confusion
head** on public evidence, 0.948 (S67) against 0.947 for its paired control (S68), one rounding
step. No attribute-target change and no heavier site-debiasing variant beat its reference; the only
local near miss, the question-level reliability filter, lost publicly (0.943, S72). The site line is
analyzed in [site bias](../../../modeling/site-bias.md), the report-versus-image label gap in
[gold versus silver labels](../../../public-labels/gold-versus-silver.md).

## Setup

- **Campaign:** 6 user-ordered seeds, then forced ideas added on the fly through the campaign
  inbox (first use of that runtime), then researcher refills, from 2026-09-26 to 2026-09-27, on
  Codex. Stopped by the user after the last selected run.
- **Reference:** `half-stack-control`, seed 2617320043, fold 3, ×4 macro 0.905205, public 0.947
  (S68). After the user adopted the scanner confusion head mid-campaign (decision
  `user-adopt-20260926T191359Z-da4f7994`), later candidates were scored against it (×4 0.904442).
- **Rule:** the [shared adoption rule](../../../../autoresearch/protocol.md#shared-evaluation-and-adoption).
  Soft AUC is a diagnostic.

## Results

Δ is ×4 macro AUC against the reference at the time.

| # | Idea | Reference | Δ | 80% interval | Verdict | Public |
|---|---|---|---:|---|---|---|
| 1 | Attribute loss weight 0.2 | control | −0.00084 | [−0.0025, +0.0008] | reject | |
| 2 | Attribute loss averaged per question group | control | −0.00062 | [−0.0022, +0.0009] | reject | |
| 3 | Drop questions with mean order agreement < 0.80 | control | +0.00104 | [−0.0004, +0.0024] | reject | 0.943 (S72) |
| 4 | Drop report-completeness options | control | −0.00049 | [−0.0020, +0.0011] | reject | |
| 5 | Drop hedged options | control | −0.00149 | [−0.0031, +0.0001] | reject | |
| 6 | Scanner gradient reversal | control | −0.00604 | [−0.0086, −0.0035] | reject | |
| 7 | Scanner gradient reversal, head at full strength | control | −0.00419 | [−0.0065, −0.0020] | reject | |
| 8 | Cell-level reliability filter | control | −0.00062 | [−0.0020, +0.0007] | reject | |
| 9 | Grade merges at severity bands | control | −0.00141 | [−0.0027, −0.0001] | reject | |
| 10 | **Scanner confusion head** (detached head, bounded KL) | control | −0.00076 | [−0.0027, +0.0011] | **adopted by the user** | **0.948 (S67)** |
| 11 | Conditional child targets | control | −0.00119 | [−0.0027, +0.0003] | reject | |
| 12 | Joint meniscus status-location targets | control | −0.00064 | [−0.0022, +0.0009] | reject | |
| 13 | Stronger confusion head | confusion | −0.00330 | [−0.0053, −0.0014] | reject | 0.943 (S71) |
| 14 | Confusion on backbone features | confusion | — | — | invalid (identity gate); hand −0.0016 | |
| 15 | Per-site logit bias (8 × 12) | confusion | +0.00040 | [−0.0011, +0.0019] | reject | 0.947 (S69) |
| 16 | Confusion over 31 scanner models | confusion | −0.00134 | [−0.0035, +0.0007] | reject | 0.944 (S70) |
| 17 | Consensus target hardening | confusion | −0.00024 | [−0.0021, +0.0016] | reject | |
| 18 | Shared pooling per question family | confusion | −0.00236 | [−0.0041, −0.0005] | reject | |
| 19 | Predicted attributes in a residual readout | confusion | — | — | invalid (identity gate); hand +0.0005 | |
| 20 | Per-model logit bias (31 × 12) | confusion | −0.00279 | [−0.0046, −0.0010] | reject | |
| 21 | Per-site bias + 31-model confusion | confusion | −0.00030 | [−0.0024, +0.0017] | reject | |

Hand comparisons (rows 14 and 19) are in
`outputs/analysis/s/scanner-confusion-backbone-features-vs-incumbent/` and
`outputs/analysis/p/predicted-attribute-residual-readout-vs-incumbent/`. Three ideas were blocked at
review without a run: the completeness bias on report length (outside permitted data), its v3
rebuild (logit offset on the fixed disease loss), and per-scanner token harmonization (needs
prediction-loader inputs).

## Decisions outside the rule

- **Scanner confusion head adopted on public evidence** (user decision, 2026-09-26): +0.001 public
  against −0.00076 locally, one rounding step. It entered the main recipe on 2026-09-27 as fixed
  behavior (`trainer.scanner_confusion_weight: 0.1`); its control, `scanner-confusion-control`
  (seed 703791404, fold 4), scored ×4 0.909733 ([design](../../../modeling/scanner-confusion-head.md)).
- **Reliability filter reevaluated publicly** (user request): 0.943 against 0.947; not adopted.

## Findings

- **Gradient reversal damages, bounded confusion does not.** Both reversal heads stayed above the
  frequency entropy: reversal maximizes an unbounded cross-entropy and bent features to fool the
  head. The detached head with a KL toward the group frequencies tracked the control.
- **Site removal shows as a prediction shift, not a per-label AUC change.** Every confusion variant
  lowered the minor site's predictions in proportion to its label inflation (Synovitis most,
  Effusion the other way), but the site is 50 of 872 validation studies.
- **Public order follows the disease cost on the major vendors:** S67 0.948 > S68, S69 0.947 >
  S70 0.944 > S71 0.943, matching major-vendor-only Δ +0.0002, +0.0005 > −0.0013 > −0.0030.
- **Attribute-target changes do not help.** Ten merge, removal, gating, pooling and selection
  variants ranged −0.0024 to +0.0010 locally; the one positive lost publicly.
- **Local and public disagree in sign** (S72: +0.0010 local, −0.004 public): local labels are
  report-derived, test labels image-derived ([gold versus silver labels](../../../public-labels/gold-versus-silver.md)).
- **Configuration defect:** the sealed policy allowed no `preprocessing_identity` differences, so
  every idea editing `modeling/model.py` ended invalid after a full run (rows 14 and 19).
