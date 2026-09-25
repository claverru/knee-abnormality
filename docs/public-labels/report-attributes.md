# Report attributes in public label tables (2026-09-22)

**Severity supervision would mostly repeat what the training targets already teach. The
attributes instead expose overlabelled low-grade positives.** Two public tables already hold
per-label severity, and three hold report state, so no LLM run is needed. Where the two severity
sources agree, the soft training targets already encode severity, and the incumbent's predictions
already rise with it. The model ranks two kinds of cell worst:

- **Low-grade report positives:** on gold, only about half are positive.
- **Negatives the report never addressed.**

Both are label noise, not missing image features. On the eight Steven v4 targets, low-grade
positives train at 0.88 against a gold-positive rate of 0.54. But `combined_v1` already orders the
report calls as gold does, so the mismatch is in spacing, not order. The first repair, the
consensus mean, pushed low-grade cells down to the not-addressed level and lowered the table's gold
agreement ([below](#consensus-mean-rule-tested-on-gold)), so it was not trained on.

- **Script:** [`check_report_attributes`](../../scripts/adhoc/check_report_attributes.py), CPU only,
  under a minute, no LLM calls; refuses an existing output directory:
  `rtk proxy uv run --locked python -m scripts.adhoc.check_report_attributes --output-dir outputs/analysis/report-attributes-new`.
- **Artifacts:** `outputs/analysis/report-attributes-20260922/`, holding the tables, input hashes and
  a source snapshot.
- **Scope:** training rows are folds 1–4 of `all_5fold_scanner_v1` (3,502 studies), with targets
  from [`combined_v1`](combined-labels.md). Grounding uses the `k_05` fold-0 predictions at ×4
  (905 studies), against the fold's evaluation targets thresholded at 0.5. The gold rows are the
  58 gold studies.

## Sources

| Table | Attributes | Coverage |
|---|---|---|
| Ramit GPT-luna (`ramitnandi/rsna-knee-labels-gpt-luna`) | Per study and label: positive, negative or not addressed; LOW/MEDIUM/HIGH severity on every positive; hedged; evidence phrase | All 4,407 studies |
| Zaid (`zaidaliiq1000/rsna-knee-llm-labels`) | Severity 0–3 averaged over its models, where −1 means not mentioned; mention flag | 3,609 studies |
| Pilkwang v2 | YES/NO/UNK verdict per label | All studies |

Ramit is also one of the nine consensus sources behind the combined labels.

## Cross-source agreement

Severity is compared on positives that both sources rated, using quadratic kappa with Zaid rounded
to three levels. Zaid values of 0 or below count as unrated, and Zaid rows with no contributing
models are dropped.

| Kappa | Labels |
|---|---|
| 0.64–0.66 | ACL, Effusion, Baker's |
| 0.45–0.47 | Medial, Lateral and PF OA |
| 0.25–0.36 | Fracture, Synovitis, Contusion, MCL, Lateral Meniscus |
| 0.14 | Medial Meniscus, where 88% of Ramit positives are HIGH |

Report state: Ramit and Pilkwang agree on whether a finding is addressed (kappa 0.81–0.99),
except for Contusion (0.33) and Fracture (0.56). Zaid's mention flag disagrees with both (kappa
0.15–0.64). Where both sources address a finding, its polarity agrees in 64–100% of cells.

## Redundancy with the training targets

Among training positives, Ramit severity correlates with the soft target: Spearman 0.79 for
Medial OA, Effusion and Baker's, 0.76 for Lateral OA, and 0.66–0.68 for ACL and MCL. The four
consensus targets already demote low grades: ACL, MCL and PF OA LOW positives train at 0.34–0.40
and Fracture at 0.59. An auxiliary severity head would therefore restate what the soft BCE
already teaches on the labels where severity is reliable.

## Agreement with the incumbent's predictions

Among fold-0 positives, `k_05`'s predictions rise with Ramit severity (Spearman, bootstrap 95%):

| Spearman | Labels |
|---|---|
| 0.49–0.58 (lower bounds 0.35–0.52) | Effusion, Baker's, Medial OA, ACL |
| 0.32–0.42 | Lateral Meniscus, Lateral OA |
| 0.09–0.24 | Medial Meniscus, Contusion, Synovitis, PF OA, MCL, Fracture |

Ranking losses concentrate in two places:

| Where | Result |
|---|---|
| LOW positives against all negatives | AUC 0.80 Effusion (299 positives), 0.83 Medial OA, 0.85 Lateral OA, 0.87 Baker's, 0.73 Lateral Meniscus (23), 0.81 Medial Meniscus (26), against 0.94–0.97 for HIGH |
| Negatives the report never addressed | Scored higher than explicit negatives (Effusion 0.46 against 0.31, Medial OA 0.39 against 0.26); positives separate from them at AUC 0.78–0.91, against 0.89–0.95 from explicit negatives |

## Gold rate by report call

Cells on the 58 gold studies, gold-positive rate against the mean training target those cells
would carry (`combined_v1` before the gold override):

| Call | Eight Steven v4 targets | Four consensus targets |
|---|---|---|
| HIGH | 84/104 (0.81), target 0.95 | 47/51 (0.92), target 0.87 |
| MEDIUM | 21/27 (0.78), target 0.96 | 10/11 (0.91), target 0.78 |
| LOW | **32/59 (0.54), target 0.88** | 4/19 (0.21), target 0.44 |
| Not addressed | 22/156 (0.14), target 0.31 | 7/56 (0.13), target 0.16 |
| Negative | 9/118 (0.08), target 0.08 | 4/95 (0.04), target 0.09 |

Across all non-gold studies, the eight Steven v4 targets hold 3,402 LOW cells at a mean target of
0.88 (0.85–0.91 per label), and 97–100% of them sit at or above 0.5. Effusion supplies 25 of the 59
gold LOW cells (12 positive). The other labels hold 3–7 cells each, so the per-label gold rate is
unmeasured. The corpus-wide 0.88 is the finding; gold only confirms it. It extends the
[OA disagreement audit](../history/sharing/oa-and-meniscus-disagreement-audit.md), whose
below-threshold OA findings carried targets of 0.80–0.93, to every Steven v4 target.

The order of the mean targets across calls matches the order of the gold rates for both groups of
targets. For AUC, only that order matters within a label.

## Consensus-mean rule tested on gold

Tested 2026-09-23 and **not trained on.** The rule built `combined_v2`: `combined_v1`, except every
cell Ramit calls a LOW-grade positive takes the nine-extractor consensus mean, whatever its label.
The four consensus targets already hold that mean and do not change. Every one of the 4,158 cells
had at least five sources.

- **Script:** [`prepare_grade_aware_labels`](../../scripts/adhoc/prepare_grade_aware_labels.py).
- **Output:** `data/labels/prepared/combined-v2/` (labels SHA-256 `e8dec30f…`), with
  `preparation.json`. The manifest is
  `configs/labels/combined-v2-label-sources.json`. CPU only, seconds, deterministic:
  `rtk proxy uv run --locked python -m scripts.adhoc.prepare_grade_aware_labels`, with fresh
  `--output-dir` and `--output-manifest` paths on a rerun. It fails if any consensus target
  changes.

On non-gold studies the changed cells move a long way:

| Label | Cells | Mean target | Positive rate (≥ 0.5) |
|---|---:|---|---|
| Effusion | 1,485 | 0.88 → 0.31 | 0.59 → 0.27 |
| Baker's | 525 | 0.89 → 0.44 | 0.25 → 0.15 |
| Medial OA | 484 | 0.87 → 0.38 | 0.37 → 0.28 |
| Lateral OA | 429 | 0.86 → 0.36 | 0.27 → 0.18 |
| Menisci, Synovitis, Contusion | 87–146 each | 0.85–0.91 → 0.65–0.69 | within 0.002 |

The positive-rate shifts would also change the class-balance weights.

As a table, `combined_v2` agrees worse with gold than `combined_v1`:

| | Gold macro AUC | Effusion (25 changed cells) | Baker's | Contusion | Medial / Lateral Meniscus | Lateral OA | Synovitis |
|---|---:|---:|---:|---:|---:|---:|---:|
| Change | 0.9075 → 0.9044 | −0.020 | −0.008 | −0.007 | −0.003 / −0.003 | −0.002 | +0.005 |

The consensus mean puts low-grade Effusion cells at 0.31, level with not-addressed cells
(0.27–0.31). Gold puts low grades at 0.54 and not-addressed cells at 0.14, so the rule breaks an
order that `combined_v1` already has right.

Any repair would have to keep low grades between not-addressed cells and the higher grades.
Choosing where to put them means fitting the gold rates, so no variant is proposed.

## Implications

- **Auxiliary severity heads:** not worth a campaign (moderate confidence). The severity that
  sources agree on is already in the targets and the predictions.
- **Report-state heads:** the hard negatives are separated by whether the report addressed the
  finding. Learning that attribute teaches reporting behaviour, which fold-0 validation rewards
  and the leaderboard need not
  ([class-balance exponent](../validation/results/2026-09-21-class-balance-exponent.md)).
- **Low-grade targets:** the consensus-mean repair failed its gold check and was not trained on.
  Any label change here would be judged on the leaderboard, not by fold 0 or autoresearch.
- **Where label error remains:** within report-call classes and in omissions, which only images
  can resolve; see [image-blended labels](../validation/results/2026-09-23-image-blend-labels-result.md).
