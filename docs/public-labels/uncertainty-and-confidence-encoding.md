# What uncertainty and confidence actually encode

## Steven: original, repaired, and blended values

The saved dataset description documents v1 `0.5` as unaddressed report evidence.
It occurs in **13,438 / 52,884 cells (25.4%)**. There are no separate verdict,
confidence, evidence-span or raw-response fields in these files.

| Target | Steven v1 unaddressed / 4,407 | Pilkwang UNK / 4,406 |
|---|---:|---:|
| ACL | 366 | 355 |
| MCL | 444 | 430 |
| Medial Meniscus | 244 | 250 |
| Lateral Meniscus | 440 | 442 |
| Medial OA | 896 | 1,125 |
| Lateral OA | 1,090 | 1,457 |
| PF OA | 763 | 815 |
| Effusion | 432 | 434 |
| Synovitis | 3,689 | 3,712 |
| Baker's | 2,126 | 2,023 |
| Contusion | 1,058 | 916 |
| Fracture | 1,890 | 2,486 |

The audit reconstructs v2 within floating-point precision:

```text
if v1.Synovitis == 0.5:
    v2.Synovitis = 0.25 + 0.5 * v1.Effusion
else:
    v2.Synovitis = v1.Synovitis
```

Every other target is unchanged. Of 3,689 originally unaddressed Synovitis
cells, 3,271 change and 418 remain `0.5`. The changed values are inferred from
Effusion; they are not new explicit Synovitis statements.

v4 is **exactly** `0.5 * Steven_v2 + 0.5 * lixin73`, aligned by study ID, without
using lixin73's confidence columns. It has only **32 exact `0.5` cells**.
Of v1's unaddressed cells, **13,433 no longer equal `0.5`** in v4; **10,085 become
`0.25`**. Conversely, 27 v4 cells equal `0.5` even though v1 did not. Therefore
neither `v4 == 0.5` nor distance from `0.5` recovers original report coverage.

The author's proposed `2 * abs(p - 0.5)` loss weight is a heuristic, not validated
calibration. Applied to v4, it assigns weight 0.5 to the 10,085 cells above.
Preserve v1's evidence indicator and the v2 repair indicator separately if
using v4. Excluding originally unaddressed non-gold training targets was
[tried and rejected](../history/sharing/2026-09-13-label-masking-result.md): full-fold AUC
fell by 0.011455. The baseline again supervises all nonmissing soft targets.

## Report-state audit and implications for supervision

**Measured 2026-09-20; the proposed training change is untested.** Reporting status
is available as auxiliary supervision, but the evidence does not establish that
replacing disease targets with three hard states improves image-model AUC. The
[backlog proposal](../backlog/small-changes.md#auxiliary-report-state-prediction)
retains balanced v4 BCE and adds a separate reporting head.

The [audit artifact](../../outputs/analysis/report-state-audit-20260920/audit.json)
records input SHA-256 hashes, methods and all twelve labels' results. It joins
Steven v1/v2/v4 and official labels by `StudyInstanceUID` across 4,407 studies.
The diagnostic groups are `v1 < 0.5`, `v1 == 0.5` and `v1 > 0.5`: negative-score,
unreported and positive-score groups. These are extractor-derived states, not
verified report annotations. No models were trained or inference rerun.

### Disease despite report silence

Among the 58 gold studies, 41 have unreported Synovitis in v1; **14 of those 41
are gold-positive**, accounting for 14 of all 27 gold-positive Synovitis studies.
Across the complete dataset, Synovitis has only 150 negative-score and 568
positive-score cells against 3,689 unreported cells. This makes reporting status
a particularly different task from disease prediction for this label. The
[official target](../metadata/official-label-criteria.md#labels-versus-report-text-discussion-733826-host-reply-2026-08-08)
is image-adjudicated presence or absence, including findings omitted from reports.

A three-state head's positive output estimates the positive report-state
probability. Renormalizing it as `P(positive) / (P(positive) + P(negative))`
conditions on the finding being addressed; it does not establish disease
probability when reporting depends on disease or other study characteristics.
The small, selected gold set demonstrates disagreement, not population prevalence.

### Information lost by collapsing the soft scores

For this diagnostic only, score the three v1 groups as `0`, `0.5` and `1` and
compute unweighted mean per-label ROC-AUC against the same 58 official gold rows:

| Label-table scores | Gold macro AUC |
|---|---:|
| Original v1 soft scores | 0.877991 |
| v1 collapsed to three ordered states | 0.809238 |
| v2 soft scores | 0.887349 |
| v4 soft scores | 0.892707 |

The collapse removes useful within-group ordering from the label table. It is
**neither an upper bound nor a forecast for a trained three-class image model**,
whose predicted probabilities could rank images within those groups. These gold
rows have already informed label development; the table is descriptive and cannot
select a new model or establish an independent gain.

### What the existing image scores already encode

The saved `balanced-bce` predictions, seed 42, fold 0 of `all_5fold_v3`, selected
epoch 7 (zero-based), were restricted to 893 non-gold validation studies. Across
all twelve labels, mean predicted disease score is higher in the v1-unreported
group than in the negative-score group. For Synovitis the means are **0.406 versus
0.265**; for Medial Meniscus, **0.360 versus 0.210**. Using the disease score to
distinguish these two report-state groups gives AUCs of **0.622–0.723** across
labels. These are reporting-status diagnostics, not disease AUCs or evidence of
improved calibration.

The audit pins the prediction bytes; the run's
[configuration](../../outputs/training/balanced-bce/effective_config.json),
[provenance](../../outputs/training/balanced-bce/provenance.json) and
[source snapshot](../../outputs/training/balanced-bce/source_snapshot.zip)
retain its settings and implementation. Its base commit is
`f4c340bab4518aa743acb7a6e4649e9d3719b9b3` with recorded working-tree changes.

### Implications and untested alternatives

The hypothesis is that an auxiliary reporting head could represent coverage
separately while preserving the existing disease signal, including the 10,085
unreported cells mapped to v4 `0.25`. The saved scores show an association; they
do not prove that separating it improves disease ranking or that the model uses
reporting-style shortcuts.

The [masking experiment](../history/sharing/2026-09-13-label-masking-result.md)
reduced full-fold AUC from 0.890976 to 0.879520, with eleven labels declining and
Synovitis losing 0.022340. That cautions against removing disease supervision from
unreported cells. It does not reject three-state training, which would still
train the representation on those cells.

A pure three-state replacement remains an untested alternative, not a rejected
experiment. It would need a defined mapping back to the twelve disease scores,
explicit treatment of gold labels and soft polarity, and comparison on the frozen
disease endpoint. Reporting states also do not repair the
[known OA severity conflicts](../history/sharing/oa-and-meniscus-disagreement-audit.md#source-lineage-and-prediction-effects).

## Pilkwang: graded scores with explicit verdicts

The downloaded table contains exactly these observed combinations:

| Verdict | Score | Confidence | Cells |
|---|---:|---:|---:|
| NO | 0.08 | 0.85 | 22,298 |
| UNK | 0.28 | 0.05 | 14,445 |
| YES | 0.68 | 0.95 | 6,461 |
| YES | 0.82 | 0.95 | 6,427 |
| YES | 0.94 | 0.95 | 3,241 |

There are no `0.5` cells; that does not mean complete report coverage.
**3,712 / 4,406 Synovitis cells (84.2%)** are UNK. The producer prompt includes
both unmentioned and explicitly uncertain findings in UNK, so its definition
is broader than Steven's documented unaddressed flag. The dataset describes
the values as ranking scores rather than calibrated probabilities.

`api_labeler.py` lines 56–99 provide the structured verdict/severity prompt and
output schema. Lines 143–195 configure the default model string
`claude-opus-5`, low effort, and a 12,000-character report limit. This identifies
the distributed source defaults, not verified settings for every released row.
No gold-label access or replacement occurs in that producer file.

The producer imports `FINDING_SPEC` and `to_scores` from an **unprovided
`llm_labeler.py`**. Its precise finding definitions and extraction-to-score
implementation therefore cannot be fully reproduced from this snapshot.
The visible prompt grades minimal, moderate/unqualified and severe findings;
it does not establish compliance with all of the host's severity thresholds.
The observed score/confidence mapping above is verified from the CSV.

The [inspected baseline notebook](../research/public-research.md#public-code-worth-inspecting)
separately falls back to lexicon labels for uncovered IDs and replaces eligible
training gold targets with official labels. Those are downstream notebook
operations, not evidence of replacement inside the downloaded CSV.

## lixin73: mostly binary targets

**50,859 / 52,884 cells (96.2%)** are exactly zero or one; 168 equal `0.5`.
The two files have identical targets. Confidence values range from 0.35 to 1,
with per-target means approximately 0.976–0.992, but no verdict fields accompany
them. The retrieved dataset description is empty and the snapshot supplies
no extractor/prompt. The model identity in the filename and upstream credit
is author-provided provenance, not independently verified execution evidence.
Do not equate its zero targets with explicit report denials or its confidence
values with demonstrated correctness.
