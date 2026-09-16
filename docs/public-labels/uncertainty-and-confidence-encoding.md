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
