# Public report-label audit

The [September 13 comparison](#2026-09-13-new-public-label-comparison) found
Tonylin's soft labels the strongest candidate by reference agreement, but the
[completed training comparison](validation.md#2026-09-13-tonylin-label-comparison)
scored **0.921 publicly versus Steven v4's 0.929**. Retain Steven v4; the
higher label-table AUC did not translate into a better submitted model.

Audited on **2026-09-10** using the downloaded tables and producer material.
Steven Lee's v2 and v4 tables are structurally usable starting candidates: both
cover all 4,407 studies with finite targets in `[0, 1]`. Pilkwang's table covers
all **4,349 report-only studies**, but omits one reference study. Following the
audit, **Steven v4 was selected for the first baseline**; see the
[recorded decision](#implication-for-the-first-baseline). No label values were changed.

The main integration constraint is **source-specific uncertainty**. Steven v1's
`0.5` and Pilkwang's `UNK` retain missing report evidence; Steven v4's blended
probabilities do not retain that distinction. Keep the component tables.
Extraction agreement on the reference set does not establish downstream
image-model quality or independent validation.

The [2026-09-14 blinded disagreement audit](sharing-disagreement-audit.md#source-lineage-and-prediction-effects)
adds a separate severity concern: 19 reviewed report-derived OA positives
describe explicitly low-grade findings under the host's high-grade endpoint.
Steven v1/v2 and lixin agree positively on all 19; v4 values are 0.80–0.93.
Confidence or source-agreement filtering would retain them. This is a selected
report-text audit, not an estimate of population label errors or image
adjudication. One additional conflict involves official gold, which is preserved.
No labels changed; the existing Steven-v4 training decision remains in place
pending a controlled OA-only supervision comparison.

## Acquisition and reproduction

The [source manifest](public-label-sources.json) records dataset IDs, explicit
Kaggle dataset versions, download commands, file sizes and SHA-256 hashes,
metadata snapshots, and the official-data/fold fingerprints. **Dataset version
6 contains label recipes v1, v2, and v4**; these version numbers are different
concepts. All three dataset snapshots declare CC0. Labels and downloaded code
are local under the Git-ignored `data/labels/public/` directory.

| Source | Pinned Kaggle version | Local directory | Audited files |
|---|---:|---|---|
| [Steven Lee](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels/versions/6) | 6 | `data/labels/public/stevenleehans/version-6/` | `llm_labels_full.csv`, `llm_labels_v2.csv`, `llm_labels_v4_blend.csv` |
| [Pilkwang](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels/versions/1) | 1 | `data/labels/public/pilkwang/version-1/` | `report_labels_v2.csv`, producer source `api_labeler.py` |
| [lixin73](https://www.kaggle.com/datasets/lixin73/rsna-knee-llm-report-labels-sol56/versions/1) | 1 | `data/labels/public/lixin73/version-1/` | `labels_llm_gpt56sol.csv`, `report_labels_gpt56sol.csv` |

lixin73 was included to trace the upstream component of Steven v4, rather than
expand the audit into a general search for labels. Downloaded Python was read,
not executed. The installed Kaggle CLI version was 2.0.2.

Run the aggregate audit from the repository root:

```bash
rtk proxy uv run --locked python -m scripts.adhoc.audit_public_labels
```

[`scripts/adhoc/audit_public_labels.py`](../scripts/adhoc/audit_public_labels.py) verifies
pinned file hashes before aligning sources by `StudyInstanceUID`. It writes
`data/labels/audit/public-v1/` with:

- `audit.json`: schemas, coverage, reference agreement/AUC, source transformations,
  fold coverage, dependency versions, manifest hash and auditor hash.
- `target_statistics.csv`, `value_counts.csv`, `verdict_mapping.csv`: per-target
  values, uncertainty indicators and auxiliary confidence/verdict distributions.
- `pairwise.csv`, `large_disagreements.csv`: source differences and 736
  study-target pairs where Steven v4 and Pilkwang differ by more than 0.5.
- `report_group_disagreements.csv`, `fold_reference_coverage.csv`: normalized
  report-group consistency and coverage of the fixed reference folds.

No report text is emitted. Source metadata can change independently of versioned
CSV downloads; the saved metadata snapshots are fingerprinted as evidence, and
a future live metadata response need not reproduce those bytes.

## Coverage and structural validity

All six CSVs have unique, nonblank study IDs, the twelve targets in submission
order, no unexpected study IDs, and no null, nonfinite or out-of-range target
values. All available confidence cells are finite and within `[0, 1]`, and
all verdict cells belong to the `{YES, NO, UNK}` vocabulary, with no nulls.

Steven's files have 13 columns: study ID and twelve targets. Pilkwang has 37:
ID plus target, confidence and verdict for each finding. lixin73's 13-column
and 25-column files have **identical targets**; the latter adds confidence.

Pilkwang's missing study belongs to **reference fold 0**. Its ID and report
group are recorded in the local `audit.json`. This leaves 57 reference rows
for extraction scoring, while covering every report-only training study.
Do not drop that study from image-model validation: its official labels and
images remain available. It can use official labels when eligible for training.
The public notebook's lexicon fallback is separate behavior, not a row supplied
by this CSV.

## Reference agreement

These are locally recomputed **label-table macro ROC-AUCs**, using the original
official binary labels without overwriting public predictions. The common
endpoint contains the same 57 reference studies for every source.

| Label table | Rows | Available reference rows | AUC on available reference rows | AUC on common 57 |
|---|---:|---:|---:|---:|
| Steven v1 | 4,407 | 58 | 0.877991 | 0.877731 |
| Steven v2 | 4,407 | 58 | 0.887349 | 0.887402 |
| Steven v4 blend | 4,407 | 58 | 0.892707 | 0.893237 |
| Pilkwang v2 | 4,406 | 57 | 0.870040 | 0.870040 |
| lixin73, either file | 4,407 | 58 | 0.835194 | 0.834699 |

The published Steven scores reproduce to their reported precision. Pilkwang's
reported 0.870 reproduces on **57**, not 58, studies. The published lixin73
0.8347 also matches the common-57 calculation; its score on all 58 is 0.8352.
Earlier summaries describing every comparison as a 58-study evaluation need
this denominator qualification.

| Target | Steven v1, 58 | Steven v2, 58 | Steven v4, 58 | Pilkwang, 57 |
|---|---:|---:|---:|---:|
| ACL | 0.9933 | 0.9933 | 0.9871 | 0.9974 |
| MCL | 0.9637 | 0.9637 | 0.9683 | 0.9757 |
| Medial Meniscus | 0.9543 | 0.9543 | 0.9483 | 0.9429 |
| Lateral Meniscus | 0.8621 | 0.8621 | 0.8789 | 0.8414 |
| Medial OA | 0.9310 | 0.9310 | 0.9318 | 0.9079 |
| Lateral OA | 0.8298 | 0.8298 | 0.8327 | 0.7885 |
| PF OA | 0.9009 | 0.9009 | 0.9015 | 0.8912 |
| Effusion | 0.8534 | 0.8534 | 0.8770 | 0.8299 |
| Synovitis | 0.6780 | 0.7903 | 0.7903 | 0.6938 |
| Baker's | 0.9457 | 0.9457 | 0.9438 | 0.9324 |
| Contusion | 0.8306 | 0.8306 | 0.8596 | 0.7678 |
| Fracture | 0.7931 | 0.7931 | 0.7931 | 0.8713 |

Use the common-endpoint per-target scores in `audit.json` for direct source
comparisons. The table above preserves the complete available endpoint for
each source; it is not a basis for selecting a different teacher per target.

None of Steven's or Pilkwang's available reference rows is wholly binary or
exactly equal to its official target vector. Thus those files have not simply
replaced all reference predictions with the twelve official binary labels.
lixin73 agrees exactly on six studies, but also disagrees on others and has
continuous values on 23 reference rows; agreement alone cannot diagnose copying.
No table's values can prove the absence of indirect reference-label tuning.

## What uncertainty and confidence actually encode

### Steven: original, repaired, and blended values

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
[tried and rejected](sharing-experiments.md#2026-09-13-label-masking-result): full-fold AUC
fell by 0.011455. The baseline again supervises all nonmissing soft targets.

### Pilkwang: graded scores with explicit verdicts

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

The [inspected baseline notebook](public-research.md#public-code-worth-inspecting)
separately falls back to lexicon labels for uncovered IDs and replaces eligible
training gold targets with official labels. Those are downstream notebook
operations, not evidence of replacement inside the downloaded CSV.

### lixin73: mostly binary targets

**50,859 / 52,884 cells (96.2%)** are exactly zero or one; 168 equal `0.5`.
The two files have identical targets. Confidence values range from 0.35 to 1,
with per-target means approximately 0.976–0.992, but no verdict fields accompany
them. The retrieved dataset description is empty and the snapshot supplies
no extractor/prompt. The model identity in the filename and upstream credit
is author-provided provenance, not independently verified execution evidence.
Do not equate its zero targets with explicit report denials or its confidence
values with demonstrated correctness.

## Source consistency and remaining limits

The fixed folds contain 54 repeated **normalized** report groups, distinct from
the 46 exact-text groups in the original metadata audit. Label vectors differ
within 37 of those groups for each Steven recipe, three groups for Pilkwang,
and one for lixin73. This is extraction consistency evidence, not proof of
duplicate patients or clinical errors; labels have not been averaged or repaired.

Across Steven v4 and Pilkwang, 736 study-target pairs differ by more than 0.5.
ACL accounts for 206 and Contusion for 188 of those pairs. Their numerical
scales and missingness conventions differ, so these are review candidates,
not adjudicated errors. The local disagreement table contains IDs and values,
without report text or reference-based filtering.

Steven's model/prompt and lixin73's extraction details remain unavailable in
these downloaded snapshots. Pilkwang's producer is incomplete without its
helper module. Consequently, this audit establishes structural validity,
observed transformations and reference agreement; it does not establish full
clinical rubric compliance or regenerate any labels.

Steven's public development reports compare repairs and blends against the
reference studies. Their scores, and comparisons selected using them, are
development evidence. Preserve the project's [fixed folds](validation.md),
but do not claim those folds undo prior public reference-set feedback.
The source claim that teacher extraction AUC bounds image-model performance
is also not a mathematical guarantee; see [supervision evidence](public-research.md#report-derived-supervision).

## Implication for the first baseline

**Decision, 2026-09-10:** use **Steven Lee's v4 blend** as the report-derived
label source for the first baseline, as selected by the user after this audit.
The pinned file is
`data/labels/public/stevenleehans/version-6/llm_labels_v4_blend.csv`, from Kaggle
dataset version **6**, identified as `steven_v4` in the
[source manifest](public-label-sources.json), which records its SHA-256.
Use the published soft targets and retain v1/v2 for missingness and repair
provenance. Baseline3 has since measured one fold using v4, but no paired image
model comparison establishes that source as better than v2 or Pilkwang.

Its 0.0054 advantage over v2 on the reused 58-study endpoint does not establish
that it trains a better image model. v2 remains a useful single-source control,
and Pilkwang is usable on the entire report-only pool.

Fold-aware training now retains official gold separately, uses it only on
eligible training rows, and preserves every held-out study under the
[all-study protocol](validation.md#project-validation-protocol). That later
implementation does not change this audit's scope: it altered no labels or
folds and ran no LLM or image-model comparison between label sources.

## 2026-09-13 new public label comparison

**Follow-up:** the recommended Tonylin fold-0 comparison completed with a
public regression, and Steven v4 was restored. See the
[training result](validation.md#2026-09-13-tonylin-label-comparison).
The original source-screening evidence below is retained for provenance.
Zhen V7 and Laymond v2 do not improve overall
reference agreement over Steven v4. Rayan's original release offers a narrower
Fracture-source experiment, but its newer release cannot be justified using the
older 0.8991 score. Those alternative sources have not been adopted or scheduled.

### Published claims and reproduced scores

Scores below are label-table macro ROC-AUC against the original official
binary labels, except where explicitly described as image-model scores.
Each source is aligned by `StudyInstanceUID`, with Steven rescored on the same
available reference studies. No public gold predictions are overwritten by
this audit. An artificial 1.0 caused by an already overwritten source is
excluded from candidate ranking.

| Source and pinned Kaggle release | Published claim | Recomputed label AUC | Interpretation |
|---|---|---:|---|
| [Steven v4, release 6](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels/versions/6) | 0.8927 on gold | **0.892707**, 58 studies | Current comparison source; claim reproduced. |
| [Tonylin soft labels, release 3](https://www.kaggle.com/datasets/tonylin1026/rsna-knee-refined-soft-labels/versions/3) | No numerical claim in retrieved card or released documentation | **0.902700**, 58 | Best measured candidate, +0.009993. Score the `*_soft` columns. |
| [Zhen V7, release 1](https://www.kaggle.com/datasets/zhen931/rsna-knee-weak-labels-v7/versions/1) | No numerical claim in retrieved card or released documentation | **0.879325**, 57 | Steven scores **0.893237** on the same 57; delta -0.013912. |
| [Laymond Qwen2.5-32B v2 filled, release 4](https://www.kaggle.com/datasets/laymond/rsna-knee-abnormality-qwen3-8b-weak-labels/versions/4) | Filled 0.8686; raw 0.8521 on 58 gold studies | **0.868606** filled; **0.852107** raw | Both claims reproduce, below Steven. Score `*__confidence`, which this v2 defines as P(present). |
| [Tasmeem corrected, release 2](https://www.kaggle.com/datasets/tasmeemreza/rsna-knee-refined-llm-labels/versions/2) | **Image-model** gold AUC 0.9062 to 0.9101; public LB 0.919 to 0.926 | Not a usable label benchmark | All 58 rows exactly copy official gold. Direct CSV AUC is 1.0. The model gains are author reports, not reproduced here. |
| [Rayan, original release 1](https://www.kaggle.com/datasets/rayanbabur/rsna-knee-2026-calibrated-soft-targets/versions/1) | 0.8991 on 58 gold studies | **0.899131**, 58 | Claim reproduced from the older release; all improvement is Fracture. |
| [Rayan, current release 2](https://www.kaggle.com/datasets/rayanbabur/rsna-knee-2026-calibrated-soft-targets/versions/2) | Card and bundled report retain 0.8991 | Not a usable label benchmark | All 58 gold rows now copy official labels; silver targets also changed substantially. The older score does not measure this release. |
| [Yunus four-source merged, release 2](https://www.kaggle.com/datasets/yunusgmsoy/rsna-knee-llm-labels-4-source-merged/versions/2) | Card reports 0.8798 for a Gemini source | Not a usable label benchmark | Card describes `report_labels_v4.csv`, while the archive contains merged `report_labels_v5.csv`; all 58 gold rows copy official labels. |

Every table covers all **4,349 report-only studies**, with unique IDs, finite
targets in `[0, 1]`, and no unexpected study IDs. Zhen's two tables and Pilkwang
omit the same one gold study; other compared tables cover all 4,407 studies.
None of Tonylin's reference rows is wholly binary or exactly equal to the
official target vector. This rules out wholesale hard-gold replacement in its
soft columns, but cannot rule out tuning or indirect use of gold labels.

### What the apparent gains contain

Tonylin improves 8 of 12 reference AUCs. Fracture contributes about **77%** of
its net macro gain; excluding Fracture, mean improvement is only **0.0025**.

| Target | Steven v4, 58 | Tonylin soft, 58 | Difference |
|---|---:|---:|---:|
| Fracture | 0.793056 | 0.885417 | +0.092361 |
| MCL | 0.968254 | 0.981859 | +0.013605 |
| Medial Meniscus | 0.948317 | 0.959736 | +0.011418 |
| ACL | 0.987132 | 0.998162 | +0.011029 |
| Lateral Meniscus | 0.878882 | 0.889441 | +0.010559 |
| Lateral OA | 0.832689 | 0.820116 | -0.012573 |
| Contusion | 0.859649 | 0.851552 | -0.008097 |

Across the report-only pool, Tonylin changes **50,642 / 52,188** numerical
targets at tolerance `1e-12`, but only **1,158 / 52,188 (2.22%)** cross the
`>= 0.5` decision threshold. Mean absolute difference is **0.08884**, mean
per-target Spearman correlation **0.90706**, and 311 cells differ by more than
0.5. These numerical differences combine label content and score semantics;
they do not establish clinical errors in either source. Tonylin provides no
generation code or detailed methodology in the inspected release.

A paired study bootstrap with seed **20260913** and **5,000** draws gives a
95% percentile interval of **[+0.00090, +0.02028]** for Tonylin's macro gain.
One draw lacking a class in a target is excluded. This measures resampling
uncertainty on the reused 58-study endpoint, not source-selection bias, prior
gold tuning, patient clustering or variability in training. It supports trying
the source, not declaring a downstream improvement.

Zhen V7 changes only Synovitis and Fracture relative to its bundled V4:
3,670 and 2,460 report-only cells respectively. Their values change without
crossing the 0.5 threshold. Its shared-57 Fracture AUC is strong
(0.896324 versus Steven's 0.791176), but its losses in Contusion, Effusion,
Lateral OA, Lateral Meniscus and Synovitis outweigh that gain. The bundled
Zhen V4 has the same gold AUC as Pilkwang (0.870040) while changing 5,923
report-only target values; equal gold AUC does not imply identical labels.

Rayan release 1 reconstructs as **Steven v4 clipped to `[0.005, 0.995]` on
eleven targets**, to floating-point precision, with **Pilkwang's Fracture
scores exactly on all 4,406 shared studies**. Its Fracture AUC increases from
0.793056 to 0.870139; the other eleven reference AUCs are identical to Steven.
Release 2 changes 51,862 report-only cells relative to release 1, so its
retained numerical claim cannot be carried forward to those new targets.

### Reproduction and experiment boundary

The [comparison source manifest](public-label-comparison-sources.json) pins
the eleven compared tables, dataset versions, explicit target-column mappings,
claims, metadata snapshots, archive/file SHA-256 hashes and official inputs.
Version-specific public Kaggle API download URLs are recorded per source.
Downloaded generator notebooks are stored as source artifacts and were not
executed. The existing production label manifest remains the source of the
active Steven v4 identity.

The [comparison script](../scripts/adhoc/compare_public_labels.py) produced
`data/labels/audit/new-public-20260913/final/`: `audit.json`,
`gold_per_target.csv`, `pairwise.csv` and `large_silver_disagreements.csv`.
The report records script/manifest hashes, dependency versions and bootstrap
configuration. Outputs contain IDs and numerical targets, without report text.
Reproduce into a fresh directory:

```bash
rtk proxy uv run --locked python -m scripts.adhoc.compare_public_labels --output-dir data/labels/audit/new-public-20260913-reproduction
```

A training comparison should vary only report-derived training targets while
retaining the official training gold labels, seed, fold membership, sampling,
model and training budget. Keep the `all_5fold_v3` frozen validation targets:
fold 0 has 904 validation studies, of which only 11 are gold. The other 893
validation targets are frozen report-derived labels, so their score also
measures agreement with the existing teacher. Report the small gold-only
diagnostic separately and assess any later public-LB evidence without treating
the all-study validation score as independent adjudication of label quality.
No training, label-source integration, tests or submission was performed.
