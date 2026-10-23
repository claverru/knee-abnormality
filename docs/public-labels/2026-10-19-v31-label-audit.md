# v3.1 label error audit (2026-10-19)

The next label pass should first repair specific rule failures, then address OA depth and
extent. This audit reviewed report text and existing predictions; it did not inspect MRI
images, change labels, or launch training. The targeted follow-up is recorded in
[v3.2 labels and comparison](2026-10-19-v32-labels.md).

## Inputs and method

- Table: `data/labels/prepared/rebuild-state-v3-1/labels.csv`, SHA-256
  `2c0494c067ba05bb0330adb66971db97c20c180597e8d98cc366c23ab0f0e652`.
  There are 4,407 studies, 58 gold studies, 40 masked silver studies and 51,708 observed
  silver cells. The table is shared with the `knee-abnormality-labels-v3` worktree.
- Manifest: `/home/claudio/src/knee-abnormality-labels-v3/configs/labels/rebuild-state-v3-1-label-sources.json`.
  Its hash matches; its recorded size is stale (1,300,919 versus 1,300,053 actual bytes).
- Implementation reviewed: that worktree's `scripts/tools/prepare_rebuilt_labels.py`,
  particularly `apply_v3_rules`; the draft in `outputs/logs/rebuild/label-audit/final/`
  differs from the implemented rules.
- Report answers: `outputs/labels-v3-reading/answers_adjudicated.parquet`.
- Image witnesses: arithmetic mean of ConvNeXt and ResNet-50 `gen1` folds 0–4, and NextViT
  `silver` folds 0, 1, 3, 4 with `nextvit-gen1-fold-2`. Each family covers all 4,407 IDs
  exactly once; every prediction's fold matches `all_5fold_scanner_v2/studies.csv`.
- Rank by the disagreement with the closest of the three families: target minus the highest
  family prediction for targets at least 0.5, or the lowest family prediction minus target
  otherwise. This avoids one extreme family dominating the queue.
- Strict lower-target candidates require target ≥0.5, every family ≤0.3, and report verdict
  `no` or `unlikely`. Higher-target candidates require target <0.4, every family ≥0.6,
  and verdict `yes` or `probable`. Gold and null cells are excluded.

These are selection signals, not independent truth: image models share training labels and
report verdicts can be wrong. Fold alignment alone does not establish independence from
earlier teacher generations. Report-only contradictions do not establish image ground truth;
the [host criteria](../metadata/official-label-criteria.md) explain this distinction.

## Coverage missed by pass 2

`outputs/labels-v3-reading/pass2/candidates.py` required v3 to equal v2 in the candidate
cell, excluding every rule-changed cell, and kept at most 20 per label and direction.
The new screen retains rule-changed cells and removes that limit.

It selects 118 cells, including 51 absent from the earlier review. Of those 51, 38 are PF OA
and 8 Medial OA. Text review of all 51 found:

| Assessment | Cells |
|---|---:|
| OA report describes grade II, at most half-thickness involvement, or a measured 2.5–8 mm lesion: priority candidates for lower targets | 29 |
| OA depth or extent remains unspecified: image confirmation needed | 17 |
| ACL avulsion or single-bundle injury: severity needs review | 2 |
| Meniscus substance-loss floor overwritten by peripheral-tear cap | 1 |
| Degenerative meniscus thinning read as substance loss: morphology needs review | 1 |
| Explicitly absent contusion raised by an acute-injury answer | 1 |

The 29 are textual threshold mismatches, not 29 newly established image-label errors. Grade
terminology can vary; retain its named grading system and evidence rather than mapping every
Roman numeral globally. No new numeric target values were fitted here.

## Concrete rule failures and high-priority cases

Batch and report numbers identify rows in `batch_index.csv`. Image scores below are the
three-family means, rounded to three decimals.

| Case | Target / image score | Evidence and implication |
|---|---|---|
| Fracture, `batch-09-en_GE` [35] | 0.300 / 0.570 | Displaced 15 mm osteochondral bone fragment, another fracture, fracture-dislocation and large lipohemarthrosis. F3 lowers v2's 0.861 because the answer contains only `osteochondral`. |
| Fracture, `batch-25-en_Siemens` [28] | 0.300 / 0.667 | Detached ossific fragment, donor site and lipohemarthrosis. F3 again suppresses definite bony injury. |
| Fracture, `batch-24-en_Siemens` [59] | 0.300 / 0.413 | Explicit acute osteochondral fracture, 1.5 × 1.3 cm. Acuity is lost in the categorical answer used by F3. |
| Contusion, `batch-08-en_GE` [2] | 0.550 / 0.109 | Explicit “No evidence of bone contusion”; suspected small osteochondral lesion became `acute_osteochondral_injury`, triggering C5 from v2's 0.037. |
| Lateral meniscus, `batch-02-de` [16] | 0.150 / 0.634 | Definite substance defect and a separate possible peripheral tear coexist. R2 floors to 0.75, then R1 caps to 0.15. |
| PF OA, `batch-00-bg` [10] | 0.723 / 0.248 | Patellar defect measures only 2.5 × 2.0 mm despite a grade-3 summary. Depth alone does not establish qualifying extent. |
| PF OA, `batch-17-en_Philips_dStream` [38] | 0.740 / 0.234 | Partial-thickness patellar defect is 6 × 5 mm; grade-3 wording should not erase its measured extent. |
| ACL, `batch-20-en_Philips_other` [87] | 0.200 / 0.906 | Only a few continuous fibers persist, but `chronic_partial` triggers A2. Chronic ACL injury is not categorically excluded by the host's ACL definition; severity remains unresolved. |

The R1/R2 overlap exists in one silver lateral-meniscus answer row and no medial rows.
Do not fix it by globally reversing the rule order: resolve whether the report describes
definite qualifying morphology and a separate uncertain finding. Likewise, keep chondral-only
lesions separate from acute fractures containing bone.

## Largest disagreements that text review cannot resolve

The largest unreviewed gap is lateral meniscus, `batch-45-tr` [50]: target 0.930, image
0.054. The report explicitly describes a bucket-handle tear, so the label matches the text.
ACL, `batch-38-hr` [97], is the reverse: target 0.024, image 0.893, with normal ligaments in
the report. Neither is grounds for an automatic text-label correction.

Three studies have strong paired meniscus reversals: the report-derived target is >0.75 on
one side and <0.2 on the other, while image predictions are <0.2 and >0.75 respectively.
For example `batch-10-en_GE_EXCITE_short` [4] has medial target/image 0.917/0.112 and lateral
0.046/0.921. Inspect images and side mapping before deciding between report mislocalization,
model routing error and an ordinary prediction error.

## Artifacts

All are under `outputs/labels-v3-reading/v3_1-audit/`:

- `provenance.json`: source hashes, prediction runs, fold checks and exact selection criteria.
- `ranked_cells.parquet`: all 52,884 study-label cells, reports, answers, three family scores,
  previous reviews and version comparisons.
- `consensus_candidates.csv`: all 118 strict candidates.
- `text_review.csv`: assessments and full reports for the 51 previously unreviewed candidates.
- `additional_rule_cases.csv`: five further fracture/ACL cases outside that strict queue.
- `meniscus_reversals.csv`: three paired reversals for image and side-mapping review.
