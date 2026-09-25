# Public report-label audit

The [September 13 comparison](2026-09-13-new-public-label-comparison.md) found
Tonylin's soft labels the strongest candidate by reference agreement, but the
[completed training comparison](../validation/results/2026-09-13-tonylin-label-comparison.md)
scored **0.921 publicly versus Steven v4's 0.929**. Retain Steven v4; the
higher label-table AUC did not translate into a better submitted model.

Audited on **2026-09-10** using the downloaded tables and producer material.
Steven Lee's v2 and v4 tables are structurally usable starting candidates: both
cover all 4,407 studies with finite targets in `[0, 1]`. Pilkwang's table covers
all **4,349 report-only studies**, but omits one reference study. Following the
audit, **Steven v4 was selected for the first baseline**; see the
[recorded decision](source-consistency-and-first-baseline-choice.md#implication-for-the-first-baseline). No label values were changed.

The main integration constraint is **source-specific uncertainty**. Steven v1's
`0.5` and Pilkwang's `UNK` retain missing report evidence; Steven v4's blended
probabilities do not retain that distinction. Keep the component tables.
Extraction agreement on the reference set does not establish downstream
image-model quality or independent validation.

The [2026-09-14 blinded disagreement audit](../history/sharing/oa-and-meniscus-disagreement-audit.md#source-lineage-and-prediction-effects)
adds a separate severity concern: 19 reviewed report-derived OA positives
describe explicitly low-grade findings under the host's high-grade endpoint.
Steven v1/v2 and lixin agree positively on all 19; v4 values are 0.80–0.93.
Confidence or source-agreement filtering would retain them. This is a selected
report-text audit, not an estimate of population label errors or image
adjudication. One additional conflict involves official gold, which is preserved.
No labels changed; the existing Steven-v4 training decision remains in place
pending a controlled OA-only supervision comparison.

## Acquisition and reproduction

The [source manifest](../../configs/labels/public-label-sources.json) records dataset IDs, explicit
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

[`scripts/adhoc/audit_public_labels.py`](../../scripts/adhoc/audit_public_labels.py) verifies
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

A rerun overwrites these files; pass `--output-dir` to write elsewhere. The source names are
fixed in the script. No report text is emitted. Source metadata can change independently of versioned
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
