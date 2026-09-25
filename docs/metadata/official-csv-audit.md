# Competition metadata

Five official CSVs are available in `data/raw/` and were audited with Polars
on 2026-09-09. Training studies have a variable number of series. Label
coverage and target counts live in [supervision](../modeling/supervision.md#available-supervision);
pixel handling lives in [data preparation](../data-preparation/INDEX.md).

## Inventory and acquisition

The files came from the
[official competition data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)
and total 9,151,736 bytes. No images were downloaded for this audit.

| File | Rows | Schema |
|---|---:|---|
| `train.csv` | 4,407 | `StudyInstanceUID`, `Report`, 12 nullable integer target columns |
| `train_series.csv` | 24,371 | Study and series UIDs, two integer flags, anatomical plane |
| `test.csv` | 3 | `StudyInstanceUID` only |
| `test_series.csv` | 15 | Same columns as training series |
| `sample_submission.csv` | 3 | Study UID and 12 floating-point prediction columns |

UIDs and report/plane text are strings. Series columns, in order, are
`StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive`, `Fat_Suppression`,
and `Anatomical_Plane`. The [target table](../modeling/supervision.md#reference-label-distribution)
records prediction columns in submission order.

To obtain just these files with the installed Kaggle CLI, run from the root:

```bash
for file in train.csv train_series.csv test.csv test_series.csv sample_submission.csv; do
    rtk proxy uv run --locked kaggle competitions download rsna-knee-abnormality-detection -f "$file" -p data/raw -q
done
rtk proxy uv run --locked python -m scripts.adhoc.audit_metadata
```

[`scripts/adhoc/audit_metadata.py`](../../scripts/adhoc/audit_metadata.py) reads the local CSVs
with Polars and writes aggregates, full schemas, null counts, and file SHA-256
hashes to `data/metadata/audit.json`; target counts also go to
`data/metadata/label_counts.csv`. It does not download files, modify raw data,
or emit report text. Reruns overwrite existing outputs; `--output-dir` selects another
destination. Data and generated audit artifacts are ignored by Git.
The CLI returned plain CSVs for this snapshot; if a future download returns a
ZIP, extract its named CSV into `data/raw/` before running the audit.

The training files anchoring the findings below have these SHA-256 hashes:

```text
train.csv        8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33
train_series.csv 573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3
```

Discovery inspected three API pages (600 entries), stopping after reaching
training image entries. `data/metadata/discovery.json` records that partial
listing; it is not a complete inventory or size estimate of the image dataset.

## Identifiers and joins

There are no missing identifiers, duplicate study IDs, duplicate series IDs,
or duplicate full rows in the relevant CSVs. Every study has series, and every
series references a study in its corresponding study table. The example test
study IDs do not overlap training; the submission template matches the test
IDs and their order, with finite predictions in `[0, 1]`. These three example
rows do not establish the hidden evaluation set's size or composition.

The [official data description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data),
rechecked on 2026-09-14, states approximately **1,300 test studies** and warns
that abnormality prevalence need not match training, public and final evaluation
sets. The [leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)
states approximately **30% public and 70% private**, implying roughly **390
public and 910 private studies**. Those are approximate derived counts, not
an inspected hidden manifest. Final standings use the private portion, not the
full public/private union. The 4,407-study training corpus is larger than the
stated test set; submission duration cannot establish hidden study counts.

Our 904-study validation fold is larger than the public subset, but its 893
report-derived and 11 gold labels measure a different endpoint from expert
adjudication. Sample count alone does not determine which ranking better predicts
private performance. See the [case-mixture investigation](../history/sharing/generalization-public-subset-and-case-mixture.md).

The CSVs provide no patient identifier, pixel geometry, or transfer syntax.
They cannot establish patient-level separation, slice ordering, codec needs,
or actual image duplication.
