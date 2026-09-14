# Competition metadata

Five official CSVs are available in `data/raw/` and were audited with Polars
on 2026-09-09. Training studies have a variable number of series. Label
coverage and target counts live in [supervision](supervision.md#available-supervision);
pixel handling lives in [data preparation](data-preparation.md).

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
and `Anatomical_Plane`. The [target table](supervision.md#reference-label-distribution)
records prediction columns in submission order.

To obtain just these files with the installed Kaggle CLI, run from the root:

```bash
for file in train.csv train_series.csv test.csv test_series.csv sample_submission.csv; do
    rtk proxy uv run --locked kaggle competitions download rsna-knee-abnormality-detection -f "$file" -p data/raw -q
done
rtk proxy uv run --locked python -m scripts.adhoc.audit_metadata
```

[`scripts/adhoc/audit_metadata.py`](../scripts/adhoc/audit_metadata.py) reads the local CSVs
with Polars and writes aggregates, full schemas, null counts, and file SHA-256
hashes to `data/metadata/audit.json`; target counts also go to
`data/metadata/label_counts.csv`. It does not download files, modify raw data,
or emit report text. Data and generated audit artifacts are ignored by Git.
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
private performance. See the [case-mixture investigation](sharing-generalization.md#public-subset-size-and-case-mixture-follow-up).

The CSVs provide no patient identifier, pixel geometry, or transfer syntax.
They cannot establish patient-level separation, slice ordering, codec needs,
or actual image duplication.

## Representative acquisition inventory

The later [disagreement audit](sharing-disagreement-audit.md#acquisition-inventory-and-population-sensitivity)
read one representative DICOM header for all 4,407 studies on 2026-09-14,
without pixel decoding or GPU use. Its
`outputs/analysis/sharing-disagreement-20260914/study_headers.parquet` records
manufacturer/model, field strength, sex and representative acquisition settings.
All reads succeeded; manufacturer/model are present throughout, with 238 missing
field-strength and sex values each. One header per study does not establish
within-study scanner consistency or identify hospital sites. The linked audit
owns its label-association and validation-stratum results.

## Heavy report normalization

On 2026-09-12, deterministic lexical normalization of all 4,407 `Report`
values in the training snapshot above produced **4,184 distinct nonempty
token fingerprints**. Two reports became empty after removing their preambles;
including the empty string as a value gives 4,185 distinct values.

| Transformation | Distinct nonempty texts | Empty rows |
|---|---:|---:|
| Original text | 4,276 | 0 |
| Case folding and collapsed whitespace | 4,257 | 0 |
| Unicode/diacritic folding and punctuation removal | 4,235 | 0 |
| Heavy normalization, retaining word and clause order | 4,212 | 2 |
| Heavy normalization with sorted, deduplicated clauses | 4,209 | 2 |
| Heavy normalization with sorted, deduplicated words | 4,184 | 2 |

The heavy transformation decodes HTML entities, folds Unicode and diacritics,
removes recognized redaction placeholders and preambles before findings,
strips report headings and selected administrative lines, maps selected
English ligament names and abbreviations to a common representation, standardizes
selected spelling variants and compounds, replaces numeric values and
measurements with placeholders, removes recognized left/right terms and
selected multilingual function words, joins hyphenated words, and deduplicates
clauses. Negation words remain. Rules cover selected forms observed in the
multilingual reports, without translation, stemming, or fuzzy clustering.

The final fingerprints have 55 repeated groups containing 276 rows, with
221 rows beyond one representative per nonempty group; the largest group
has 37 rows. This is a deliberately lossy lexical count. Inspection found
that word-set equality can merge reports differing in which meniscus is torn
or whether a Baker's cyst is absent: retained words do not retain negation
scope or anatomy associations. Even the ordered heavy result discards numeric
grades, measurements and laterality. These counts do not establish clinical,
patient or image equivalence. The two empty outputs have findings headings
without findings text; one still has a clinical-history preamble in its raw
report.

Reproduce with:

```bash
rtk proxy uv run --locked python -m scripts.adhoc.normalize_reports
```

[`scripts/adhoc/normalize_reports.py`](../scripts/adhoc/normalize_reports.py)
writes `data/metadata/report_normalization/summary.json`,
`normalized_reports.parquet` with raw text and every normalization stage,
and `duplicate_groups.csv` with final fingerprints and study membership.
The JSON records input/script/output SHA-256 hashes, settings, Python,
Polars and Unicode versions, counts and limitations. The run uses no
randomness. Outputs stay under Git-ignored `data/`; the raw CSV is unchanged.

## Series profiles

Training studies have 3–14 series (median 5, 95th percentile 9). The observed
metadata combinations are:

| Fluid sensitive | Fat suppression | Plane | Series | Studies |
|---:|---:|---|---:|---:|
| 0 | 0 | Axial | 1,179 | 857 |
| 0 | 0 | Coronal | 3,985 | 3,406 |
| 0 | 0 | Sagittal | 5,197 | 4,266 |
| 1 | 1 | Axial | 4,719 | 4,407 |
| 1 | 1 | Coronal | 4,624 | 4,248 |
| 1 | 1 | Sagittal | 4,667 | 4,150 |

In this training snapshot, the two flags always agree and all studies have a
fluid-sensitive, fat-suppressed axial series. Do not assume those properties
for unseen data; see the [host's metadata clarification](data-preparation.md#public-data-contract-and-decoding).
A total of 1,936 studies repeat at least one metadata combination, spanning
2,898 repeated groups. Matching metadata does not establish matching pixels,
so these groups must not be deleted as duplicates. The six observed combinations
also do not establish the participant's separate five-category mapping in the
[original preparation notes](data-preparation.md#data-shape-and-redundancy).

## Images per series

On 2026-09-10, a read-only aggregation of the completed acquisition journal
`data/raw/.download/progress.sqlite` counted paths matching `train_series/%.dcm`
by study and series UID, then joined them one-to-one to `train_series.csv`.
All 24,371 training series matched, totaling 819,078 DICOM files. This used
the existing inventory without scanning image directories or decoding pixels.

| Plane | Fluid sensitive / fat suppression | Minimum | Median | 95th percentile | Maximum |
|---|---|---:|---:|---:|---:|
| Axial | 0 / 0 | 14 | 32 | 80 | 254 |
| Axial | 1 / 1 | 12 | 32 | 144 | 192 |
| Coronal | 0 / 0 | 11 | 30 | 37 | 48 |
| Coronal | 1 / 1 | 15 | 30 | 38 | 80 |
| Sagittal | 0 / 0 | 11 | 30 | 60 | 200 |
| Sagittal | 1 / 1 | 15 | 29 | 47 | 320 |

Across all series, the minimum is 11, median 30, 5th percentile 18, 95th
percentile 45, and maximum 320. **69.92%** have at most 32 files. Quantiles
use Polars' `nearest` interpolation, and every series contributes one count;
these are not weighted equally by study or restricted to a selected acquisition.

Per-series counts are saved in `data/metadata/train_series_image_counts.parquet`.
`data/metadata/train_series_image_count_summary.json` records the aggregates,
the metadata source hash and the Parquet hash. The source snapshot is identified
by the [completed acquisition](data-preparation.md#completed-acquisition-and-local-verification).
These are DICOM-file counts, including malformed files. They do not establish
the number of usable slices, unique physical positions, frames, slice spacing
or anatomical coverage. Pixel/geometry preparation must establish those.

### Shortest-series image limits

The initial single-acquisition recipe selected the shortest acquisition for
each study/type and used the median image count of those acquisitions as its K.
The current baseline instead pools inputs across usable acquisitions; this
section preserves the historical derivation of the earlier limits.
On 2026-09-10, Polars grouped the saved counts by study, plane and the two
contrast flags, sorted by image count and then series UID, and retained the
first acquisition. One file was subtracted from each of the two series with
[confirmed malformed payloads](data-preparation.md#public-data-contract-and-decoding)
before selection. Other files were not exhaustively decoded.

| Plane | Fluid sensitive / fat suppression | Selected acquisitions | Median / initial K |
|---|---|---:|---:|
| Axial | 0 / 0 | 857 | 34 |
| Axial | 1 / 1 | 4,407 | 32 |
| Coronal | 0 / 0 | 3,406 | 29 |
| Coronal | 1 / 1 | 4,248 | 30 |
| Sagittal | 0 / 0 | 4,266 | 30 |
| Sagittal | 1 / 1 | 4,150 | 30 |

These medians give one observation to each represented study/type. They can
differ from the all-acquisition medians above because repeated acquisitions
no longer give some studies more weight. All six medians are integers.

`data/metadata/shortest_series_image_limits.json` records the selection rule,
excluded series, counts, Polars version and input Parquet SHA-256
`75c5f7b77f69d6c522529d42cb5e3dddfa0eff2120f501a6b298d2a2a6a6ce55`.
The values are historical inputs to the first model plan, not claims of optimal
coverage or usable-image counts for every acquisition.
