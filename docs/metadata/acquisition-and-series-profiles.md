# Acquisition and series profiles

## Representative acquisition inventory

The later [disagreement audit](../history/sharing/oa-and-meniscus-disagreement-audit.md#acquisition-inventory-and-population-sensitivity)
read one representative DICOM header for all 4,407 studies on 2026-09-14,
without pixel decoding or GPU use. Its
`outputs/analysis/sharing-disagreement-20260914/study_headers.parquet` records
manufacturer/model, field strength, sex and representative acquisition settings.
All reads succeeded; manufacturer/model are present throughout, with 238 missing
field-strength and sex values each. One header per study does not establish
within-study scanner consistency or identify hospital sites. The linked audit
owns its label-association and validation-stratum results;
[vendor label associations](vendor-label-associations.md) owns the corpus-level
label rates by vendor.

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
for unseen data; see the [host's metadata clarification](../data-preparation/public-data-contract-and-decoding.md).
A total of 1,936 studies repeat at least one metadata combination, spanning
2,898 repeated groups. Matching metadata does not establish matching pixels,
so these groups must not be deleted as duplicates. The six observed combinations
also do not establish the participant's separate five-category mapping in the
[original preparation notes](../data-preparation/unverified-preprocessing-notes.md#data-shape-and-redundancy).

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
by the [completed acquisition](../data-preparation/downloading-and-extracting-images.md#completed-acquisition-and-local-verification).
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
[confirmed malformed payloads](../data-preparation/public-data-contract-and-decoding.md)
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
