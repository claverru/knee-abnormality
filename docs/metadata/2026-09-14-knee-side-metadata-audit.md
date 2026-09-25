# 2026-09-14 knee-side metadata audit

**Explicit tags and description text do not cover every study.** The
[coordinate investigation below](knee-side-coordinate-investigation.md) extends
candidate coverage beyond these fields. The completed
CPU audit covers all 4,407 training studies and three local example-test studies:
24,371 training series plus 15 test series, with three headers per series at
the first, middle and last lexicographic filename positions. All 73,158 reads
succeeded and matched their expected study/series UIDs. Other instances within
each series were not read; this is full study/series coverage with sampled headers.

| Evidence status | Training studies | Percentage |
|---|---:|---:|
| Consistent explicit side, without detected textual disagreement | 2,151 | 48.81% |
| Side inferred from series-description text only | 360 | 8.17% |
| No usable side evidence in inspected fields | 1,863 | 42.27% |
| Conflicting side evidence | 33 | 0.75% |

The 2,511 possible assignments cover **56.98%** of training studies using tags
and descriptions. This is field coverage, not measured anatomical accuracy or
a limit on coordinate-based inference. Text-only assignments
are provisional. The explicit group contains 1,074 left and 1,077 right knees;
the text-only group contains 188 left and 172 right candidates.

Twenty-five studies have both L and R in explicit `Laterality` tags across
different series; all are from Philips acquisitions. No sampled series changes
its explicit tag across its three inspected files. Eight additional studies
have one explicit side but an opposite side in descriptions: six Siemens and
two Philips. Examples include `Laterality=L` with `RT_pd_tse_fs_cor`, and
`Laterality=R` with `ROTULA IZQ`. Neither majority voting nor tag priority has
been used to resolve these disagreements.

Descriptions contain side evidence in 535 studies with a single explicit side;
527 agree and eight disagree. This overlap does not establish accuracy for the
360 text-only cases or for another acquisition source. The audit recognizes
full side words, common `LT`/`RT`/`IZQ` shorthand and context-dependent L/R
tokens. It does not assign side from ambiguous protocol suffixes such as `_d`.
The initial stricter parser and source snapshot are retained separately; the
reviewed parser is the canonical result below.

`StudyDescription`, `ProtocolName`, `ImageLaterality` and `PatientOrientation`
provide no nonempty values in the inspected headers. Available explicit evidence
comes from `Laterality`; the inspected anatomy/frame sequences provide no
additional side assignments. DICOM defines separate series, image and frame
laterality attributes, but definitions do not guarantee populated or accurate
values. [DICOM laterality](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.3.html).

## Acquisition dependence and test examples

| Manufacturer group | Studies | Explicit | Text only | Unknown | Conflict |
|---|---:|---:|---:|---:|---:|
| Siemens | 1,949 | 1,618 | 49 | 276 | 6 |
| Philips | 1,301 | 470 | 106 | 698 | 27 |
| GE | 905 | 37 | 205 | 663 | 0 |
| Toshiba | 181 | 0 | 0 | 181 | 0 |
| Canon | 45 | 0 | 0 | 45 | 0 |
| Fujifilm | 16 | 16 | 0 | 0 | 0 |
| Hitachi | 8 | 8 | 0 | 0 | 0 |
| Both Philips and Siemens recorded | 2 | 2 | 0 | 0 | 0 |

Groups normalize manufacturer names across all inspected series; the two mixed
studies remain separate. Missing side is strongly associated with acquisition
source. An unknown-side embedding would therefore expose an acquisition cue,
and complete-case-only training would change the training distribution.

Of the **three example-test studies**, the Siemens study has an explicit L tag;
the GE and Toshiba studies remain unknown. Hidden Kaggle test metadata is not
available locally, so its coverage cannot be inferred from these three examples.

Image position and orientation describe points in a patient coordinate frame;
they do not establish a universal body-midline origin or supply the missing
organ-side label. Inferring knee side from the sign of absolute X would require
additional, validated assumptions about acquisition positioning.
[DICOM coordinate convention](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html).

## Implications and reproduction

The [eight model proposals](../research/knee-side-and-location.md) require explicit
handling of unknown and contradictory metadata. A first embedding experiment
can retain an unknown state; universal anatomical coordinate normalization needs
another reliable side source or a deliberately defined missing-side path.
An image-based side predictor is a separate unimplemented proposal. Training
reports cannot supply a side input at hidden-test inference. These findings do
not authorize changing model code, dropping studies or assigning conflict cases.

The [reviewed summary](../../outputs/analysis/knee-laterality-20260914/reviewed/summary.json),
[study assignments](../../outputs/analysis/knee-laterality-20260914/reviewed/studies.parquet),
[field evidence](../../outputs/analysis/knee-laterality-20260914/reviewed/evidence.parquet)
and [manifest](../../outputs/analysis/knee-laterality-20260914/reviewed/manifest.json)
preserve counts, paths, metadata values, versions and hashes. Original DICOM
files were not fully hashed to avoid reading pixel data; the header snapshots
are fingerprinted. No pixels were decoded, models executed, tests run or GPU used.

Reproduce the reviewed analysis without rereading DICOMs:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES= POLARS_MAX_THREADS=1 uv run --locked python -m scripts.adhoc.audit_knee_laterality --headers outputs/analysis/knee-laterality-20260914/headers.parquet --output-dir outputs/analysis/knee-laterality-reanalysis
```

Omit `--headers` to collect a fresh snapshot from the official local inventories.
The [audit script](../../scripts/adhoc/audit_knee_laterality.py) uses two CPU
threads by default and refuses an existing output directory.
