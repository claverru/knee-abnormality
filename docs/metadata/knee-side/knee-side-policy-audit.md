# Knee-side policy audit

**This policy assigned 4,237 of 4,407 training studies (96.14%)** and all
three example-test studies. It was the side source for the retired
[knee-side embedding](../../modeling/retired-side-and-compartment-variants.md#knee-side-embedding), shared by training,
validation and Kaggle inference; its resolver is preserved in the
archived sources (`f076a09:docs/history/knee-side-sources/laterality.py.txt`). Inputs were DICOM headers the loader already
read for supported series; reports, labels, study IDs, lookup tables and
manufacturer never determined the side.

Evidence per study:

- **Explicit:** `Laterality` and `ImageLaterality` values normalized from
  `L`/`LEFT`, `R`/`RIGHT` and `B`/`BOTH`/`BILATERAL`, from every readable header.
- **Text:** the reviewed conservative parser over study/series descriptions,
  protocol name and body part.
- **Coordinates:** per series, the first, middle and last indexed records in filename
  order must have valid native poses; take the median physical image-center X.
  The study center is the median across series. It qualifies with at least two
  series, |X| > 40 mm and the same sign in every series. Positive X is left.

| Evidence | Output |
|---|---|
| Bilateral explicit value, or explicit/text containing both sides | Unknown |
| Qualified coordinates contradict explicit or text side | Unknown |
| Consistent explicit side; coordinates agree or are inconclusive | Explicit side |
| No explicit side; qualified coordinates with absent or agreeing text | Coordinate side |
| Otherwise, including text alone | Unknown |

| Training outcome | Studies |
|---|---:|
| Explicit, coordinates agree | 2,071 |
| Explicit, coordinates inconclusive | 84 |
| Coordinates only | 1,737 |
| Coordinates agreeing with text | 345 |
| Unknown: no qualified evidence | 107 |
| Unknown: conflicting explicit/text | 33 |
| Unknown: coordinates contradict metadata | 15 |
| Unknown: text only | 14 |
| Unknown: bilateral explicit tag | 1 |

Sides are 1,988 left, 2,249 right and 170 unknown. Explicit assignments are
1,071 left and 1,084 right; coordinate assignments are 917 left and 1,165 right.
Coordinates are near the origin in 201 studies and disagree in sign across series
in four. Fold 0 has 412 left, 450 right and 42 unknown of 904; the other folds have
24–37 unknown each. Conflicts are Philips and Siemens acquisitions; contradictions
involve 12 Siemens and 3 Philips studies.

Against the earlier sampled audit, all 2,136 assigned studies with a consistent
explicit reference agree and 15 abstain; all 345 assigned text-reference studies
agree and 15 abstain. Qualified coordinates alone agree with 2,052 of 2,067 explicit
references and all 346 text references. These references are metadata, not
anatomical truth, so the numbers do not measure accuracy on coordinate-only studies.

The policy differs from the earlier [96.85% potential coverage](knee-side-coordinate-investigation.md#agreement-and-potential-coverage)
because it drops text-only assignments, abstains on tag/coordinate contradictions and
requires coordinate sign consensus. Running the loader's own indexing over all
24,386 supported train/test series (819,633 indexed records) gives exactly the same
evidence and sides as applying the resolver to the saved three-header snapshot; the
two known bad instances and unsupported series change nothing.

Example tests: the Siemens study is explicit left (+82.6 mm); the Toshiba
(−88.9 mm) and GE (−93.6 mm) studies are coordinate right knees. Hidden-test
coverage cannot be measured locally.

Artifacts are in `outputs/analysis/k/knee-side-policy-20260914/`: `summary.json`,
per-study `studies.parquet`, loader and snapshot assignments, an empty
`reconciliation.csv`, manufacturer/model side tables, source copy and hashed manifest.
The `dataset.py` hash predates a later collate-only change that carries side records
as dictionaries; the resolver is unchanged. The audit script needs the removed loader
integration: reproduce it from the knee-side run's `source_snapshot.zip` or by applying the
[archived patch](../../history/sources/knee-side/README.md) at `ea19772`.
