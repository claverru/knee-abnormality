# Heavy report normalization

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

[`scripts/adhoc/normalize_reports.py`](../../scripts/adhoc/normalize_reports.py)
writes `data/metadata/report_normalization/summary.json`,
`normalized_reports.parquet` with raw text and every normalization stage,
and `duplicate_groups.csv` with final fingerprints and study membership.
The JSON records input/script/output SHA-256 hashes, settings, Python,
Polars and Unicode versions, counts and limitations. The run uses no
randomness. Outputs stay under Git-ignored `data/`; the raw CSV is unchanged.
Reruns overwrite existing outputs; `--input` and `--output-dir` select other paths.
