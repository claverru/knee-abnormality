# Vendor label associations

**The four minor vendors are a separate English-reporting cohort whose effective labels are
elevated on ten of twelve targets. The evidence points at reporting style, not at a scanner
or a sicker population.** Measured 2026-09-20 on the frozen effective labels, no pixels.

## Cohort

| Vendor | Studies | English words | Spanish words |
|---|---:|---:|---:|
| Siemens | 1,951 | 27.2% | 14.5% |
| Philips | 1,301 | 45.4% | 35.3% |
| GE | 905 | 52.7% | 17.2% |
| Toshiba | 181 | 100% | 3.9% |
| Canon | 45 | 100% | 2.2% |
| Fujifilm | 16 | 100% | 12.5% |
| Hitachi | 8 | 100% | 12.5% |

Those columns are keyword hit rates (` knee|meniscus|tear` and `rodilla|menisco|artrosis`), not a
language classification, so they undercount both languages; what they establish is the contrast.

The three large vendors carry mixed Spanish and English reports, while every one of the 250
minor-vendor reports matches the English vocabulary, so vendor separates a site here, not only a
scanner. Toshiba is
`Vantage` (129, no field strength), `Titan` (51, 1.5T) and one `Vantage Elan`.

## Label rates

Effective labels at `>= 0.5`, minor vendors against the three large ones:

| Target | Minor (250) | Major (4,157) | Ratio | Toshiba (181) |
|---|---:|---:|---:|---:|
| Synovitis | 0.400 | 0.112 | ×3.56 | 0.470 |
| ACL | 0.376 | 0.198 | ×1.90 | 0.459 |
| Fracture | 0.120 | 0.069 | ×1.75 | 0.133 |
| Lateral OA | 0.392 | 0.266 | ×1.48 | 0.354 |
| Medial OA | 0.508 | 0.363 | ×1.40 | 0.486 |
| Baker's | 0.324 | 0.242 | ×1.34 | 0.298 |
| Lateral Meniscus | 0.200 | 0.154 | ×1.30 | 0.199 |
| PF OA | 0.568 | 0.451 | ×1.26 | 0.530 |
| Contusion | 0.204 | 0.172 | ×1.19 | 0.249 |
| Medial Meniscus | 0.452 | 0.401 | ×1.13 | 0.414 |
| MCL | 0.168 | 0.152 | ×1.11 | 0.199 |
| Effusion | 0.556 | 0.596 | ×0.93 | 0.580 |

## Reporting style, not label noise

Toshiba reports have a median 1,317 characters against 968 elsewhere, and enumerate findings
the Spanish reports often omit. The labels follow that text faithfully: a literal
`sinovit|synovit` match appears in exactly 85 of 181 Toshiba reports, the same 85 that carry a
positive Synovitis label, agreeing per study in 179 of 181 cases. The two targets that do not
rise, Effusion and Medial Meniscus, are the two already reported almost everywhere, which is
what a completeness difference predicts and a sicker population does not.

## What this does not establish

These are effective-label rates, not verified disease prevalence. Vendor comes from one
representative header per study, so within-study scanner consistency is unchecked, and site
identity is inferred from vendor plus report language rather than given. The group is 5.7% of
training and spreads across folds (Toshiba 48/28/38/32/35 for folds 0–4), so it moves macro AUC
little, and it holds one gold study in total, so gold labels cannot check it.

## What it changes

Report per-vendor subgroup deltas for changes that touch OA or Synovitis, and read a
vendor-conditioned gain as a possible label-convention effect before calling it an image gain.
The [sharing disagreement audit](../history/sharing/oa-and-meniscus-disagreement-audit.md)
owns the model-ranking sensitivity to acquisition mixture and the vendor-prior diagnostic; this
page owns the corpus-level rates behind it.

## Reproduce

Join `data/folds/all_5fold_v3/targets.csv` (thresholded at `>= 0.5`, the
[validation protocol](../validation/validation-splits.md#all_5fold_scanner_v2)'s frozen effective labels)
and `data/raw/train.csv` reports to the per-study `vendor` column of
`outputs/analysis/s/sharing-disagreement-20260914/study_headers.parquet`, then group by vendor.
