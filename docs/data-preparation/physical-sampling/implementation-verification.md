# Implementation verification

The maintained metadata command calls the production sampler on all 128 pinned
studies, with seed 20260913 and 32 randomized epochs per type, plus deterministic
2× validation. The 595 accepted-center groups and 20 local-fallback groups are
reported separately. Outputs contain source/config/input hashes, final sampler
identity, target/realized depths, source entitlements/counts, run diagnostics and
placement errors. Historical comparison results remain embedded in the summary.

```bash
rtk proxy uv run --locked python -m scripts.adhoc.analyze_physical_sampling --output-dir outputs/analysis/physical-sampling-new/geometry
```

It reads saved metadata only and refuses an existing output directory. The saved
headers carry no series tags and bypass the loader's instance guard, so its source
ranking sees only crop coverage and UID. Since the known-bad-instance filter was
removed on 2026-09-20 with the repaired DICOMs, the cohort contributes two slices
that earlier runs skipped.

Final artifacts are in
[`geometry-verified`](../../../outputs/analysis/physical-sampling-implementation-20260913/geometry-verified/summary.json),
[`preparation-verified`](../../../outputs/analysis/physical-sampling-implementation-20260913/preparation-verified/report.json)
and the [implementation report](../../../outputs/analysis/physical-sampling-implementation-20260913/implementation-report.json).
Both manifests match the final production preprocessing identity and preserve
source/config/input identities. Earlier directories in this implementation
folder are intermediate verification passes.

| Production physical path | Randomized, 32 draws | Deterministic validation |
|---|---:|---:|
| Selected triplets | 99,968 | 6,248 |
| Target fraction beyond ±30 mm, weighted by target count | 37.40% | 37.36% |
| Realized middle fraction beyond ±30 mm | 37.29% | 37.68% |
| Realized middle fraction beyond ±40 mm | 19.45% | 19.21% |
| Mean unique source files/type | 14.99 | 27.46 |
| 95th percentile placement error | 1.98 mm | 1.95 mm |
| Mean absolute source-share error in repeated groups | 2.99 percentage points | 0.94 percentage points |
| Supplier changes / missing inputs | 0 / 0 | 0 / 0 |

The original sigma-80 prototype also supplied 6,248 validation triplets, 37.68%
beyond ±30 mm, 27.46 unique files/type and 1.95-mm p95 displacement. Its ±40-mm
fraction was 19.24%. Production uses explicit operation seeds and canonical
source/geometry ordering, so final tie placements and randomized draws can
change. The original `requested_fraction` averaged per-group fractions; the new
target fraction weights actual target counts and should not be compared as the
same aggregation. The physical count remains **10.4% above the old sampler's
5,659 validation inputs**, with no runtime or accuracy inference.

All 721 acquisitions passed the run and plane-agreement rules: zero exclusions
for missing valid runs, gap splits, singleton runs or plane disagreements. The
four rejected-center studies supplied 20 local groups, 210 validation triplets
and 3,360 randomized triplets, with no shortfalls or supplier changes.
Their validation p95 local placement error was 0.0320 on the [0,1] coordinate.
The tables retain per-type budgets, repeated/dense/wide and split-membership
strata, target depths and source-specific intended/preferred/realized counts.

The CPU inspection covered eight studies in both modes: four accepted centers
and all four rejected centers. All serial/four-thread and reversed-row outputs
matched. The 88 prepared acquisitions contained 1,881 channel appearances from
1,508 unique-slice appearances across preparations, including 373 reuses.
Independent clipping populations and bounds matched; positions, native maps,
rectangles and reused images were aligned. All 48 local preparations had zero
valid XYZ flags. No decode failures occurred. Gap/singleton/plane-disagreement,
SliceLocation-only and retry branches therefore remain unexercised by these
real-data cohorts; no artificial fixtures were created.

Ruff lint and format checks passed for the affected scope (32 Python files),
config parsing passed, the catalog check passed for 25 scripts, and
`git diff --check` passed. A broader Ruff check found pre-existing issues in
unrelated scripts, listed in the implementation report; those files were not
changed. No tests, models, training, optimizer steps, checkpoint evaluation or
submission execution were run. The completed plan is retained with the
implementation artifacts; its durable behavior is documented here.
