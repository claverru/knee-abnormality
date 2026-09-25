# Historical regional fusion

The replaced model combined global triplet descriptors with 27 width-64 spatial
queries and per-plane attention. The `spatial-fusion-radius` checkpoint selected
epoch 8 at 0.88635162 AUC. Its regional tokens received 4.94% mean final attention,
and omitting them reduced AUC by 0.00069668 on 904 validation studies. Its
independently trained advantage over `crop140mm` was only 0.00022998 AUC;
these single-fold results do not establish repeatability. See the
[recorded diagnosis](../validation/results/2026-09-13-per-plane-fusion-usage-diagnosis.md),
[earlier runtime rejection](../validation/results/2026-09-12-spatial-fusion-engineering-result.md)
and [archived sources](../history/regional-sources/README.md).
The max-pooling comparison selected 0.87907846 AUC versus 0.88416888 for its
average-pooling reference; it did not justify promoting max pooling. Regional
bandwidth, finer regional grids and adaptive queries are retired.

## Per-plane coverage radius

Selected on 2026-09-13, before training: **18 mm**, independently of the fixed
35-mm attention bandwidth. The selection rule takes the 95th percentile of the
nearest valid anchor distance for accepted-center query/plane groups with at
least one valid anchor, then rounds up to the next whole millimeter. Missing
planes are excluded from that distance distribution and remain uncovered.

The measurement used a uniform 128-study sample from the `all_5fold_v3` fold-0
training partition, seed 42. Original dataset indices preserve epoch-zero
training acquisition and triplet sampling with the configured 1× budgets.
Actual prepared 224-pixel crops supply the 7×7 anchors and all-channel validity;
augmentation and series dropout are disabled for radius selection. No held-out
geometry or predictions enter the choice, and no model executes.

Of 128 studies, 121 had accepted centers, supplying 9,801 measured query/plane
groups. Nearest-anchor distances were 10.07 mm at the median, 14.48 mm at p90,
17.29 mm at p95 and 25.37 mm at p99. The selected radius permits **93.78%** of
all sampled regional queries; plane coverage is 89.90% axial, 88.95% coronal
and 92.51% sagittal, counting rejected centers as uncovered. These describe
this pre-augmentation sample, not training-wide coverage or model accuracy.

The [measurement record](../../outputs/analysis/fusion-coverage-20260913/results.json)
preserves the effective config, split/label provenance, input and source hashes,
and quantiles. Its sibling `inputs.json`, `nearest_anchors.parquet` and `source/`
preserve selected paths, geometry hashes, per-query distances and executable
sources. Training membership SHA-256 is
`31422731caf633a55b8d7bf2406dfa1fe1f791e78a16482c14ca5fe2089af749`.
Reproduction requires the original analysis source bundle and configuration.
The coverage source snapshot (`f076a09:docs/history/regional-sources/measure_fusion_coverage.py.txt`)
is historical; its 18-mm radius has no role in the current spatial-token model.
The [regional usage results](../validation/results/2026-09-13-per-plane-fusion-usage-diagnosis.md)
remain historical evidence, with repeatability and deployment cost unresolved.
