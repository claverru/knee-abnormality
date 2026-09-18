# Peripheral coverage of the Gaussian sampler

Where each label's evidence sits relative to the sampler's reach, complementing the
[realized coverage fractions](../data-preparation/physical-sampling/sampling-analysis-coverage-and-exposure.md).

Triplets are placed along each plane's axis by a truncated Gaussian, `data.sampling_sigma_mm: 40`,
so coverage falls with distance from the study centre. Several labels keep most of their evidence
out there. Measured read-only from the fold-0 token cache and the depth profiles; scripts and
tables in `outputs/analysis/peripheral-coverage-20260918/`.

- **The sampler thins with depth.** Selected share ÷ candidate share beyond 40 mm is 0.45–0.59 for
  every type, and beyond 50 mm it falls to 0.36 for coronal fluid and 0.14–0.16 for the sagittal
  types. Selection probability drops from about 0.40 at the centre to 0.19 at 40 mm and beyond.
- **14–31% of a series' slices are never sampled** at the default counts, worst for axial T1/PD
  (mean 31%, p90 63%).
- **Share of a label's depth-weighted evidence sitting beyond ±20 mm:** Baker's cyst in coronal
  fluid 80%, lateral meniscus in sagittal fluid 74%, synovitis in axial fluid 67%, effusion in
  axial fluid 65%, medial meniscus in sagittal fluid 59%, MCL in axial fluid 56%. Synovitis peaks
  at 40–50 mm superior, where the selection probability is half the central value; effusion at
  30–40 mm, at 0.28. The medial meniscus in coronal fluid sits close enough that sigma 40 covers
  it.
- **Not a fix that follows for free:** lifting axial fluid's density at 40–50 mm to the central
  level implies a sigma of roughly 65–80 for that type, while global sigma 60, 80 and 120 were all
  measured worse and 30 tied
  ([overnight 2026-09-15](../validation/results/2026-09-15-overnight-results.md)). Per-type sigma and
  per-type centre offsets have never been tried, and both can be scored on the caches without the
  GPU.
- **Limits:** the separation metric behind the evidence shares is small-sample and noisy in tail
  bins, fracture is too sparse to read at all, and geometry cannot recover in-plane tissue
  clipping — the reference point is defined near each series' image centre, so the pixel-based
  figures in [input resolution](../research/input-resolution.md) remain the only source for the 140 mm
  crop (sagittal tissue exceeds it in 34% of series, almost all posterior).

Measured 2026-09-18. The per-type sigma and per-type centre-offset follow-ups are queued in
[backlog candidates](../backlog/overnight.md#candidates-for-later-queues).
