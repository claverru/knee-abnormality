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

## Spatial importance map

A leave-one-out window-ablation field over the ensemble, exported as a Plotly page outside this
repository and extracted to `outputs/analysis/spatial-importance-20260918/`. All 12 targets share
one 1,092-voxel grid at 9 mm spacing, spanning L [-68, 58], P [-76, 59] and S [-76, 77] mm about
the study centre; the value is positive LOO utility in within-study SD units, density corrected.
Support averages 135 studies per voxel but ranges 25-761, highest at the centre.

- **Read the cross-target contrast, not the per-target field.** Half the absolute fields are
  explained by radius alone (quadratic fit R^2: Contusion 0.80, ACL 0.77, medial OA 0.60, MCL
  0.57), which is the sampler envelope rather than anatomy. Standardizing each voxel across the 12
  targets removes that shared envelope.
- **Single-voxel peaks are noise.** 3.3% of the 13,104 voxel-target cells exceed contrast z = 2,
  about what the standardization yields by construction, and the field is spatially smoothed.
  Hemisphere means, not maxima, are the readable unit.
- **Hemisphere contrasts that survive** (mean contrast z beyond +/-18 mm, superior-inferior /
  anterior-posterior): PF OA +0.43 / **+1.85**, effusion **+1.23** / +0.95, synovitis +0.16 /
  +1.13, Baker's **-0.90** / **-1.23**, medial OA -1.07 / -0.83, lateral OA -1.05 / -0.56, MCL
  +0.29 / -1.32. Patella anterior, effusion anterosuperior, Baker's cyst posteroinferior and the
  tibiofemoral OA pair inferior are anatomically right, from a measurement that never saw anatomy.
- **The medial-lateral axis is unreadable.** The export centres and scales L but never mirrors by
  knee side, so left and right knees sit in one pooled frame in which medial and lateral reverse.
  Fold 0 is 412 left, 450 right and 42 unknown, so a true medial-lateral contrast survives pooling
  at |450-412|/862 = 4.4%. Measured L contrast is 0.00-0.36 in absolute value for every target,
  consistent with that cancellation. The signs do line up (medial meniscus +0.36, lateral meniscus
  -0.32, Baker's +0.23, medial OA +0.09, lateral OA -0.10), but at this size they are not evidence.
  Re-exporting with left knees mirrored carries a sharp prediction: a real contrast grows about
  20-fold, noise stays the same size and moves elsewhere.
- **It gives the depth profiles a direction.** The profiles place effusion's axial evidence 30-40 mm
  off centre at a 0.28 selection probability without saying which way; the map makes it the
  strongest superior contrast in the table.
- **No per-label pull can move the shared sampler.** The contrast is zero-sum by construction:
  standardizing each voxel across the 12 targets forces the mean pull on every axis to 0.000, so
  effusion superior is cancelled by the OA pair inferior, and PF OA and synovitis anterior by
  Baker's and MCL posterior. The quantity one shared sampler can act on is total utility over
  targets, and that is already centred, with a utility-weighted centroid of +1.4 mm on the axial
  through-plane axis, -3.3 mm coronal and -0.9 mm sagittal against a 9 mm grid, flat within
  +/-30 mm and falling only outside it. This closes the centre-offset follow-up without a run.
- **The sagittal axis has a central trough, and it is the only one.** Total utility over all 12
  targets by 9 mm slab, flanks (27-50 mm) minus centre (within 9 mm): sagittal **+0.126**, coronal
  -0.634, axial -1.099. The sagittal profile is near-symmetric, peaking at -31.5 mm (3.758) and
  +31.5 mm (3.725) either side of a 3.406 centre. That is the signature of left and right knees
  superimposed in the export's unmirrored frame, and the per-target split matches: the compartment
  labels carry the trough (lateral meniscus +0.194, medial meniscus +0.064, Baker's +0.057, lateral
  OA +0.041, medial OA +0.021) while the midline ones prefer the centre (PF OA -0.082, effusion
  -0.082, contusion -0.061, MCL -0.055, ACL -0.038). Support falls off centre on all three axes
  identically, so the trough is not an exposure artifact.
- **Sigma is a per-plane quantity, and the map reproduces the sigma history.** Expected utility per
  sampled slice under a truncated normal falls monotonically with sigma on every axis, matching the
  measured ordering in which 30 tied 40 and 60, 80 and 120 lost. The swing across sigma 20-120 is
  **0.092 sagittal, 0.310 coronal, 0.677 axial**: the axial draw is about seven times as sensitive
  to width as the sagittal one. A single shared sigma is therefore set by the axial penalty, which
  is why global moves came out near-neutral.
- **Limits:** the generating code and the per-study utilities are not in this repository, so only
  the extracted field can be re-derived; Fracture has the largest field spread (sd 0.15) on the
  sparsest labels and should not be read; and the profiles' synovitis peak at 40-50 mm superior
  does not reproduce as a superior contrast here (+0.16), which is unresolved.

Measured 2026-09-18. Per-type centre offsets are
[rejected](../backlog/small-changes.md#per-type-sampler-centre-offset-rejected) on the map above,
with no GPU spent. Per-plane sigma survives it and gains a mechanism, queued as a
[small change](../backlog/small-changes.md#per-plane-sampler-sigma).

The whole map is observational, measured under the current sampler, so rarely-visited depths carry
few ablated windows and their utility is understated. That bias points toward the centre on every
axis, so read the differences between axes, which share the bias, rather than the absolute optimum,
which does not survive it.
