# Physical sampling analysis

Metadata-only investigation completed on 2026-09-13 before implementing the
shared-center sampler. Acquisition choice independent of slice count preserves
the intended source exposure with little additional placement error. A Gaussian
with sigma 30 mm substantially narrows coverage relative to the current sampler;
a much broader Gaussian is needed to approach its peripheral exposure.

The production sampler was implemented on 2026-09-13 using the decisions below.
The historical analysis remains evidence for the initial width and acquisition
policy, not an accuracy result. Its prototype and old-sampler comparison are
preserved with their original source under
`outputs/analysis/p/physical-sampling-20260913/geometry-v3/source/`.

## Cohort and reproduction

The saved geometry sample contains all 26,804 headers from 721 series in 128
studies, originally selected without replacement from sorted training-study UIDs
with NumPy seed 42. Its selection and input identity are in
`outputs/analysis/g/geometry-metadata-20260912/selection.json`. The current geometry
rules accept all 721 series and the shared reference in 124 studies; four studies
fail the maximum RMS criterion. The sampling comparison therefore contains 595
study/plane/contrast groups, including 95 repeated groups with 197 acquisitions.
There are 31 groups containing an acquisition with median spacing below 2 mm and
98 groups spanning more than 120 mm. These thresholds only define descriptive
strata. Of the original 128 studies, 99 belong to the fold-0 training partition
and 29 to its held-out partition; 461 and 134 accepted groups respectively remain.

The protocol audit includes all 128 studies and therefore 99 repeated groups,
including groups from studies whose shared center was rejected. It reads one
original DICOM header per series, selecting the first sorted path and stopping
before pixel data. The geometric simulation reads only the saved header table.
Neither command decodes images, runs a model, trains, evaluates labels or runs
tests.

The protocol command remains active. Reproduce the original geometric comparison
only inside its archived source/runtime with its recorded original configuration:

```bash
uv run --locked python -m scripts.adhoc.audit_sampling_protocols --output-dir outputs/analysis/physical-sampling-new/protocols
```

The protocol command refuses an existing output directory.

The original protocol `protocols-v2` and geometry `geometry-v3` directories
already contain the completed runs. Use the matching full historical runtime
and the manifest-recorded config when executing their archived analysis source.
Each manifest pins
the inputs, analysis source and relevant dependencies. The geometry artifacts
include `summary.json`, `centers.parquet`, `slots.parquet`, `eligibility.parquet`,
`draws.parquet`, `selected.parquet`, and `sampling-comparison.png`/`.svg`, with
source snapshots and output hashes. The protocol artifacts include the emitted
field snapshot, summary and source snapshot. Original DICOM files are not hashed;
the selected metadata snapshot is hashed. Earlier directories without `-v2` or
`-v3` are preliminary runs, not the canonical result.

## Compared sampling rules

The archived comparison called its then-production capacity allocator and triangular
sampler directly on reconstructed, physically ordered records. It retains 3:1
selection weights, acquisition-midpoint weighting, pooled allocation, complete
non-overlapping triplets, and the baseline type budgets. Its seeded per-group
random-number streams differ from the live loader's study-wide sequencing; this
is a reproducible distribution comparison, not a replay of saved training inputs.

The prototype holds the current accepted center fixed. Within each supported
plane/contrast type it constructs the union of eligible middle-slice coverage,
counting overlaps once. The second through penultimate slices are candidates.
It maps each slice plane to its intersection with the corresponding fixed
patient-coordinate axis through the shared center:

```text
target_point(t) = center + t * patient_axis
slice_depth = dot(normal, ImagePositionPatient - center) / dot(normal, patient_axis)
```

The axis is superior for axial, posterior for coronal and left for sagittal
acquisitions. This handles oblique orientations explicitly. The Gaussian width
is measured along that shared axis; it is not the perpendicular distance from
the center to each differently tilted plane. Gaussian probability is normalized
once over the union of available interior-depth intervals.

Training divides the cumulative probability into K equal bands and draws one
random percentile per band. Validation uses each band's midpoint. The simulation
seed is 20260913, with 32 training draws per group. Training-mode sampling is
simulated on all sampled studies; partition flags allow separate examination
without reading targets. Validation has twice the configured triplet budget.

Two source-selection rules share the same physical targets:

- **Nearest:** jointly select the distinct available middle planes minimizing
  total squared depth error, allowing any acquisition whose coverage contains
  the target.
- **Balanced:** first select a supplier independently of its slice count.
  Training chooses uniformly among covering acquisitions. Validation accumulates
  each eligible source's fractional entitlement and assigns the source with the
  largest remaining deficit, with a fixed seeded tie order. Joint assignment
  then minimizes missing inputs, supplier changes, and squared depth error in
  that order. No supplier changes or missing inputs were needed by the balanced
  variants in this sample.

Overlapping neighboring triplets are allowed. To avoid buying the same middle
plane twice, candidate planes form complete-link geometry groups when every pair
has at most 0.1 mm of offset difference plus normal-direction deviation over a
70-mm radius. Assignment uses each such group at most once. This is an explicit
analysis tolerance, not a protocol-equivalence classifier or an established
optimal anatomical separation. Different sequence images remain alternatives
within the same geometric group. Adjacent physical planes remain eligible.
