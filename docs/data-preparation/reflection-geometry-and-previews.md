# Reflection geometry and previews

## Reflection geometry and eligibility

Preparation retains a separate acquisition reflection result and compact rejection
reason. Every indexed native pose must be trustworthy, positions must have patient
provenance, and existing series orientation, center-line and frame reliability
rules must pass. For each distinct full native orientation, the dominant absolute
(column, row, cross-product normal) axes must be `(X,Y,Z)` for axial,
`(X,Z,Y)` for coronal and `(Y,Z,X)` for sagittal. Each expected component must
exceed every other component by more than the existing 0.001 orientation
tolerance. Either sign is accepted; ambiguous axes and plane disagreements reject
reflection. Eligibility reuses indexed metadata without additional DICOM reads.

Collation requires every supplied acquisition to pass, excluding absent acquisitions
and padding, before series dropout. One rejected acquisition preserves the entire
study. A rejected study center alone does not reject reflection: reliable local
acquisitions still reverse pixels/channels and sagittal 1D positions while their
preview anchors keep invalid geometry. Reflection never promotes unavailable geometry. Loader images remain
uint8; all transformation arithmetic in normal training runs on the GPU.

With `S = diag(-1,1,1)` in patient LPS, final-image anchor `q`, existing half-pixel
inverse affine `U`, and original middle-channel pixel-to-LPS map `M`, the preview's anchor
point is `S @ M @ F @ U @ q` for a reflected study; the model has computed no patient points
since the [attention prior's retirement](../history/model-lineage.md#retired-attention-prior). Applying `S` to absolute points leaves
relative offsets identical to the former centered form. `F` reverses columns around `W-1` only for reflected axial/coronal images.
This preserves full acquisition tilt without resampling onto orthogonal planes.
Source maps, valid rectangles and selected paths stay immutable. Sagittal source
channel association becomes `[2,1,0]`; the middle source stays fixed. Validity uses
`F @ U @ q` against all associated original rectangles, avoiding double reflection.
2D anchors stay in the augmented image frame, and float32 geometry stays outside autocast. The DICOM convention is documented in
[patient coordinates](https://dicom.nema.org/medical/DICOM/current/output/chtml/part03/sect_C.7.6.2.html).

The normal transforms obey `n' = S n` for axial/coronal and `n' = -S n` for
sagittal, explaining the preserved versus negated 1D position even under obliquity.
Pure reflection is an involution: column/channel reversal, sagittal sign change
and the spatial reflection each square to identity. Affine composition
is `F @ U`, not `U @ F`. This source/algebra review does not claim stochastic
augmentation invertibility, anatomical registration or model-logit invariance.

## Reflection preview evidence

The viewer defaults to pure forced reflection for one synchronized study, still
respecting eligibility. It shows source channel paths, before/after signed
positions and patient-coordinate anchors centered on the preview's own study
reference on a 3×3 final-image grid, visual validity, independent geometry validity,
source laterality where recorded,
obliquity and compact rejection reasons. Uncheck **Reflection only** to inspect
combined affine/intensity/noise transforms with forced or sampled study reflection.
The overview PNG shows pure reflection. Source laterality is source metadata;
DICOM files and reports are never rewritten. CPU execution remains supported.

Each sampled variant seeds one CPU generator, draws reflection gates per study,
then continues that generator for acquisition augmentation parameters. The gates
consume separate random values while remaining reproducible. Previews archived
before this correction restarted the generator and correlated sampled reflection
and intensity gates; regenerate them to inspect sampled combinations.

`--split validation` selects fold-0 validation images. `--offset`, `--max-studies`
and `--study-uid` bound the selection; `--sampling training --epoch N` previews
seeded sampling without training. `--geometry-headers` points to an existing header
audit for optional laterality display. Missing categories are reported explicitly.
`--profile-cuda true` adds a bounded augmentation-only CUDA profile on the same
prepared workload with zero and forced-eligible reflection, fixed supplied affine
parameters and paired noise generators. It records warmup/iteration counts,
hardware, shapes, seeds, incremental peak allocated/reserved memory, source/config
identity and input hashes in `manifest.json`. It does not run a backbone or optimizer.

```bash
rtk proxy uv run --locked python -m scripts.tools.preview_augmentations configs/training.yaml --variants 1 --profile-cuda true --output-dir outputs/previews/reflection-new
rtk proxy uv run --locked python -m scripts.tools.preview_augmentations configs/training.yaml --split validation --offset 604 --max-studies 8 --variants 1 --output-dir outputs/previews/reflection-validation-new
```

The [example-test preview](../../outputs/previews/reflection-test-20260913/preview.html)
and its [manifest](../../outputs/previews/reflection-test-20260913/manifest.json)
cover three eligible studies, 15 acquisitions and 138 triplets, all three planes,
four acquisitions tilted over 10 degrees and missing types in all three studies.
On the RTX 3090, seed 42, five warmups and 30 iterations, zero reflection took
6.904 ms and forced reflection 7.731 ms per augmentation pass. Incremental peak
allocated memory was 242.33 versus 320.00 MiB; reserved memory was 246 versus
320 MiB. These are bounded augmentation costs, not end-to-end training throughput.
The manifest and archived source/config snapshot own the exact measurement identity.

The [left-source preview](../../outputs/previews/reflection-left-20260913/preview.html)
adds one study with eight acquisitions, 64 triplets and a padded crop. The
[validation preview](../../outputs/previews/reflection-validation-20260913/preview.html)
adds eight studies with 53 acquisitions and 452 triplets, recorded right-source
laterality, 20 acquisitions tilted over 10 degrees, one padded crop and five studies
with absent types. All nine validation studies were eligible with accepted centers.
These previews do not establish full-dataset eligibility or an accuracy gain.

The [rejected-center preview](../../outputs/previews/reflection-rejected-center-20260913/preview.html)
adds one eligible left-source study with `excessive_rms`, six acquisitions and
54 triplets. Its local acquisitions reflect while every displayed 3D anchor
remains unavailable. The [bounded metadata screen](../../outputs/analysis/reflection-screen-20260913/report.json)
covered the 29 validation studies present in the saved 128-study header audit:
two centers were rejected and no acquisition failed reflection eligibility.
No real ineligible acquisition was found in this bounded selection; that preview
category remains unavailable. Overview images and selected padded/tilted inputs
were inspected, and generated viewer JavaScript passed syntax checking. Ruff,
Python compilation and the catalog consistency check passed; no tests were run.
