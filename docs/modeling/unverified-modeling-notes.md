# Unverified modeling notes

Evidence: current repository artifacts through 2026-09-11 and unverified
working notes; see [provenance](../research/provenance.md).

The most concrete baseline lead is a fast ReXNet backbone applied to
three-slice windows, with multiple-instance learning (MIL) at study level.
The full architecture is unavailable.

The separate [public-source investigation](../research/public-research.md#geometry-sampling-encoders-and-losses)
adds controlled and anecdotal positive/negative results, including small-versus-
large encoders, slice geometry, losses and medical pretraining. Its
[notebook comparison](../research/public-research.md#public-code-worth-inspecting) identifies
inspectable DINO and CoAtNet implementations. Those are additional baselines,
not implementations of the unverified model described below.

## Unverified model structure

- The setup was described as MIL with one representation per window rather
  than one per series. The exact pooling/attention mechanism, metadata inputs,
  and output heads are unknown. The available detail does not establish
  whether the implementation contains a transformer.
- Separate metadata features may have been tried early and later dropped.
  This does not establish whether metadata was
  absent from the classifier, or change its role in geometry preprocessing.
- Three neighboring slices are packed into the input channels. Five-slice
  inputs reportedly helped some models, without a quantified result.
- Repeating one slice across three channels tested the same or worse than
  using neighboring slices. Reducing the input convolution to one channel was
  debated without a demonstrated result.
- Input normalization appeared to depend on the backbone, mostly using
  ImageNet normalization.
- `rexnetr200` was recommended as a fast backbone, with about one hour per
  fold at 384 pixels in the recorded setup. `rexnet300` and later
  `rexnetr300` were also named. Confirm the exact library identifiers and
  pretrained checkpoints rather than treating these spellings as interchangeable.
- An unnamed stronger model took roughly 6–8 hours to train. Improvements
  found with a small model reportedly transferred to that larger model.

## Window sampling

Early experiments trained on a changing random subset of about half the slices.
Training on all slices showed no gain; using all slices at inference initially
gave a small lift.

The more specific 2026-09-08 description was:

- Training: 24 windows, each containing three slices.
- Validation: 94 windows were stated explicitly.
- A later explanation used 96 windows and said about 70% of studies had fewer
  than that; increasing the count reportedly reduced the score.
- Outer slices were described as frequently uninformative.

The 94/96 discrepancy, cap scope (study versus series), overlap between windows,
and exact inference policy remain unresolved. The later capped description
qualifies the earlier claim that inference uses all images. Twenty-four windows
also need not mean 72 distinct slices if windows overlap.

## Augmentation and complementary views

- Adding an unspecified augmentation set produced a reported local improvement
  large enough for a single-fold model to beat an earlier seven-model ensemble.
  Public confirmation was explicitly still pending the next day.
- Horizontal flipping improved local results despite concerns about
  side-dependent targets. The label/orientation policy and exact transform
  configuration were not recorded, so this is an ablation lead, not an
  established label-preserving default.
- Whole-series dropout was described on 2026-09-09 as the best overall
  improvement found, forcing learning from other planes. No dropout probability
  or score delta was given.
- Different models selected important windows from different planes. Sagittal
  views were said to dominate, with rough estimates varying from 60% to 70%;
  coronal contribution was once estimated at 25%. The attribution method and
  target breakdown are absent. This does not establish that other planes can
  be discarded without loss.

## Loss and ensemble diversity

Focal loss improved some target-level diagnostics but worsened others, and
standalone models were described as slightly worse overall. Mixing loss
functions was proposed to improve ensemble diversity; no isolated ensemble
gain was reported. See [validation](../validation/target-level-diagnostics.md)
for the AUC/F1 distinction and ensembling uncertainty.

Architectural diversity, decorrelating small models, and distilling ensembles
were recurring directions. Earlier notes also suggested that older, weaker
models could hurt an ensemble. None of the projected final scores are measured
results. The unverified runtime/score pairs are in
[inference and efficiency](../inference/runtime-evidence-and-leads.md#unverified-submission-results).

## Expensive attempt and exploratory ideas

A denoising model took about 24 hours to train, followed by roughly 24 hours
for an autoencoder. Classification using the compressed representation was
slightly worse than using slices. The attempt did not encode across the slice
axis; its intended advantage was faster subsequent iteration through a smaller
cache. Claims about near-lossless compression were aspirations, not results.

Other ideas included exploiting imperfectly aligned retakes for denoising,
3D reconstruction, and model-assisted annotations or segmentation. No
controlled competition results are recorded for those directions.
