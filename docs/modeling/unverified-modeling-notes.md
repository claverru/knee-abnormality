# Unverified modeling notes, checked against our results

Working notes about another participant's model, recorded through 2026-09-11 without code,
configs or scores ([provenance](../research/provenance.md)). Each claim below is set against what
this project later measured; claims nothing here could test were dropped. Controlled public
results are in the [public-source investigation](../research/competition-and-prior-art/public-research.md#geometry-sampling-encoders-and-losses).

## Unverified model structure

| Claim | Our evidence | Verdict |
|---|---|---|
| Multiple-instance learning at study level over three-slice windows, one representation per window | The project's model was attention MIL over spatial tokens of every window from the first baseline on ([current model](current-model.md), [model lineage](../history/model-lineage.md)) | Consistent; the exact pooling of the original is unknown |
| Repeating one slice on three channels tested the same or worse than neighbouring slices | For RadImageNet weights the repeated middle slice was better (one participant: 0.8542 against 0.8339 out of fold), and every final family reads the repeated middle slice ([RadImageNet ResNets](backbones/resnet-radimagenet-backbones.md#why)) | Contradicted for our recipe |
| `rexnetr200` as a fast backbone; `rexnet300` and `rexnetr300` named | ReXNet-200 was our backbone until 2026-09-29; `rexnetr_200` lost its first run (−0.0053) with an over-large rate; ReXNet was retired on 2026-10-04 ([backbone swaps](backbones/backbone-swaps.md#same-cost-backbone-candidates)) | Useful start, not the final choice |
| Input normalization mostly ImageNet | ImageNet statistics for ConvNeXt and NextViT; the ResNet-50 uses its checkpoint's own statistics (0.3162 / 0.3213), adopted at +0.000581 ([adoption history](current-model.md#adoption-history)) | Partly |

## Window sampling

The notes described 24 training windows of three slices and 94 or 96 validation windows, with outer
slices often uninformative and more windows reportedly lowering the score. Our sampler drew 26
training slices per study (a study-total floor adopted on 2026-09-18, +0.0033 inside the control
spread), with a relative Gaussian sampler width (`data.sampling_sigma_fraction` 0.867), and read ×4 at
inference; 36 training slices gave +0.0018 (weak) on the ResNet-18 and 35 distinct slices scored
0.959 against 0.960 on the ResNet-50 (S146, S143), so the count stayed at 26
([triplet inputs](triplet-inputs-and-sampling.md),
[implemented sampler](../data-preparation/physical-sampling/implemented-sampler.md)).

## Augmentation

- **Horizontal flipping helps:** confirmed in the form of study-level reflection with side-aware
  targets, +0.001012 locally and 0.931 public on 2026-09-13, kept at probability 0.5 to the end
  ([study-level knee reflection](schedule-and-augmentation.md#study-level-knee-reflection)).
- **Whole-series dropout as the best improvement:** contradicted for our recipe. Removing series
  dropout gained +0.0023 in the `radimagenet-rn18` campaign and series dropout was deleted from the
  code on 2026-10-14 ([adoption history](current-model.md#adoption-history)).
- **Sagittal views dominate (60–70%):** not what we measured. Removing one series type cost most
  for fluid-sensitive series and the axial and coronal planes, and single-plane checkpoints lost
  0.03–0.07 macro AUC ([plane/contrast importance](../validation/results/pooling-and-attention/2026-09-17-plane-contrast-importance.md)).

## Loss and ensemble diversity

Focal loss was reported to help some targets and hurt others, and mixed losses were proposed for
ensemble diversity. Locally a loss ensemble gained only a little over seed diversity
([loss ensemble simulation](../validation/results/labels-and-losses/2026-09-21-loss-ensemble-simulation.md)).
Architectural diversity did pay at the end, but only for selected families: ConvNeXt, ResNet-50
and NextViT together beat any subset, while adding MambaVision, EfficientViT or FocalNet moved the
ensemble down ([final solution](../final-solution.md)).
