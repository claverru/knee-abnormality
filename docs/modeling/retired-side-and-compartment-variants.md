# Retired side and compartment variants

## Knee-side embedding

The embedding remains retired. The audited side resolver was restored on 2026-09-21
for the adopted [anatomical attention](../../autoresearch/anatomical-attention.md);
it now supplies a geometric bias rather than a left/right token embedding.

**Retired on 2026-09-14.** Research approach 1 added a zero-initialized learned
left/right vector to every token, used a fixed zero vector for unknown side,
swapped sides under reflection and applied study-level side dropout of 0.10.
Sides came from the audited [metadata policy](../metadata/knee-side-policy-audit.md).
The run `outputs/training/knee-side-embedding/` scored 0.892410 fold-0 macro AUC, a
[tie](../validation/results/2026-09-14-knee-side-embedding-result.md) with the champion's 0.892360.
Swapping or suppressing side on its checkpoint left predictions unchanged.

The side resolver, loader field, embedding, `model.side_dropout`, identity entries,
run side manifests and diagnostic side interventions were removed, restoring the
`ea19772` champion sources. [Archived sources](../history/knee-side-sources/README.md)
preserve the integration patch, resolver, policy audit and plan. The run's
source archive, side manifests and analyses remain local; its checkpoints, which current
code rejected, were deleted on 2026-09-16.

## Meniscus classifier coupling

**Historical experiment; training implementation removed on 2026-09-14.**
Candidate 1 from the
[transfer research](../research/medial-lateral-transfer.md#candidate-1-weak-classifier-coupling)
kept the twelve classifier vectors, biases and attention scorers independent.
Its training objective was:

```text
loss = mean_observed_BCE
       + 0.1 * mean_features((w_medial_meniscus - w_lateral_meniscus)²)
```

The mean was over the 256 classifier features. The penalty directly updated only
the two meniscus weight rows; it did not penalize biases or OA rows. Changed
supervised gradients could still affect the shared representation during training.
Target indices were resolved from the saved names, requiring exactly one medial
and one lateral meniscus. The experiment preserved model construction, random
initialization order, augmentation, forward inference and validation metrics.

The archived configuration used `loss.meniscus_coupling_weight: 0.1`, the first
predefined weak setting; zero was the explicit unregularized control. The loss
field is absent from the current schema. With the independent linear-weight
initialization, expected initial mean squared distance is `2/(3×256)`, giving
an expected penalty of approximately **0.0002604** at 0.1. That is a scale
reference, not a measured gradient balance or proof of weak influence under AdamW.
No GPU gradient check was run, and the coefficient is not claimed optimal.

The remaining settings matched `outputs/training/reflection-sigma40/effective_config.json`:
seed 42, Steven v4 with gold overrides, `all_5fold_v3` fold 0, ReXNet-200,
224 pixels/140 mm, reflection 0.5, sigma 40, ten epochs, batch 8, 16 workers and
2× validation images, trained from fresh pretrained initialization.

The [saved effective config](../../outputs/training/meniscus-soft-coupling/effective_config.json)
and [original source archive](../../outputs/training/meniscus-soft-coupling/source_snapshot.zip)
preserve the experiment. Reproducing or resuming its training requires that
archived source and environment; the current training code uses observed-target
BCE without the penalty. Historical run provenance and checkpoint metadata
record the loss formula, reductions, target names and loss-source SHA-256.
The experiment's resume path required matching loss settings and source identity;
export retained training metadata without adding an inference head or penalty.

Archived `train_loss` includes the penalty; CSV logs separately record epoch averages of
`train_bce` and `train_meniscus_coupling`. Before the first training step,
`fold_0/epochs/classifier_initialization.json` records weight MSE, Euclidean
distance, both norms and cosine similarity. Each epoch's existing metric JSON
contains those same values under `classifier_coupling`. For a resumed run the initial
record described the loaded weights. Validation BCE excluded the penalty,
and checkpoint selection used full-fold macro AUC.

The [completed comparison](../validation/results/2026-09-14-meniscus-soft-coupling-result.md)
scored 0.890868573 versus 0.892360168: both meniscus targets declined despite
a 20.5% reduction in classifier distance relative to the champion. It retains
the predefined family, BCE, exclusive-positive and co-occurrence/manufacturer
diagnostics. Reflection/sigma 40 remains the champion, and no additional training
is scheduled. The [historical CPU analysis](../history/meniscus-coupling-sources/README.md)
ran against saved artifacts without the removed training loss; it can no longer be rerun
because the run's checkpoints were deleted on 2026-09-16.

## Historical disease sharing

The current model has twelve independent classifier vectors and twelve
independent attention scorers. The completed alternatives were:

| Experiment | Final classifier weights | Final attention readout |
|---|---|---|
| 1A | One shared meniscus vector; all other targets independent | Twelve independent scorers |
| 1B | Separate shared meniscus and medial/lateral OA vectors; other targets independent | Twelve independent scorers |
| 1C | Same classifier grouping as 1B | Two disease vectors plus one signed compartment vector for the four paired targets; eight independent scorers |

All twelve classifier biases remain independent. PF OA is outside the OA pair.
Each target normalizes its own attention over valid study tokens. Reflection
preserves medial/lateral target names. 1C uses 128-dimensional disease and
compartment factors, initialized uniformly at ±1/sqrt(2×128); their projections
use autocast and their sums use float32. Constant attention biases cancel in
softmax and are omitted for those factors. Official target order is restored by
name-derived indices.

Their architecture identities are respectively
`spatial_tokens_shared_meniscus_classifier_v1`,
`spatial_tokens_shared_meniscus_oa_classifiers_v1` and
`spatial_tokens_disease_compartment_attention_v1`.
[Historical sources](../history/sharing-sources/README.md),
[validation and intervention results](../history/sharing/INDEX.md), and the
[transfer investigation](../research/medial-lateral-transfer.md) preserve the
implementation rationale and findings. Reproduce each checkpoint with its own
saved source/configuration; sharing is not an option in the current model.
