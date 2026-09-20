# Per-label loss specialization

Research completed 2026-09-20. **Compare a small bank of independent classifiers on shared
frozen features, select a loss for each label, then evaluate the assembled recipe with trainable
features.** Learned mixtures and changes of loss during training are promising extensions.
Implementation and training remain unstarted; the
[Big bets entry](../backlog/big-bets.md#per-label-loss-selection-and-learned-mixtures) owns the
experiment sequence. The [objective survey](auc_loss_approaches.md) owns the broader catalog.

## Trusted motivation

The user accepts the contact's ACL/MCL results as trusted evidence. The task is to exploit the
principle independently of their implementation; reproducing that implementation is not a
prerequisite. These are externally measured results, distinct from our own experiment records.

| Target | Result supplied by the user | Implication |
|---|---|---|
| ACL | Ranking variant 0.967446, about +0.0080 over its parent; repeats 0.9661 and 0.9671 | Include ranking in the first candidate set |
| ACL ensemble | Earlier ranking/hinge ensemble 0.966776 | Different objectives may also contribute complementary predictions |
| MCL | Focal variant 0.959760, about +0.0087; pure ranking about 0.9497, below its parent | Preserve focal and pointwise supervision when exploring ranking |

A common evaluation metric does not require a common training loss. All twelve labels need
good positive/negative ordering, but their prevalence, separability, supervision uncertainty and
model errors differ. Losses allocate learning effort differently, and the best allocation can
also change with the representation and training stage. These mechanisms explain why searching
per label is useful; they do not prescribe the winner from prevalence alone.

BCE can already learn an AUC-optimal ordering at its unrestricted population optimum under
correct binary supervision. Practical gains from other objectives concern finite data, imperfect
labels, restricted models and finite optimization. See the
[objective survey](auc_loss_approaches.md#what-can-improve-ranking-2026-09-19).

## Primary research and its relevance

| Source | Finding | Application here |
|---|---|---|
| [Focal Loss, ICCV 2017](https://arxiv.org/abs/1708.02002) | Downweights easy examples in dense detection | Different allocation of learning effort; it neither creates positives nor automatically rejects noisy labels |
| [Gao and Zhou, IJCAI 2015](https://arxiv.org/abs/1208.0645) | Establishes AUC consistency for pairwise logistic and exponential losses, while ordinary hinge is inconsistent under their setting | Use a well-defined ranking surrogate; the different ranking losses are not interchangeable |
| [Robust Deep AUC Maximization, ICCV 2021](https://arxiv.org/abs/2012.03173) | Reports gains on several medical image tasks; Appendix J uses CE pretraining followed by AUC optimization | Supports specialization after feature learning; its min-max AUC loss differs from pairwise logistic |
| [Compositional AUC training, ICLR 2022 implementation](https://docs.libauc.org/examples/compauc.html) | Alternates CE/AUC updates with CompositionalAUCLoss and PDSCA | Training stage matters as well as loss identity; this method is not simply a weighted sum |
| [AutoLoss, KDD 2021](https://arxiv.org/html/2106.06713) | Searches candidate losses with a validation-updated controller in recommendation; equal mixtures can underperform one loss | Close precedent for learned selection; its per-example controller is more complex than our proposed per-label coefficients |
| [AutoBalance, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/191f8f858acda435ae0daf994e2a72c2-Abstract.html) | Learns class/group-specific loss parameters against a separate validation objective | Supports a small parameterized family, warmup and constrained search; adapt its multiclass accuracy/fairness setting to multilabel AUC |
| [Wu et al., EMNLP 2019](https://aclanthology.org/D19-1444/) | Learns label-specific CE weights and prediction thresholds for multilabel text tasks | Related specialization; threshold and F1 gains are not AUC evidence |
| [GradNorm, ICML 2018](https://proceedings.mlr.press/v80/chen18a.html) | Balances task gradients and learning rates | Balancing contributions between labels does not select the best loss family within a label |

These sources provide working precedents and mechanisms, not a forecast of our knee score.
The proposal below is a synthesis for this repository. The earlier blanket loss ranking in the
objective survey should not override the trusted per-label results or our stronger current control.

## Fit to our model

Code inspected on 2026-09-20, working tree based on
`dc3f410fe53c0d33995e8582ad7f6be3cdc9c69d`, with unrelated local changes present:

- [Training](../../src/knee_abnormality/training/lightning.py) uses masked soft-target BCE.
  [Class balancing](../modeling/supervision.md#fixed-class-balanced-bce) is enabled in
  [the recipe](../../configs/training.yaml): fixed training-fold weights multiply whole BCE
  cells according to `y >= 0.5`, while targets remain soft. Use this measured baseline as the
  main control, rather than ordinary BCE.
- [The model](../../src/knee_abnormality/modeling/model.py) produces pooled features with shape
  `[studies, 12, 256]`. Each label has its own classifier weight row and bias. Extra independent
  classifiers can reuse the pooled vectors; a loss bank is not implemented yet.
- `encode_batch()` exposes projected token features; `pool_encoded()` produces pooled vectors
  internally. Caching pooled vectors freezes attention as well as the encoder. Alternative
  attention training requires tokens or recomputation.
- Three classifiers per label require `12 * 3 * (256 + 1) = 9,252` readout parameters. One
  float32 pooled-feature view for all 4,407 studies is about 52 MiB before metadata. MRI feature
  extraction is the expensive part; extra sampling/augmentation views increase that cost.
- With frozen features, selected classifier rows can replace the current twelve rows, preserving
  inference cost. Cached scalar logits cannot train a new 256-input classifier, and monotone
  output rescaling alone cannot improve a label's AUC.

### Ranking needs usable study pairs

The configured microbatch is eight studies, with gradient accumulation one. Under uniform
sampling without replacement, the chance of no positive/negative pair is
`[C(Nnegative, 8) + C(Npositive, 8)] / C(Ntotal, 8)`.
Calculated from the saved
[balanced-BCE fold-0 counts](../../outputs/training/balanced-bce/fold_0/class_balance.json):

| Label | Positives / 3,503 training studies | Batches without a usable binary pair |
|---|---:|---:|
| ACL | 734 | 15.2% |
| MCL | 538 | 26.3% |
| Fracture | 252 | 55.0% |

These are analytical expectations, not measured training frequencies. Class weights do not
change batch membership. More slices of one study do not add independent studies, and ordinary
gradient accumulation does not create cross-microbatch ranking pairs.

Cached features allow large batches or sampled pairs across the training feature table, with
both sides scored by current classifier parameters. This avoids image-memory pressure and stale
logits. Average pairs within each label before averaging labels; mask missing targets, exclude
self-pairs and define the zero-pair case. End-to-end ranking needs an explicit pair-coverage
design; balancing batches for one label does not automatically cover all twelve.

## Candidate losses and extensions

Start with current balanced soft BCE, balanced focal and pairwise logistic ranking for every
label. Match classifier capacity, initialization policy, feature views and training budget; give
each family the same small learning-rate search budget. Compare AUC, not raw loss values.
Loss scale, focal exponent and ranking temperature change updates even at the same nominal LR.

A proposed soft-target focal definition is the expectation of binary focal loss:

`FL(y,p) = -y*(1-p)^gamma*log(p) - (1-y)*p^gamma*log(1-p)`.

Apply the control's whole-cell class weight. At `gamma = 0` this is current balanced soft BCE;
modest positive gamma values are candidates. Differentiate through the focal factors. This is
our proposed definition, not a claim that all focal implementations handle soft targets this way.
Do not introduce another positive-class weight simultaneously.

For ranking, initially consider
`mean softplus((score_negative - score_positive) / temperature)` within each label, with binary
pair membership using `>= 0.5`. This follows the evaluated ordering but discards soft-target
uncertainty in that term. Retain soft BCE as a control and explicitly record the changed target
treatment. Many pairs sharing one positive do not create many independent positive observations.

Extensions within this investigation, not adopted recipe choices:

1. **Fixed BCE-plus-ranking mixtures:** retain broad soft supervision while adding ranking
   pressure. Compare with both constituent objectives. Fix component scales from training-only
   reference batches before interpreting coefficients.
2. **Loss schedules:** learn features with the current recipe, then specialize during a short
   continuation. Compare with an equally long BCE continuation. This simpler proposed schedule
   is distinct from compositional AUC training.
3. **Learned mixtures:** use a small set of coefficients and validation feedback, first on frozen
   features where virtual updates and large ranking batches are inexpensive.
4. **Noise-aware ranking if diagnostics justify it:** investigate confidence-weighted pairs or
   probabilistic pair targets as explicitly different objectives. Inferred probabilities are not
   necessarily calibrated. Preserve ordinary-pair coverage instead of selecting only the hardest
   pairs, which can amplify label errors. Focal also can retain high weight on mislabeled cases
   that remain difficult; it is not automatically robust to noisy positives.

## Learning mixture weights

For each label `c`, use `w[c,:] = softmax(a[c,:])` and
`L = mean_c sum_k w[c,k] * scaled_loss[c,k]`. Three candidate losses need 36 coefficients,
with 24 independent degrees of freedom. No per-image controller is needed initially.

Optimizing those coefficients against the same training loss favours the numerically smallest
component. At fixed predictions the derivative with respect to a mixture logit is proportional
to `w[k] * (loss[k] - mixture_loss)`. Enforcing a sum of one or normalizing scales does not
turn that into evidence of better generalization.

Use a fixed outer objective independent of the learned coefficients:

1. Make a virtual model/classifier update on training examples using the current mixture.
2. Evaluate the updated predictor on separate tuning studies with a fixed, per-label-averaged
   ranking surrogate. Retain exact AUC for selection and reporting.
3. Differentiate through the virtual update to adjust the mixture, then perform the actual
   training update. Detaching the virtual update removes the required signal.

For a single SGD-like step, this signal approximately measures alignment between a candidate
training gradient and the tuning-objective gradient: does that teaching signal improve other
studies? This proposed simplification is motivated by
[validation-gradient reweighting](https://proceedings.mlr.press/v80/ren18a.html) and AutoBalance,
not an exact reproduction. AdamW needs its actual update or an acknowledged approximation.

Proposed ways to stabilize the small-data search: warm up the predictor before adapting weights,
update coefficients less often than model parameters, and shrink per-label coefficients toward
a common prior unless tuning evidence supports differences. Freeze reference component scales;
log gradient contributions alongside weights. Keep the outer ranking surrogate's temperature
fixed so that the controller cannot improve its objective merely by changing that temperature.

With frozen independent classifiers, 36 candidate fits can select one loss per label without
training all `3^12` combinations. Trainable shared features couple the labels, so evaluate the
assembled recipe jointly. A mixture's largest coefficient does not establish which loss would
have won if trained alone throughout. One adaptive run finds a candidate, not a guaranteed optimum.

## Experiment sequence and evaluation

| Stage | Comparison | Decision |
|---|---|---|
| Feature/classifier screen | One extraction per fixed view; BCE, focal and ranking classifiers for all labels, repeated classifier seeds | Which objectives help on the current representation? |
| Assemble | Stable per-label winners against the original model and refitted-BCE controls | Does selection improve AUC at unchanged inference cost? |
| Joint continuation | Selected losses versus matched current-loss continuation | Does specialization survive representation learning? |
| Mixtures | Fixed hybrids versus a small bilevel controller at matched measured compute | Does adaptation add enough beyond fixed choices? |

Start with deterministic feature views and repeat promising comparisons with another view or
parent checkpoint. A negative frozen-readout result does not rule out gains through attention
or encoder learning. If complementary predictions justify ensembling, use the
[ensemble guidance](../validation/ensemble-interpretation.md) to handle different score scales.

Use report-group-preserving inner splits within training data for head selection and mixture
adaptation. A clean tuning split must also be excluded from supervised feature learning;
otherwise label the screen development-only. Fold 0 is already a development fold. Evaluate
locked choices on an unused fold or nested outer split before claiming an independent gain.
Do not use the tiny gold subset to train a controller; follow the
[validation protocol](../validation/project-validation-protocol.md) and
[local/public evidence](../validation/local-to-public-score-relationship.md).

Keep the candidate set small, favour stable AUC differences across seeds/splits, and retain the
current loss when evidence is unresolved. Report every label and macro AUC, paired uncertainty
over studies or report groups where useful, pair coverage, extraction time, classifier/controller time,
memory and total compute. Preserve revision, parent checkpoint, split/label hashes, feature-view
identity, seeds, exact formulas, component scales and weight trajectories.
A +0.008 gain on one label contributes +0.000667 to the twelve-label mean only if the others stay
fixed; check that in the assembled model. Training remains user-launched.

## Scope of this review

Reviewed primary papers, author-maintained documentation, current loss/readout code and saved
training counts. No models were trained and no knee-score gain was estimated. OpenReview blocked
the compositional paper's page; its method was checked through the authors' LibAUC documentation.
Full min-max AUC optimizers, large per-example controllers, threshold tuning and a broad expansion
of the candidate-loss catalog are outside the initial experiment.
