# Training objectives, weighting and sampling for mean per-label ROC-AUC

The search space includes the loss, the weight of each study–target contribution, which studies
reach the model, and which examples share a batch. The current recipe retains uniform study
exposure and uses [fixed class-balanced soft BCE](../../modeling/supervision.md#fixed-class-balanced-bce).
Neither choice is imposed by the metric. Calibration is not an endpoint. For different loss
families within each label, shared-feature comparisons and learned mixtures, see
[per-label loss specialization](per-label-losses.md).

**Default per-label reduction:** $\mathcal L = \frac{1}{C}\sum_{c=1}^{C}\mathcal L_c$.
For pairwise objectives, average pairs within each label before averaging labels; do not pool
cross-label pairs. Unequal training weights are a separate candidate below; evaluation always
gives each label equal weight.

## What can improve ranking (2026-09-19)

The current loader shuffles studies without replacement and averages soft-target BCE over all
observed cells ([implementation](../../../src/knee_abnormality/training/lightning.py)). With complete
labels, each study and each target has equal exposure; positives and negatives retain their
observed frequencies, rather than having a 1:1 class ratio.

Per-label ROC-AUC depends on within-label ordering. A strictly increasing transformation of a
fixed label's scores preserves that ordering, so output bias shifts or positive output-temperature
scaling cannot improve its AUC, apart from numerical ties. This is different from changing an
internal attention temperature, which can change ordering. AUC accepts scores without requiring
calibrated probabilities ([metric reference](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)).

BCE is still a valid ranking surrogate: with correct binary supervision and an unrestricted
population optimum it learns the conditional positive probability, whose ordering is optimal for
AUC. For a constant positive weight $w>0$, direct minimization gives
$p^*(x)=w\eta(x)/(1-\eta(x)+w\eta(x))$, where $\eta(x)=P(Y=1\mid x)$.
This is a monotone transform of $\eta$, so weighting does not improve that ideal ranking merely
by shifting probabilities. The practical hypothesis is that finite training, shared model capacity,
regularization and imperfect supervision make a different allocation of gradients or image
exposure learn a better ordering. Worse calibration is acceptable; better held-out AUC is the
criterion. Our soft report-derived targets need not equal the true conditional probabilities.

## Weighting and sampling design space

**Open candidates, not selected implementations.** These can retain BCE, accompany a ranking
loss, or change over training. The earlier loss ranking below does not rank this expanded space.

| Mechanism | What changes | Main distinction |
|---|---|---|
| Positive/negative term weights | Relative contribution of the two terms inside each label's BCE (§9) | Directly changes fitting of soft targets; no extra study exposure |
| Study–target cell weights or masks | Weight or retention probability of individual cells, possibly balancing thresholded positive/negative groups | Preserves other labels on the study; weighting a soft cell scales both BCE terms, unlike `pos_weight` |
| Whole-target weights | $\mathcal L=\sum_c\alpha_c\mathcal L_c/\sum_c\alpha_c$ | Allocates shared-model gradients among labels; equal evaluation weights do not require equal training weights |
| Study resampling | Frequency of study visits, using label frequencies, combinations or other training information | Changes image exposure and all labels on each selected study |
| Batch composition | Which studies are grouped together, potentially preserving total study visits | Can supply within-label positive–negative comparisons without necessarily changing epoch frequencies |
| Adaptive weighting or sampling | Any of these allocations changes with historical training signals | Requires a useful feedback signal, beyond identifying the largest raw loss |

For binary labels on one task, class resampling and class-weighted loss can match expected
gradients up to normalization at fixed model parameters. They need not produce the same training
trajectory or batch statistics. In our multilabel setting, oversampling a study for one finding
also repeats its other eleven labels; co-occurring findings change the marginal distributions.
With soft labels, selecting studies by a threshold is also different from weighting their positive
BCE terms. The [distribution-balanced approach](#10-distribution-balanced-loss) addresses related
co-occurrence effects.

One illustrative sampler chooses four distinct labels, one positive study per label, and four
random studies for a batch of eight. It is an example, not an agreed design or a preferred ratio.
Alternatives include choosing positive and negative pools, preserving epoch exposure while
rearranging batches, or weighting cells without changing study visits. A positive-directed batch
does not guarantee negatives. [LibAUC's samplers](https://docs.libauc.org/api/libauc.sampler.html)
provide related binary and multilabel positive/negative batch construction, not this exact mixture.

Nonuniform sampling without correction deliberately changes the empirical objective. Inverse
sampling-probability weights can instead preserve its expected gradient while changing estimator
variance; those are different hypotheses. [Importance-sampling research](https://arxiv.org/abs/1803.00942)
uses gradient information to reduce variance, and does not establish that high-loss resampling
improves this project's AUC.

### Adaptive allocation

Candidate feedback includes positive/negative exposure, excess loss separately within those
groups, relative learning progress, gradient contribution, and within-label ranking errors from a
larger historical pool. [GradNorm](https://proceedings.mlr.press/v80/chen18a.html) is a related
adaptive task-weighting method using gradient magnitudes and relative training rates; it is not a
study sampler or evidence of a knee-AUC gain.

Raw soft-target BCE is not directly comparable as reducible error:
$\mathrm{BCE}(y,p)=H(y)+D_{\mathrm{KL}}(\mathrm{Bern}(y)\Vert\mathrm{Bern}(p))$.
Even a perfect prediction of $y=0.5$ has loss $\log 2$ and zero logit gradient. Subtracting $H(y)$
removes this floor from a diagnostic score, but does not change BCE's training gradients or remove
label noise. Prevalence, sampling-induced distribution changes and stale observations also affect
comparisons. The largest remaining error need not be the best use of additional computation.

Any adaptive design needs smoothed statistics, continued coverage and bounded allocation so one
noisy target or study cannot dominate. Define the source of feedback: selected training batches
are biased toward previous choices; validation-driven adaptation would make those labels part of
training decisions and require separate evaluation. Keep per-target weighting distinct from
per-study hard-example selection when choosing a concrete experiment.

### Evidence and experiment ownership

Fixed class-balanced BCE reached 0.895282 locally, +0.008796 over the closest same-seed
historical reference, with preprocessing-source differences limiting causal attribution. Other
new weighting and sampling candidates remain unmeasured. The asymmetric-loss
result below applies to that experiment; it does not reject the wider family. The full-dataset
label frequencies describe exposure, not which intervention will improve AUC.

Static weighting and study/batch sampling belong to
[small changes](../../backlog/untried-ideas.md#supervision-and-labels), including the existing
[cell-masking proposal](../../backlog/untried-ideas.md#pooling-and-the-attention-head).
Adaptive allocation ran in the [noisy-labels campaign](../../validation/results/labels-and-losses/2026-09-23-noisy-labels-campaign.md), none adopted.
The first proposed comparison is
[fixed class-balanced BCE](../../validation/results/labels-and-losses/2026-09-19-balanced-bce-result.md), completed on
2026-09-19; [supervision](../../modeling/supervision.md#fixed-class-balanced-bce)
owns its exact behavior. Compare one mechanism at a time against a fresh control, preserving
validation, soft supervision unless explicitly varied, optimizer-step budget and schedule.
Record sampling/weight definitions, seeds, per-label exposure, unique/repeated studies, runtime
and per-label/macro AUC. An epoch under resampling need not be one visit to every study, and equal
steps need not cost equal wall time. Choose any pool or frequency estimates from the training fold.

### Weighting strength evidence (2026-09-21)

- Local: plain BCE lost ~0.009–0.012 to full balancing; soft-mass balancing (−0.0092) and
  biproportional weighting (−0.0044) both compressed rare-label weights but each also changed
  another factor, so neither isolates strength. The isolated exponent runs (2026-09-21) scored 0.945 on the
  public leaderboard at k=0.5, 0.943 unweighted and 0.940 at k=1, the opposite order to fold-0
  validation ([result](../../validation/results/labels-and-losses/2026-09-21-class-balance-exponent.md)).
- Focal loss α sweep, γ=0 ([Table 1a](https://arxiv.org/abs/1708.02002)): best positive weight
  ≈0.75 (≈3:1) at ~1:1000 imbalance; α=0.999 far worse.
- Kang et al. ([Table 5](https://arxiv.org/abs/1910.09217)): sampling ∝ n^q, square root 46.8 and
  progressive 47.2 beat class-balanced 45.1.
- LVIS repeat-factor sampling ([paper](https://arxiv.org/abs/1908.03195)): square-root with
  threshold best (23.2 vs 21.0 none); full class-aware sampling 18.7.
- Seesaw ([Table 5](https://arxiv.org/abs/2008.10032)): exponent p=0.8 best (25.7) vs 1.0 (25.5).
- Class-balanced loss ([paper](https://arxiv.org/abs/1901.05555)): on CIFAR-100-LT β 0.99–0.999
  helped, 0.9999 (≈inverse frequency) hurt.
- Medical multi-label: Baltruschat 2019
  ([study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6476887/)) found no significant difference from
  weighting on ChestX-ray14; CheXFusion ([paper](https://arxiv.org/html/2308.03968)) bounded weights
  e^(1−r)/e^r moved macro AUC −0.002; no knee MRI ablation found.
- Theory: at the population optimum weighting cannot change within-label AUC; effects come from
  finite training ([Byrd & Lipton](https://arxiv.org/abs/1812.03372)) and large weights add SGD
  gradient variance ([An et al.](https://arxiv.org/abs/2009.13447)).

The exponent is implemented as `trainer.class_balance_exponent`; see
[supervision](../../modeling/supervision.md#fixed-class-balanced-bce).

## Ranking (2026-09-18)

Historical ranking of the loss candidates only, excluding the expanded weighting/sampling space
above. Estimated expected improvement on fold-0 macro AUC. These are priors from each method's mechanism
under this project's constraints, not measurements; none of these losses has been run here.
`Replaces` marks whether the arm substitutes for soft labeling rather than adding to it.

| Rk | Loss | E[d] | range | P(helps) | Replaces | Code |
|---:|---|---:|---|---:|---|---|
| 1 | SPLC | +0.0020 | -0.008/+0.010 | 50% | yes | med |
| 2 | ROC-star/buffer | +0.0015 | -0.004/+0.006 | 58% | partial | med |
| 3 | AUCM | +0.0015 | -0.005/+0.007 | 55% | yes | med |
| 4 | Hill | +0.0015 | -0.006/+0.008 | 52% | yes | low |
| 5 | MIDAM | +0.0015 | -0.004/+0.006 | 55% | no | high |
| 6 | BCE + pairwise | +0.0010 | -0.002/+0.004 | 57% | no | low |
| 7 | CompAUC / AUC-FT | +0.0008 | -0.002/+0.003 | 58% | no | low-med |
| 8 | pos_weight BCE | +0.0003 | -0.002/+0.002 | 52% | no | trivial |
| 9 | Dist-balanced | +0.0002 | -0.003/+0.003 | 50% | partial | med |
| 10 | Focal | +0.0000 | -0.003/+0.002 | 45% | no | trivial |

Rejected on merit: asymmetric loss (§7) is the one arm with a measurement against it, Two-Way
Multi-Label (§14) spends half its objective on cross-label ordering that macro AUC never scores, and
RAL (§13) only matters if an untuned pointwise loss deserves a run. Multilabel contrastive
pretraining (§16) builds its pairs from the labels, so with 58 of 4,407 studies gold it consumes the
pseudo-label noise rather than replacing it; as a training stage rather than a loss it is out of
scope for this ranking either way, and a label-free self-supervised variant belongs in
[pretraining and external data](../backbones-and-pretraining/pretraining-and-external-data.md).

## Direct AUC and ranking objectives

### 1. Pairwise logistic ranking

Penalizes negative examples scoring above positive examples **for the same label**:

$$
\mathcal L_c = \mathbb E_{+,-}\left[\operatorname{softplus}\left(\frac{s_c^- - s_c^+}{\tau}\right)\right].
$$

Here, $s_c^+$ and $s_c^-$ are positive and negative logits; $\tau>0$ controls the score-difference scale. Requires positive and negative examples in the batch or comparison pool. [Reference](https://arxiv.org/abs/1208.0645)

### 2. BCE + pairwise ranking

Retains BCE supervision and adds a ranking objective:

$$
\mathcal L_c = \mathcal L_{\mathrm{BCE},c} + \lambda\mathcal L_{\mathrm{rank},c}.
$$

The ranking term can be pairwise logistic or squared hinge. $\lambda$ controls its contribution. This is a simple hybrid objective, not compositional AUC training.

### 3. Pairwise squared-hinge ranking

Penalizes positive–negative pairs whose score difference is below a margin $m$:

$$
\mathcal L_c = \mathbb E_{+,-}\left[\max\left(0,\;m-(s_c^+-s_c^-)\right)^2\right].
$$

Pairs exceeding the margin receive zero loss. This differs from ordinary hinge and from an untruncated squared loss. [Reference](https://arxiv.org/abs/1208.0645)

### 4. AUC-margin loss — AUCM

Uses class-conditional score statistics to reduce within-positive and within-negative variance while encouraging a margin between their mean scores. Its min–max formulation avoids explicitly enumerating every positive–negative pair.

Use separate auxiliary variables per label, such as LibAUC's `MultiLabelAUCMLoss`, with the dedicated **PESG** optimizer. [Loss](https://docs.libauc.org/api/libauc.losses.html) · [Optimizer](https://docs.libauc.org/api/libauc.optimizers.html)

### 5. ROC-star

A margin-based pairwise ranking approach using sampled predictions from the previous epoch and an adaptively estimated margin. Historical predictions supply comparison examples beyond the current batch. For mean per-label AUC, keep comparisons and historical pools label-specific. [Implementation](https://github.com/iridiumblue/roc-star)

### 6. Compositional AUC training — CompAUC

Combines BCE-driven representation learning with AUC optimization through a compositional objective, schematically:

$$
\min_w\;\mathcal L_{\mathrm{AUC}}\left(w-\eta\nabla_w\mathcal L_{\mathrm{BCE}}(w)\right).
$$

LibAUC implements an alternating training scheme with **PDSCA**. This is not equivalent to simply adding BCE and AUC losses. [Reference](https://docs.libauc.org/examples/compauc.html)

## Pointwise imbalance-aware losses

### 7. Asymmetric loss — ASL

Uses separate focal exponents for positive and negative labels, usually $\gamma_->\gamma_+$, to suppress easy negatives more strongly. An optional negative probability shift, $p_m=\max(p-m,0)$, removes sufficiently easy negatives from the loss. It targets multilabel imbalance rather than directly optimizing AUC. [Reference](https://arxiv.org/abs/2009.14119)

### 8. Focal loss

Multiplies binary cross-entropy by $(1-p_t)^\gamma$, where $p_t$ is the probability assigned to the observed binary target. This reduces easy-example contributions and emphasizes difficult examples. An optional $\alpha$ term balances positives and negatives. [Reference](https://arxiv.org/abs/1708.02002)

### 9. Positive-weighted BCE

Assigns a positive-class weight $w_c$ independently for each label:

$$
\mathcal L_c=-\mathbb E\left[w_c y_c\log p_c+(1-y_c)\log(1-p_c)\right].
$$

This changes positive-versus-negative training emphasis within a label; it is different from weighting the entire label's contribution to the macro loss. Implement with `BCEWithLogitsLoss(pos_weight=...)`. [Reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)

### 10. Distribution-balanced loss

Modifies BCE using **label-co-occurrence-aware reweighting** and **negative-tolerant regularization**. It addresses long-tailed multilabel distributions without treating each label's frequency as an isolated binary-balancing problem. [Reference](https://arxiv.org/abs/2007.09654)

## Missing-label and robustness approaches

### 11. Hill loss

Uses a hill-shaped weighting of negative examples: semi-hard negatives receive greater emphasis, while very easy negatives and suspected false negatives receive less. Designed for incomplete multilabel annotations where some recorded negatives are actually unannotated positives. [Reference](https://arxiv.org/abs/2112.07368)

### 12. Self-paced loss correction — SPLC

Uses model predictions to identify likely missing positive labels and adjusts their loss treatment during training. Unlike fixed reweighting, this changes how suspected false negatives contribute to the objective. [Reference](https://arxiv.org/abs/2112.07368)

### 13. Robust asymmetric loss — RAL

Combines asymmetric polynomial loss terms with Hill-style regularization to address long-tailed multilabel classification. The regularization is intended to reduce sensitivity to the loss's multiple hyperparameters. [Reference](https://arxiv.org/abs/2308.05542)

## Structured and representation-learning approaches

### 14. Two-Way Multi-Label Loss

Promotes discrimination in two directions: between positive and negative labels **within an example**, and between positive and negative examples **for a label**. The second direction supports per-label ranking; the first adds a cross-label objective rather than directly matching macro AUC. [Reference](https://staff.aist.go.jp/takumi.kobayashi/publication/2023/CVPR2023.pdf)

### 15. Multi-instance deep AUC maximization — MIDAM

Optimizes AUC when each labeled example is a bag of instances, such as a study containing slices or patches. It addresses pooled bag-level predictions and stochastic instance sampling. A multilabel use case requires separate label objectives and a macro reduction. [Reference](https://arxiv.org/abs/2305.08040)

### 16. Multilabel contrastive pretraining + BCE

Learns representations with a multilabel-aware contrastive objective, then fine-tunes the classifier using BCE. This is a representation-learning strategy rather than a direct replacement for the final classification loss. [Reference](https://papers.miccai.org/miccai-2025/paper/1834_paper.pdf)
