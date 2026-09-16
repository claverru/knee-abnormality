# Loss approaches for mean per-label ROC-AUC

**Per-label reduction:** $\mathcal L = \frac{1}{C}\sum_{c=1}^{C}\mathcal L_c$. For pairwise objectives, average pairs within each label before averaging labels; do not pool cross-label pairs.

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
