# EMA submission

The `03-ema-0.999` run of queue `outputs/training/overnight-20260916T214520Z-t8k_kw2z` is the
restored full-slice recipe with EMA weight averaging (decay 0.999); its live weights reproduce
that queue's `00-control` exactly. Checkpoint selection chose epoch 10 of the averaged weights at
**0.893155 fold-0 macro AUC**, +0.00016 over the live weights and the highest fold-0 score of the
full-slice runs ([overnight results](../../validation/results/2026-09-17-overnight-results.md)).
It was exported on 2026-09-17 to `outputs/submission/exports/knee-ema-0999-20260917`
(checkpoint SHA-256 `b87fc2df46bc491c0f03aca40e6edca64e2e7f55805fb4777a0ee3b0c67f55da`) with the
fast runtime: 2× images, four studies per GPU, 16 total loader workers and automatic FP16
backbone chunks on T4. The bundle holds only the averaged weights; the live weights stored in the
training checkpoint are not exported.

Competition submission **56298280 completed with public AUC 0.929**, below the champion's 0.936
and `00-baseline`'s 0.934 despite the higher fold-0 score. The dataset is
`claverru/knee-ema-0999-20260917` and the notebook
`claverru/knee-ema-0999-20260917-inference`; the
[workflow record](../../../outputs/submission/records/knee-ema-0999-20260917.json) holds the
transitions. The dataset upload took 7 s, the notebook ran for 4m 01s, and scoring took 9m 04s
after submission (13m 47s from launch to score).

- Fold-0 macro AUC and public AUC disagree again: 0.893155 → 0.929 here, against 0.892360 → 0.936
  for the champion and 0.892763 → 0.934 for `00-baseline`. The three fold-0 scores lie within the
  0.0032 seed spread measured the same night, so fold 0 alone cannot rank these models.
- The public leaderboard scores part of the hidden test set; with one submission per model it
  cannot separate EMA from training-run variability either.
