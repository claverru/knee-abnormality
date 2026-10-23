# Dense-bag pooling on the ResNet-50 (2026-10-12)

**One attention temperature of 0.9 at ×4 stays the setting.** On three folds of the Refined
ResNet-50, no per-study rule beats one τ at ×4, quality falls past ×4, holding each series at its
×1 share of the attention recovers most of that fall, and a temperature per label is real for one
trained model but does not carry to another.

All scores are macro AUC on the validation fold of `all_5fold_scanner_v2` with reflection TTA,
unless a line says otherwise. Intervals are the central 80% of a paired report-group bootstrap
(2,000 draws). One model per fold, so each fold is one run's evidence.

## Models

`rn50-aligned-mixing-folds-0-2` (`outputs/training/rn50-aligned-mixing-folds-0-2`):
`configs/refined-resnet50.yaml` (removed) as on main at `0bd598f` (aligned contrast mixing, usual labels,
teacher share 0), seed 1110484336, about 265 s per epoch, best epoch 10 of 10 on each fold.

| Fold | Studies | ×4, τ 1, no TTA | ×1 |
|---|---:|---:|---:|
| 0 | 912 | 0.940906 | 0.933184 |
| 1 | 874 | 0.933392 | 0.925323 |
| 2 | 875 | 0.937000 | 0.928460 |

## Method: a token cache

The model pools with a softmax over one score per token and label, times one evidence value per
token and label (`plane_readout`). Caching those two numbers per token, with validity, view and
series type, reproduces the model's logits (within 0.0002 in full precision) and lets any pooling
rule be scored offline in seconds.

- **Cache:** fold 0 at ×1, ×4, ×5, ×6 and ×8, plain and mirrored, in
  `outputs/analysis/pooling-token-cache/fold0` (23 GB; about 5 minutes of GPU per multiplier).
  Folds 1 and 2 have predictions over a wide τ range at ×4 only
  (`outputs/analysis/rn50-aligned-mixing-wide-tau`) and τ 0.70 to 1.10 at ×4, ×5, ×6 and ×8
  (`outputs/analysis/rn50-aligned-mixing-sweep`, not scored).
- **Scripts:** run from the session scratchpad and not kept.
- **Half precision:** the model computes the attention scores in bf16, which moves a logit by up
  to 0.008 against full precision; the cache holds full-precision scores.
- **Sweep pitfall:** `scripts/adhoc/sweep_child_temperatures.py` re-pools one encoding at several
  τ, but the model memoises the position prior per encoding since the anatomy-map adoption, so
  every τ after the first reuses the first τ's prior. Clear `memo_position_prior` before each τ.
- **What τ reaches:** `apply_attention_temperature` scales the main scorer, the view label bias,
  the position prior and the routing temperature. It does not scale the scorer of the soft-tissue
  read (`soft_attention`). The encoding also uses the main scorer (the frame from the meniscus
  attention spots), so a baked τ changes the encoding too; the cache encodes at τ 1.

## One temperature per multiplier (fold 0)

| Multiplier | Best τ | Macro | Share of each series' slices used (median) |
|---|---:|---:|---:|
| ×1 | 1.1 | 0.933599 | |
| ×4 | 0.9 | 0.941272 | 0.77 |
| ×6 | 0.8 | 0.940530 | 0.99 |
| ×8 | 0.8 | 0.940176 | 1.00 |

- Past ×4 the bag is the whole study, and quality falls at every τ: ×6 −0.0007 and ×8 −0.0011
  against ×4.
- τ 1.0 to 0.9 at ×4 is worth +0.0003; 0.80 to 0.95 are within 0.00015.
- From ×4 to ×8 at each label's own best τ: lateral OA −0.0085, MCL −0.0020, ACL −0.0019,
  fracture −0.0017; synovitis +0.0038, lateral meniscus +0.0011.
- τ 0.9 is also the best single value at ×4 on fold 1 (0.933793) and fold 2 (0.937689).

## Rules per study (fold 0)

Against the best single τ at the same multiplier:

| Rule | ×4 | ×6 | ×8 |
|---|---:|---:|---:|
| Each series at its ×1 share of the attention, τ 0.9 | +0.00006 | +0.00029 | +0.00084 [+0.00026, +0.00146] |
| Each view type at its ×1 share, τ 0.85 to 0.90 | +0.00016 | +0.00020 | +0.00069 [+0.00019, +0.00122] |
| A separate τ (0.7 to 0.9) for the soft-tissue read | −0.00002 to −0.00006 | | |
| Slices weighted by the training sampling density within a series | −0.00021 | −0.00003 | −0.00014 |
| τ per study from its token count, ln N at ×1 / ln N now | −0.00017 | −0.00031 | −0.00034 |
| Only the top ×1-sized set of tokens | −0.00037 | −0.00058 | −0.00058 |
| τ per study and label matching its ×1 attention entropy | −0.0023 | −0.0029 | −0.0029 |

- **The share rule** adds `log(slices at ×1 / slices now)` to the scores of a series' tokens. With
  it the spread across ×4, ×6 and ×8 at τ 0.9 falls from 0.0011 to 0.0005, and ×4 stays the best
  multiplier (0.941333, 0.940814, 0.941017).
- **Study size does not move τ at ×4:** 80% of studies have 104 to 148 triplets (median 124,
  minimum 73) and the best τ per size third is 0.90, 0.90 and 0.95; a τ per third adds +0.00014
  in-sample.
- **What stays lost past ×4 with the share rule:** PF OA (−0.0021 at ×6) and lateral OA (−0.0013).

## A temperature per label

**The best τ per label belongs to the trained model, not to the label.** Best τ at ×4 in the range
0.4 to 3.0:

| Label | Fold 0 | Fold 1 | Fold 2 |
|---|---:|---:|---:|
| ACL | 0.6 | 0.5 | 1.1 |
| MCL | 1.1 | 0.6 | 1.5 |
| Medial Meniscus | 1.1 | 1.25 | 0.6 |
| Lateral Meniscus | 1.0 | 0.8 | 0.9 |
| Medial OA | 0.8 | 0.8 | 0.9 |
| Lateral OA | 0.5 | 1.25 | 0.9 |
| PF OA | 0.6 | 1.0 | 0.8 |
| Effusion | 2.0 | 1.1 | 1.1 |
| Synovitis | 1.0 | 2.0 | 1.5 |
| Baker's | 1.25 | 2.0 | 1.25 |
| Contusion | 0.9 | 0.9 | 1.25 |
| Fracture | 1.0 | 0.6 | 0.9 |

- **A free τ per label on fold 0:** +0.0012 in-sample at ×4, and −0.0007 chosen on half the
  studies and applied to the other half (positive in 17% of 25 splits); −0.0010, −0.0006 and
  −0.0005 at ×1, ×6 and ×8.
- **Three values on fold 0** (one for PF OA, lateral OA and ACL, one for effusion, one for the
  other eight; groups chosen on the whole fold): chosen on half and applied to the other half,
  +0.00076 at ×4, +0.00062 at ×6 and +0.00079 at ×8 over the best single τ (positive in 95 to 97%
  of 40 splits); picked values 0.6, 1.5 and 0.9.
- **Those values on the three folds, against τ 0.9 at ×4:**

| Fold | τ 0.9 | 0.6 / 1.5 / 0.9 | Δ |
|---|---:|---:|---|
| 0 | 0.941218 | 0.942269 | +0.00105 [+0.00054, +0.00160] |
| 1 | 0.933793 | 0.933150 | −0.00064 [−0.00112, −0.00020] |
| 2 | 0.937689 | 0.936995 | −0.00069 [−0.00121, −0.00017] |

  On the four moved labels: fold 0 PF OA +0.0046, lateral OA +0.0036, effusion +0.0035, ACL
  +0.0009; fold 1 effusion +0.0002, ACL +0.0002, lateral OA −0.0031, PF OA −0.0050; fold 2 ACL
  −0.0016, lateral OA −0.0017, effusion −0.0021, PF OA −0.0029.
- **With the series share on fold 0** (in-sample, against ×4 at τ 0.9): +0.00111 at ×4, +0.00048
  at ×6, +0.00067 at ×8.

## S147

Fold 0 alone at ×4 with reflection TTA and the three values per label (0.6 for ACL, lateral OA and
PF OA, 1.5 for effusion, 0.9 for the rest), pooled three times from one encoding at τ 1: **0.959
public**. The bundle scores 0.942268 on fold 0 against 0.941272 at τ 0.9. The same weights were not
submitted at τ 0.9, so the public score has no like-for-like reference; S142 (fold 1, the recipe
without aligned mixing, τ 0.9) scored 0.959 and S144 (fold 1, aligned mixing at 0.25 teacher share)
0.960. Its Kaggle description reads `tau1`: the submission job ran from main, which does not know
the option.

The option (`export --label-attention-temperature`, `runtime.label_attention_temperature`, pooled
per distinct value in `inference/engine.py`) is in the worktree `knee-abnormality-label-temperature`,
branch `label-attention-temperature`, uncommitted.

## Open

Not pursued: each series at its ×1 share as an inference option, and a sharpness per label fitted per
model. The pooling head at the inference bag size is in
[untried ideas](../../../backlog/untried-ideas.md#pooling-and-the-attention-head).
