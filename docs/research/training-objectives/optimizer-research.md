# Optimizer research

Research and proposals written on 2026-09-16; selection metric the unweighted mean of per-label
study-level AUROC. Part of it was executed on 2026-09-17 and 2026-09-18 (see the table below); the
rest was never run. The settings proposed were starting points, not settings shown to improve this
dataset, and evidence for a method on other datasets does not establish a knee-MRI AUROC gain.

## Recommendation and what became of it

AdamW stayed the baseline. The report proposed a lower backbone learning rate with the head rate
unchanged, then a shorter second-moment horizon (`beta2=0.99`), then EMA of the weights as the first
additional feature, and afterwards AdamP, SAM + AdamW or PESG with an AUC-margin loss.

| Proposal | Setting | Result |
|---|---|---|
| Lower backbone LR | `backbone_lr` 1e-4, head 6e-4 (ratio 3:1 to 6:1) | ReXNet-200: `12-backbone-lr-1e-4` −0.0030 ([2026-09-18 overnight](../../validation/results/campaigns-and-queues/2026-09-18-overnight-results.md)) |
| Slower head | `head_lr` 3e-4 | `08-slower-head` −0.0017 [−0.0047, +0.0011] ([2026-09-17 overnight](../../validation/results/campaigns-and-queues/2026-09-17-overnight-results.md)) |
| Shorter second-moment horizon | `beta2` 0.99 | `06-beta2-0.99` −0.0025 [−0.0056, +0.0004] (same queue) |
| EMA | decay 0.999 | `03-ema-0.999` +0.0001 [−0.0011, +0.0014]; public 0.929 against the champion's 0.936; later removed from the code (see [schedule and augmentation](../../modeling/schedule-and-augmentation.md)) |
| EMA at decay 0.9999 on B0 | 0.9999 over 4,380 steps | void: the averaged copy was degenerate ([B0 queues](../../validation/results/campaigns-and-queues/2026-09-18-b0-hyperparameter-queues.md)) |
| AdamP, SAM + AdamW, layer-wise decay, PESG with AUC-margin loss, CoAtNet learning-rate ranges | below | no run found in the validation results (text search only) |

## Repository baseline at the time

Read from the checkout on 2026-09-16, during the crop-selection investigation. Sources: `configs/baseline.yaml` and `training.py` of that day (the config was later renamed `configs/resnet50.yaml` and then replaced by per-recipe configs; the optimizer code is now `src/knee_abnormality/training/lightning.py`). The scout was removed later that day, when the code returned to the [full-slice model](../../modeling/current-model.md). The optimizer settings below are unchanged in the `00-baseline` recipe.

| Component | Inspected setting |
|---|---|
| Backbone | Pretrained `rexnet_200.nav_in1k`; scout backbone `mobilenetv3_small_100.lamb_in1k` |
| Input | Three-channel adjacent slice triplets |
| Prediction | Study-level multilabel classification |
| Label source | `steven_v4` |
| Validation | `all_5fold_v3`, fold `0` |
| Loss | Masked `binary_cross_entropy_with_logits` |
| Training budget | 10 epochs |
| Batch size | 8 studies, one device, accumulation of 1 |
| Precision | `bf16-mixed` |
| Gradient clipping | `gradient_clip_val: 1.0` |
| Scheduler | Optimizer-step-based warmup followed by cosine decay |

Existing optimizer configuration:

```yaml
optimizer:
  backbone_lr: 0.0002
  head_lr: 0.0006
  weight_decay: 0.01
  beta1: 0.9
  beta2: 0.999
  epsilon: 0.00000001
  warmup_epochs: 1
  minimum_lr_ratio: 0.01
```

The inspected `build_optimizer()` already:

- Assigns `backbone.*` and `scout.backbone.*` parameters to the backbone learning rate and all other trainable parameters to the head learning rate.
- Excludes parameters with `ndim < 2`, names ending in `cls_token`, and names containing `type_embedding` from weight decay.
- Excludes frozen parameters and creates an AdamW optimizer over the resulting groups.

The scheduler starts at 10% of each group's configured learning rate, warms up over one epoch, and then applies cosine decay toward a 1% learning-rate floor. This schedule was kept in the comparisons.

The loss averages BCE over observed entries using `target_observed`. That behavior was kept in the optimizer-only comparisons.

## AdamW configuration experiments proposed

- **Lower backbone LR** (`backbone_lr` 1e-4, other fields unchanged) changes the head/backbone ratio
  from 3:1 to 6:1. The hypothesis was gentler updates preserving pretrained features; the competing
  one was insufficient backbone adaptation.
- **`beta2=0.99`** on the better LR configuration: AdamW's squared-gradient memory is about
  `1/(1-beta2)` updates (about 1,000 at 0.999, 100 at 0.99), a heuristic time scale. Shorter horizons
  react faster and smooth less.
- **Secondary tuning** (not a Cartesian sweep): backbone LR 1e-4 or 2e-4, head LR 6e-4 or 1e-3,
  `beta2` 0.999 or 0.99, weight decay 0.001, 0.01 or 0.05, other settings fixed and no compensating
  longer training.
- **CoAtNet branch:** ReXNet's LR is not an assumed optimum there; proposed ranges were backbone LR
  5e-5 to 1e-4 and head LR 3e-4 to 6e-4, with optional layer-wise decay factors 0.8 or 0.9 (1.0 as
  control). `timm` exposes layer decay and custom grouping, which the local optimizer builder did not
  implement. These ranges were hypotheses.

## EMA

EMA keeps a smoothed copy `ema = decay * ema + (1 - decay) * live`. PyTorch's `AveragedModel`
defaults to SWA averaging, not EMA, and needs an explicit BatchNorm policy (average buffers or
recompute statistics). The proposal was to initialize the copy when warmup ends, update after each
completed optimizer step and compare decays 0.99 and 0.999 (per update, not per microbatch or
epoch), validating ordinary and EMA weights separately, with the copy outside the optimizer, kept in
FP32, and exported as one ordinary inference state.

## SAM + AdamW

SAM minimizes loss in a neighbourhood of the parameters and needs two gradient evaluations per update
(about double compute). Proposed `rho` 0.02 against 0.05, `adaptive` false, on the better AdamW
base, prioritized if training loss improved while validation AUROC fell. The two-pass update had to be
implemented explicitly, with unperturbed parameters restored before the base update, no double update
of BatchNorm statistics, and scheduler and EMA advanced once per completed base update.

## AdamP

AdamP removes the radial update component for scale-invariant weights. `timm.optim.AdamP` exposes
`delta` 0.1 and `wd_ratio` 0.1; it keeps the existing groups, needs weight decay retuned, and has no
SAM-style second gradient evaluation. It was proposed as the cheaper alternative to SAM.

## AUC-specific loss and optimizer track

The optimizer follows the objective: masked BCE and differentiable pairwise ranking losses stay on
AdamW; LibAUC `MultiLabelAUCMLoss` pairs with PESG and `CompositionalAUCLoss` with PDSCA (multilabel
adaptation to be verified). PESG handles primal and dual variables of the AUC-margin objective and is
not a drop-in that keeps BCE unchanged.

- **Small batches.** With 5% prevalence and eight studies per batch, `(1 - 0.05)^8 = 0.6634`: about
  66% of batches hold no positive for that label (illustrative, not measured prevalence). Extra slices
  from the same studies are not independent study-level examples, and gradient accumulation does not
  create cross-microbatch ranking pairs.
- **Missing and soft targets.** LibAUC identifies positives and negatives by exact comparison with 1
  and 0, so missing entries and targets strictly between 0 and 1 need an explicit treatment; absent
  positive, absent negative and unobserved labels need checking, and the per-label reduction must not
  weight by pair counts.
- **Sequence.** Establish the stronger BCE/AdamW baseline, then fine-tune from that checkpoint with
  the AUC objective; loss, optimizer and sampling changes form one combined experiment, not an
  optimizer-only improvement.

## Evaluation rule

Screening used the same split, seed, initialization, sampling, augmentations and validation inputs.
Each label's AUROC is computed over the whole held-out fold and then averaged without weights (not
minibatch AUROCs, not micro-AUC), with the policy for undefined AUROC stated. Promising changes were
to be confirmed on more folds and, for close calls, more seeds.

## Sources

References checked on 2026-09-16: the repository training configuration and implementation of that day;
[PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html);
[PyTorch AveragedModel](https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html);
[timm optimizers](https://huggingface.co/docs/timm/reference/optimizers);
[SAM](https://arxiv.org/html/2010.01412v3); [AdamP](https://arxiv.org/abs/2006.08217);
[LibAUC optimizers](https://docs.libauc.org/api/libauc.optimizers.html) and
[losses](https://docs.libauc.org/_modules/libauc/losses/auc.html).
