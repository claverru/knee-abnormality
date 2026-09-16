# Optimizer research

**Date:** 2026-09-16  
**Selection metric:** unweighted mean of per-label AUROC, evaluated at study level.  
**Status:** research conclusions and proposed experiments; no optimizer comparisons have been run.

## 1. Recommendation

Keep **AdamW** as the baseline. First test a lower backbone learning rate while retaining the head learning rate, then test a shorter second-moment averaging horizon. Add **exponential moving average (EMA)** of model weights as the first additional training feature.

After those experiments, consider **AdamP** for a one-gradient-evaluation alternative on ReXNet, or **SAM + AdamW** when additional training computation is acceptable. Treat **PESG + AUC-margin loss** as a separate loss-and-optimizer experiment, not an optimizer-only substitution.

The exact hyperparameters below are proposed starting points, not settings demonstrated to improve this dataset. Evidence for a method on other datasets does not establish a knee-MRI AUROC gain.

## 2. Repository baseline used for the research

Checked against the local checkout on 2026-09-16, during the crop-selection investigation. Sources: the training config and optimizer code, since renamed [`configs/training.yaml`][repo-config] and [`src/knee_abnormality/training/lightning.py`][repo-training] (then `configs/baseline.yaml` and `training.py`). The scout was removed later that day, when the code returned to the [full-slice model](../modeling/current-model.md). The optimizer settings below are unchanged in the `00-baseline` recipe.

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

The scheduler starts at 10% of each group's configured learning rate, warms up over one epoch, and then applies cosine decay toward a 1% learning-rate floor. Preserve this schedule initially.

The loss averages BCE over observed entries using `target_observed`. Preserve that behavior during optimizer-only comparisons. Any change to masking, label weighting, loss reduction, or sampling belongs in a separate experiment.

## 3. AdamW configuration experiments

### 3.1 First experiment: reduce only the backbone learning rate

```yaml
optimizer:
  backbone_lr: 0.0001
  head_lr: 0.0006
  weight_decay: 0.01
  beta1: 0.9
  beta2: 0.999
  epsilon: 0.00000001
  warmup_epochs: 1
  minimum_lr_ratio: 0.01
```

This changes the head/backbone learning-rate ratio from **3:1 to 6:1**.

**Hypothesis:** more conservative updates may preserve useful pretrained features while the new projection, pooling, and classification components continue adapting. The competing possibility is insufficient backbone adaptation. Compare against the unchanged baseline; do not assume the lower rate wins.

This experiment uses existing configuration fields and should not require optimizer code changes.

### 3.2 Second experiment: `beta2=0.99`

Use the better learning-rate configuration from the preceding comparison and change only:

```yaml
optimizer:
  beta2: 0.99
```

AdamW maintains an exponential average of squared gradients. An approximate memory scale is `1 / (1 - beta2)`: about 1,000 optimizer updates at `0.999`, versus 100 at `0.99`. This is a heuristic time scale, not a hard window. The shorter horizon responds faster to changes in gradient scale but provides less smoothing. Small batches and rare positives do not make either value universally preferable. [AdamW documentation][adamw]

Keep `beta1=0.9` and `epsilon=1e-8` initially.

### 3.3 Secondary tuning

Run targeted comparisons around the better configuration rather than a full Cartesian sweep.

| Parameter | Proposed values | Purpose |
|---|---|---|
| Backbone learning rate | `0.0001`, `0.0002` | Conservative versus stronger backbone adaptation |
| Head learning rate | `0.0006`, `0.001` | Test whether the new components are adapting too slowly |
| Second-moment coefficient | `0.999`, `0.99` | Long versus short gradient-scale memory |
| Weight decay | `0.001`, `0.01`, `0.05` | Regularization sensitivity |

Keep batch size, clipping, warmup, schedule, epochs, loss, and data settings fixed during these comparisons. Do not compensate for a lower learning rate by simultaneously extending training.

### 3.4 CoAtNet branch

Do not transfer ReXNet's learning rate as an assumed optimum.

Proposed initial CoAtNet search:

```text
backbone_lr: 0.00005–0.0001
head_lr:     0.0003–0.0006
optimizer:   AdamW
```

After the simpler backbone/head comparison, optionally test layer-wise learning-rate decay with a factor of `0.8` or `0.9`; use `1.0` as the no-decay control. Earlier backbone groups should receive smaller rates than later groups.

`timm` exposes layer-wise decay and custom parameter grouping, but these settings are not implemented by the inspected local optimizer builder. Verify the selected backbone's group mapping, retain the separate head rate, and preserve no-decay exclusions rather than blindly replacing the existing grouping. [timm optimizer reference][timm-optim]

These CoAtNet ranges and decay factors are hypotheses, not architecture-specific empirical results from this project.

## 4. EMA: first additional training feature

EMA maintains a separate smoothed parameter copy:

```text
ema_weights = decay * ema_weights + (1 - decay) * live_weights
```

PyTorch's `AveragedModel` supports EMA through an explicit exponential averaging function. Its default averaging is SWA, not EMA. BatchNorm requires an explicit statistics policy: the documentation describes averaging buffers or recalculating activation statistics. [PyTorch averaged-model documentation][ema]

**Proposed experiment:** initialize the EMA copy from the live model when warmup ends, update once after each completed optimizer update, and compare decay `0.99` with `0.999`. Validate ordinary and EMA weights separately. These decays are per optimizer update, not per microbatch or epoch.

Implementation requirements for the local agent:

- Keep the EMA copy out of the optimizer and in evaluation mode during validation. Use a deliberate precision policy, preferably FP32 for averaging.
- Define and verify floating-point buffer and integer-counter handling. When recalibrating BatchNorm, use training data only and adapt the pass to this model's dictionary input.
- Save and restore EMA weights, buffers, update count, decay, and initialization state. Reconstruct the averaging function on resume; PyTorch does not store that function in `state_dict`.
- Export the selected EMA state as one ordinary inference model. Do not accidentally export the live weights or require both copies at inference.

EMA adds a training-time model copy and update work; exporting one selected state does not change the inference architecture. Its validation gain remains untested here.

## 5. SAM + AdamW: generalization experiment

SAM minimizes loss in a neighborhood of the current parameters. Its original experiments reported improvements on image-classification and fine-tuning benchmarks, including label-noise experiments. Standard SAM requires two gradient evaluations per update, so forward/backward computation is approximately doubled; measure actual wall-clock overhead. [SAM paper][sam]

**Proposed starting settings:**

```yaml
# Conceptual settings: not supported by the inspected YAML schema.
base_optimizer: adamw
rho: 0.02          # Compare with 0.05.
adaptive: false
```

Use the better AdamW configuration as the base. Prioritize this experiment when training loss improves while validation AUROC deteriorates. This is a diagnostic hypothesis, not proof SAM will help.

Implementation requirements:

- Implement the two-pass update explicitly; changing the optimizer constructor alone is insufficient.
- Reuse the sampled input batch and document the stochastic augmentation/dropout replay policy.
- Prevent unintended double updates to BatchNorm running statistics.
- Restore unperturbed parameters before the base optimizer update.
- Verify clipping, mixed precision, accumulation, checkpointing, and nonfinite-gradient behavior.
- Advance the scheduler and EMA once per completed base update, not once per SAM pass.

Test SAM independently before combining it with EMA. Compare both equal-epoch and equal-training-time budgets. SAM itself does not add inference layers.

## 6. AdamP: alternative for the ReXNet branch

AdamP removes the radial component of updates for identified scale-invariant weights to address effective-step-size reduction associated with weight-norm growth. The paper evaluates it across multiple benchmarks; that is motivation for a normalized CNN experiment, not evidence of a gain on this task. [AdamP paper][adamp]

Start from the better AdamW learning rates with:

```yaml
# Conceptual settings: requires optimizer-factory/schema changes.
optimizer_name: adamp
beta1: 0.9
beta2: 0.999
epsilon: 0.00000001
delta: 0.1
wd_ratio: 0.1
```

`timm.optim.AdamP` exposes these parameters. Preserve the existing parameter groups and retune weight decay instead of assuming numerical equivalence with AdamW. AdamP does not introduce SAM's second model-gradient evaluation. [timm optimizer reference][timm-optim]

Treat this as an alternative to the higher-compute SAM experiment, not a prerequisite for it.

## 7. AUC-specific loss–optimizer track

Optimizer selection must follow the objective.

| Objective | Optimizer direction |
|---|---|
| Existing masked BCE | AdamW baseline |
| Differentiable pairwise ranking loss, or BCE plus ranking | AdamW remains applicable |
| LibAUC `MultiLabelAUCMLoss` | PESG |
| LibAUC `CompositionalAUCLoss` | PDSCA, with a verified multilabel adaptation |

LibAUC documents these pairings. PESG handles primal and dual variables associated with the AUC-margin objective; it is not an AUC-enhancing drop-in replacement while keeping BCE unchanged. Do not copy AdamW's learning rate, regularization, or scheduler assumptions into PESG without retuning and checking the implementation. [LibAUC optimizers][libauc-optim]

### 7.1 Small-batch limitation

For illustration, assume independent random study sampling and a label prevalence of 5%. With eight studies per batch:

```text
P(no positive for that label) = (1 - 0.05)^8 = 0.6634
```

Approximately 66% of batches would contain no positive for that label. This is an illustrative calculation, not a measured prevalence in this dataset.

Additional slices from the same studies do not create independent study-level positive/negative examples. Ordinary gradient accumulation also does not create cross-microbatch ranking pairs when each microbatch loss is computed independently. Increasing the effective batch size for pointwise BCE is therefore not equivalent to enlarging the pairwise comparison set.

Address per-label positive/negative coverage, pair construction, and observed-label counts alongside the loss. Do not assume that balancing one label balances every label. An AUC-margin formulation is not identical to explicit pairwise loss; inspect its own absent-class behavior.

### 7.2 Missing and soft targets

LibAUC's documented AUC implementations identify positives and negatives using exact comparisons with `1` and `0`. Their standard interfaces are not drop-in equivalents to this repository's masked BCE. [LibAUC loss source][libauc-loss]

Before using them, define treatment of missing labels and any targets strictly between zero and one. Do not silently turn missing entries into negatives or discard/threshold soft targets.

Check absent-positive, absent-negative, and completely unobserved labels. For macro-AUC-oriented training, explicitly document per-label reduction and avoid accidental weighting by observed-entry or pair counts.

### 7.3 Proposed development sequence

Establish the stronger BCE/AdamW baseline first. A later experiment can initialize from that checkpoint and fine-tune with an AUC objective; LibAUC also discusses ordinary-loss pretraining before AUC fine-tuning. [LibAUC loss source][libauc-loss]

Verify score conventions, margin, sampler behavior, auxiliary-state updates, and checkpoint restoration before training. Treat changes in loss, optimizer, and sampling as a combined experiment, with separate ablations when feasible. Do not promote it as an optimizer-only improvement.

## 8. Experiment order and evaluation

Use sequential screening, not an exhaustive search. “Best AdamW” below means the configuration selected by completed local experiments; no winner is supplied by this document.

| Stage | Experiment | Comparison |
|---|---|---|
| A0 | Unchanged AdamW baseline | Control |
| A1 | Backbone LR `0.0001`; head LR unchanged | A0 |
| A2 | `beta2=0.99` | Better of A0/A1, changing only `beta2` |
| B1 | EMA decay `0.99` and `0.999` | Best AdamW, ordinary versus EMA weights |
| B2, optional | Head LR and weight-decay tuning | Targeted single-factor comparisons |
| C1, optional | AdamP | Best comparable AdamW configuration |
| C2, optional | SAM + AdamW, `rho=0.02/0.05` | Equal epochs and equal training time |
| D, separate | Matched AUC loss, optimizer, and sampling | Strongest established BCE baseline |

Use the same split, seed, initialization, study sampling, augmentations, and validation inputs for screening. Preserve the existing label source and gold/non-gold validation reporting.

Compute each label's AUROC across the complete held-out fold, then take the unweighted mean. Do not average minibatch AUROCs or substitute micro-AUC. Preserve and report the policy for labels with undefined AUROC.

Record the resolved configuration, code revision, library versions, optimizer-update count, learning-rate groups, epoch time, total training time, peak memory, and per-label metrics. Distinguish ordinary and EMA predictions/checkpoints unambiguously.

Confirm promising changes on additional folds and, for close comparisons, additional seeds. Do not infer a reliable improvement from one favorable fold or leaderboard fluctuation. Preserve the inference pipeline during optimizer comparisons.

## 9. Local-agent implementation checklist

Review these files first:

```text
configs/training.yaml
src/knee_abnormality/config.py
src/knee_abnormality/training/lightning.py
src/knee_abnormality/training/checkpoints.py
src/knee_abnormality/modeling/model.py
src/knee_abnormality/evaluation/metrics.py
src/knee_abnormality/inference/checkpoint.py
src/knee_abnormality/inference/bundle.py
```

**Immediate scope:** reconcile the baseline and create separate A0/A1/A2 experiment configurations. Do not change labels, losses, architecture, or sampling in those experiments.

**Extensions:** EMA, AdamP, SAM, and layer-wise decay require optional configuration support and implementation changes relative to the inspected training code. The conceptual YAML above is not a drop-in schema. Add configuration only for options that will actually be varied, and reject unsupported options rather than silently ignoring them.

Before full runs, verify:

| Check | Required behavior |
|---|---|
| Parameter grouping | Every intended trainable parameter appears exactly once; no EMA parameters enter the optimizer |
| Learning rates and decay | Backbone/head rates and no-decay exclusions remain correct |
| Scheduler | Warmup and decay follow completed optimizer updates, including accumulation |
| Resume | Model, optimizer, scheduler, sampling state, and optional EMA/auxiliary states restore consistently |
| SAM | Two passes produce one base update; perturbations are removed even on failure |
| Validation/export | Selected weights and reported metrics refer to the same model state |
| AUC losses | Missing/soft labels and absent classes have explicit, verified behavior |

Do not treat documentation examples as proof that APIs match the locally installed versions. Keep changes modular so each experiment has a clear comparison.

## 10. Sources

External references were checked on 2026-09-16, but API compatibility still depends on local package versions.

- [Repository baseline configuration][repo-config]
- [Repository training implementation][repo-training]
- [PyTorch: AdamW][adamw]
- [PyTorch: AveragedModel, EMA, and BatchNorm handling][ema]
- [timm: optimizer factory, parameter grouping, layer decay, and AdamP][timm-optim]
- [Foret et al.: Sharpness-Aware Minimization for Efficiently Improving Generalization][sam]
- [Heo et al.: AdamP — Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights][adamp]
- [LibAUC: optimizer–loss compatibility and optimizer APIs][libauc-optim]
- [LibAUC: AUC loss implementations and target handling][libauc-loss]

[repo-config]: ../../configs/training.yaml
[repo-training]: ../../src/knee_abnormality/training/lightning.py
[adamw]: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
[ema]: https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html
[timm-optim]: https://huggingface.co/docs/timm/reference/optimizers
[sam]: https://arxiv.org/html/2010.01412v3
[adamp]: https://arxiv.org/abs/2006.08217
[libauc-optim]: https://docs.libauc.org/api/libauc.optimizers.html
[libauc-loss]: https://docs.libauc.org/_modules/libauc/losses/auc.html
