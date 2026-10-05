# Pretraining-input campaign contract

## Objective

Close the gap between what we feed the Refined-RadImageNet backbone and what it was pretrained
on, in input preparation and in training augmentation, and retest the authors' mask procedure on
a clean footing. The previous campaign fixed the normalization; label-free measurements with the
frozen pretrained network then showed where the remaining gap is
([prior work](prior-work.md)): the fluid-sensitive series arrive with about two thirds of the
other series' contrast, the pretraining's own AutoAugment closes that gap, and three of our
augmentations move the input away from the pretraining statistics.

The reference is a ResNet-10t, because it has both standard and mask-trained checkpoints like
the ResNet-50, which the ResNet-18 lacks, and it is the fastest backbone. What is adopted here
is carried to the ResNet-50 by the user.

The reference and starting incumbent is `rn10t-radimagenet-normalization`
(`outputs/training/20261004-203750-rn10t-radimagenet-normalization`): the `configs/training.yaml`
recipe of this branch (Refined-RadImageNet ResNet-10t, standard weights, mean 0.3162 and std
0.3213, backbone LR 6e-5, seed 1533697213, fold 0 of `all_5fold_scanner_v2`), ×4 0.927579, ×1
0.921891 in 1,162 s. It was trained before the options below were built; they are all off by
default, so the source-hash difference is a declared exception and the new keys are declared
equivalent to their off values.

## Evaluation

The [shared evaluation and adoption rule](../../protocol.md#shared-evaluation-and-adoption)
applies unchanged against the campaign incumbent. Every result also reports ×1, the per-label
deltas and the idea's own scalars.

## Seeds

Forced, in file order ([seeds 1](seed-ideas.yaml), [seeds 2](seed-ideas-2.yaml),
[seeds 3](seed-ideas-3.yaml), [seeds 4](seed-ideas-4.yaml), [seeds 5](seed-ideas-5.yaml)). Each applies to the incumbent at
its turn, so an accepted seed carries into the later ones.

| # | Seed | Kind |
|---|---|---|
| 1 | `loader-upper-quantile-099` | configuration |
| 2 | `loader-upper-quantile-098` | configuration |
| 3 | `heated-weights` | configuration |
| 4 | `slice-upper-quantile-098` | configuration |
| 5 | `autoaugment-intensity` | configuration |
| 6 | `fluid-type-upper-quantile` | configuration |
| 7 | `fluid-brightness` | configuration |
| 8 | `no-bilateral-smoothing` | configuration |
| 9 | `no-contrast-mixing` | configuration |
| 10 | `no-gin` | configuration |
| 11 | `rotation-30` | configuration |
| 12 | `study-cutmix` | configuration |
| 13 | `union-bag-loss` | configuration |
| 14 | `input-mask-centred-one` | configuration |
| 15 | `heated-weights-mask-release` | configuration |
| 16 | `split-knee-bag-loss` | configuration |
| 17 | `split-knee-bag-average` | configuration |
| 18 | `split-knee-bag-or` | configuration |
| 19 | `front-back-bag-loss` | configuration |
| 20 | `transplant-bag-loss` | configuration |
| 21 | `learned-mask-policy` | source change |

- **Seeds 1, 2, 4, 6 and 7 attack the same contrast gap** with different tools. Each keeps the
  incumbent's other intensity settings, so an accepted one stays in place under the next.
- **`heated-weights` changes the starting weights**, with the rate and anchor strengths their
  scale implies. If it is accepted, the later seeds run on it.
- **`heated-weights-mask-release` sets the heated weights itself** if the incumbent does not
  have them, and is then reported against both the incumbent and seed 3's run.
- **Not seeds:** proportional-label CutMix across studies (the label of a sparse finding does
  not scale with the share of slices; `union-bag-loss` is its correct form), MixUp (measured: it
  moves the input away from the pretraining statistics), wider views (measured: no closer), and
  the DICOM display window (the loader does not read the window tags), and a consistency loss
  between our view and a pretraining-style augmented view (a second forward pass; left to the
  researcher as a source-change idea).

## Budget

Wall-clock range: each candidate's training time must lie between 75% and 200% of the
reference's training time (about 870–2,320 s for a clean ResNet-10t run of 1,160 s). Ten
epochs, batch size 8, the studies seen per epoch and the slices per study are fixed.

## Fixed

Split, fold, evaluation labels (`rebuild_state_v2`), class-balanced BCE and its exponent 0.5,
image size 224, the views and sampling geometry, 26 slices per study, ten epochs, batch size 8,
the evaluation and prediction multipliers, the half-knee and report-attribute terms and their
weights, the backbone architecture (`resnet10t.c3_in1k`).

## Permitted

- The twenty-one seeds.
- After the seeds, researcher ideas of two kinds (user, 2026-10-04): ideas that bring our input
  preparation or training augmentation closer to the pretraining data and recipe, or that use
  the authors' mask procedure; and combinations of two prospects from this campaign's own runs.
- Per-label rules, with every choice fixed before the run.

## Excluded

Pretraining on RadImageNet images or any external data; weight averaging; image sizes above
224, the views and field of view, more slices per study, ensembling and test-time augmentation;
fewer epochs; reflection-probability changes; the class-balance exponent; label-source changes
and gold-study weighting; multi-seed or confirmation runs; trained models in the loader;
choosing anything on fold-0 validation labels; MixUp; ideas already measured at the same value
([prior work](prior-work.md)).

## Built options

Every row is implemented in this branch, off by default, and switched in
`configs/training.yaml`. A configuration-only idea changes these values and nothing else.

| Key | Off value | Effect when set | Where | Scalars logged per epoch |
|---|---|---|---|---|
| `data.upper_quantile` | 0.995 (incumbent) | Bright-end quantile of the loader's per-series intensity scaling | loader, training and prediction | `input_fluid_mean`, `input_fluid_std`, `input_other_mean`, `input_other_std` |
| `data.fluid_upper_quantile` | null | Overrides `data.upper_quantile` for the three fluid-sensitive series types | loader, training and prediction | same four |
| `model.slice_upper_quantile` | null | Each slice is divided by its own quantile of that rank and clamped to 1, before augmentation | model, training and prediction | same four |
| `model.fluid_brightness` | 1.0 | The slices of fluid-sensitive series are multiplied by it and clamped to 1 | model, training and prediction | same four |
| `data.autoaugment_intensity.probability` | 0.0 | Per slice, with this probability, one sub-policy of torchvision's AutoAugment ImageNet policy; its rotate, shear and colour steps are skipped | model, training only | `autoaugment_intensity_slices` |
| `data.study_cutmix_augmentation.probability` | 0.0 | Per slice, with this probability, a random box (side √(1−u), u uniform) is replaced by the same box of another slice of the same study and series type | model, training only | `study_cutmix_slices`, `study_cutmix_area` |
| `trainer.union_bag_loss_weight` | 0.0 | Studies of a batch are paired at random; each pair's tokens are pooled as one bag; target `1 − (1 − a)(1 − b)`; a label counts when both are observed or an observed one is positive; class-balanced BCE times this weight is added to the loss | Lightning module, training only | `union_bag_loss`, `union_bags`, `union_bag_positive_rate` |
| `trainer.split_knee_bag_loss_weight` | 0.0 | Studies of a batch are paired at random; one keeps its tokens more than 10 mm medial of the midline, the other its tokens more than 10 mm lateral; the kept tokens are pooled as one bag; Medial Meniscus, Medial OA and MCL come from the medial donor, Lateral Meniscus and Lateral OA from the lateral donor; with `trainer.split_knee_other_labels` `none` (default) the other seven are left out, with `average` they are the mean of the two donors, with `or` their OR; class-balanced BCE times this weight is added to the loss | Lightning module, training only | `split_knee_bag_loss`, `split_knee_bags`, `split_knee_kept_token_share` |
| `trainer.front_back_bag_loss_weight` with `trainer.front_back_other_labels` | 0.0, `none` | The same chimera cut front to back at 10 mm from the knee centre: PF OA from the anterior donor, Baker's from the posterior donor, the other ten by the same three rules | Lightning module, training only | `front_back_bag_loss`, `front_back_bags`, `front_back_kept_token_share` |
| `trainer.transplant_bag_loss_weight` | 0.0 | Studies of a batch are paired as donor and host; the donor keeps the union of its top 2% tokens by attention score for each label; these and all host tokens are pooled as one bag with OR labels; class-balanced BCE times this weight is added to the loss | Lightning module, training only | `transplant_bag_loss`, `transplant_bags`, `transplant_donor_token_share` |
| `trainer.input_mask.kind` with `grid`, `kernel`, `sigma`, `low`, `epochs` | `none` | `random`: the normalized input is multiplied by a smooth random field: cells uniform in [`low`, `low` + 1] on a `grid` × `grid` lattice, resized to 224 by nearest neighbour, Gaussian blur (`kernel`, `sigma`); for the first `epochs` epochs, or all when null | Lightning module, training only | `input_mask_mean`, `input_mask_std`, `input_mask_slices`, `input_mask_expectation_mean` |
| `model.input_mask_expectation` | false | Must be true exactly when the mask kind is not `none`. Prediction multiplies the normalized input by a stored constant: `low` + 0.5 while the mask is on, 1 after it is released | model | — |
| `model.pretrained_weights` | `radimagenet` | `radimagenet_policy` (heated) and `radimagenet_policy_ft` load the mask-trained ResNet-10t checkpoints, cached and hash-pinned | model | — |

The existing augmentations are switched off by setting their `probability` to 0 and leaving
their other fields as they are.

## Activation signatures

The build logs one scalar per option that is measured during training and separates the option
switched on from the incumbent by construction. A configuration-only idea declares the row of
its option verbatim, at epoch 0 unless the row says otherwise; thresholds are not derived from
the label-free measurements of [prior work](prior-work.md), which describe unaugmented loader
images. An idea that sets two options declares the row of the option named first in its title.

| Option | Metric | Signature | Why it separates |
|---|---|---|---|
| `data.upper_quantile` q | `loader_other_bright_tail` | `>` the midpoint of the incumbent's 1 − q and the candidate's (0.99: 0.0075; 0.98: 0.0125) | 1 − the bright-end quantile read each epoch from the training dataset object that scales the series (`dataset.data.upper_quantile`); 0.005 at the unchanged recipe |
| `data.fluid_upper_quantile` q | `loader_fluid_bright_tail` | `>` the midpoint of the incumbent's fluid tail and 1 − q (0.98 over 0.995: 0.0125) | the same reading for the fluid-sensitive types, with the override applied as the loader applies it |
| `model.slice_upper_quantile` | `slice_quantile_rescaled_slices` | `> 0` | epoch count of training slices whose own quantile was below 1, so that the division changed them; logged only when the option is set |
| `model.fluid_brightness` | `fluid_brightness_slices` | `> 0` | epoch count of training slices multiplied by a gain other than 1; logged only when the option is not 1.0 |
| `model.pretrained_weights` `radimagenet_policy` or `radimagenet_policy_ft` | `pretrained_conv_rms` | `> 0.04` | root mean square of the backbone's convolution weights right after loading: 0.01576 for the standard checkpoint (logged by a run of this build), 0.05305 for the heated one |
| an existing augmentation switched off (`intensity`, `affine`, `noise`, `smoothing`, `resolution`, `gin`, `contrast_mixing`) | `<family>_off_slices` | `> 0` | epoch count of training slices encoded while that family's probability is 0; never logged while the family is on |
| `data.affine_augmentation.rotation_degrees` d above the incumbent's | `affine_rotation_max_degrees` | `>` the incumbent's `rotation_degrees` (10.0) | the largest absolute rotation applied in the epoch; it cannot exceed the configured limit, so the incumbent stays at or below 10 |
| `data.autoaugment_intensity.probability` | `autoaugment_intensity_slices` | `> 0` | epoch count of slices the policy changed |
| `data.study_cutmix_augmentation.probability` | `study_cutmix_slices` | `> 0` | epoch count of slices that received a box |
| `trainer.union_bag_loss_weight`, `trainer.split_knee_bag_loss_weight`, `trainer.front_back_bag_loss_weight`, `trainer.transplant_bag_loss_weight` | `union_bags`, `split_knee_bags`, `front_back_bags`, `transplant_bags` | `> 0` | epoch count of bags that entered the added loss |
| `trainer.input_mask.kind` `random` | `input_mask_slices` | `> 0` | epoch count of masked training slices |
| `trainer.input_mask.low` 0.5 (mask centred at 1) | `input_mask_mean` | `> 0.75` | the cells are uniform in [`low`, `low` + 1], so the mean is 0.5 at `low` 0 and 1.0 at `low` 0.5 |

Every run also logs `input_fluid_saturated_share` and `input_other_saturated_share` (the share of
pixels at 1 after the loader's scaling and the model's rescaling, before augmentation) as the
measured effect of the intensity options; they are read with the result, not used as a signature.
A source-change idea adds and declares its own scalar.

## Build facts for source-change ideas

- **Input path.** `KneeAbnormalityModel.encode_batch` (`modeling/model.py`): slices are divided
  by 255, optionally rescaled (`slice_upper_quantile`, `fluid_brightness`), reflected, mixed and
  augmented in [0, 1] (`_augment_images`, then AutoAugment and CutMix), expanded to three
  channels, normalized with `data.mean` and `data.std`, optionally masked, then encoded by
  `_encode_images`, which returns `maps` (N × 512 × 7 × 7).
- **Training-only modules** live on the Lightning module (`training/lightning.py`), like
  `report_attribute_heads` and `input_mask`, and reach the optimizer through the `heads` tuple
  of `build_optimizer`. Prediction loads only `model.*` keys, strictly, into a model built from
  the saved config (`inference/checkpoint.py`).
- **Clipping.** Lightning clips the global gradient norm of every optimized parameter at 1.0; a
  training-only network needs its own clipping (`configure_gradient_clipping`).
- **Mixed precision** is on; distributions and logarithms are computed in float32 with autocast
  disabled.
- **Built modules:** `modeling/pretraining_augmentation.py` holds `autoaugment_intensity(images,
  probability)`, `study_cutmix(images, token_study, token_type, probability)` and
  `RandomInputMask`; `modeling/radimagenet.py` holds the weight table and loaders.
- **Frozen sources of the previous campaign** are under
  `outputs/autoresearch/refined-radimagenet/experiments/<run>/source/`; the policy network of
  `policy-mask-rn10t-20261004T114633Z-b68939c7` is the starting point of `learned-mask-policy`.

## Evidence requirements

- Every candidate's scalars from the table above are read before its score: a mechanism whose
  scalars equal the incumbent's did not run.
- The reference run was trained before the build and has none of the new scalars. On the
  unchanged recipe `input_fluid_std` is about 0.20 and `input_other_std` about 0.28; the first
  candidates establish the logged values, and later runs are compared with them.
- Every result records ×4, ×1, the per-label deltas, the training seconds and those scalars.
- The researcher starts each refill from the per-epoch curves and per-label deltas of every
  decided run and names the behaviour each proposal targets.
