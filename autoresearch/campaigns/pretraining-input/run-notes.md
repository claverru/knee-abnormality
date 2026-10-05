# Pretraining-input run notes

Per-run details the decision summary does not hold: per-label deltas, mechanism readings and the
lesson of each run. Every row is one run and weak evidence. ×4 macro AUC on fold 0 (912 studies),
80% paired report-group interval, 2,000 bootstrap draws. Reference: `historical-control`,
×4 0.927579, ×1 0.921891, 1,162 s of training.

## Keeper log

- **2026-10-04 23:30 CEST, first start.** Budget range set by the user to 0.75–1.35 of the
  reference's training time (871–1,568 s) before any run trained.
- **2026-10-04 23:40 CEST, seed 1 blocked at review, not by a result.** The reviewer refused the
  activation signature `input_fluid_std > 0.215`: the threshold came from unaugmented loader
  measurements, while the logged scalar is measured after augmentation. No built scalar proved a
  loader quantile, a switched-off augmentation, the rotation limit or the heated weights had
  acted. The build got one scalar per option (`loader_fluid_bright_tail`,
  `loader_other_bright_tail`, `slice_quantile_rescaled_slices`, `fluid_brightness_slices`,
  `pretrained_conv_rms`, `<family>_off_slices`, `affine_rotation_max_degrees`) plus the
  diagnostics `input_fluid_saturated_share` and `input_other_saturated_share`, and the contract a
  fixed [activation signature](contract.md#activation-signatures) per option. A one-epoch,
  320-study smoke confirmed they log. The campaign freezes its source at start, so it was started
  again from scratch at 23:47; the first attempt's folder is
  `outputs/autoresearch/pretraining-input-aborted-20261004` (no training). The reviewer accepted
  the new signature on the first seed.
- **2026-10-05 08:55 CEST, user orders.** A second run of the incumbent recipe
  (`incumbent-rerun`, same seed because the campaign pins the incumbent's seed; it measures the
  run-to-run spread of one recipe) and ways to get both sides of the clipping trade:
  `slice-window-channels` (stretched, unstretched, stretched in the three input channels) and
  `slice-quantile-keep-axial-fluid` (no stretch on the axial fluid-sensitive series). All three
  in `keeper-ideas-9.yaml`, queued right after `learned-mask-policy`.

## Runs

### 1. `loader-upper-quantile-099` — rejected (2026-10-05)

`loader-upper-quantile-099-20261004T214842Z-6978c236`: `data.upper_quantile` 0.99 in place of
0.995, nothing else.

- **Result:** ×4 0.928018 against 0.927579, **+0.000439**, 80% [−0.000048, +0.000911], 95%
  [−0.000288, +0.001158]; ×1 0.922122 against 0.921891 (+0.0002); soft AUC +0.000369. Rejected
  by the rule: the 80% interval reaches below zero by 0.00005. Training 1,126 s.
- **Per label (×4 AUC delta):** ACL +0.0021, Lateral OA +0.0020, Fracture +0.0016, PF OA
  +0.0015, Medial OA +0.0008, MCL +0.0008, Contusion +0.0003, Lateral Meniscus +0.0001, Medial
  Meniscus −0.0001, Effusion −0.0004, Baker's −0.0013, Synovitis −0.0020. Eight of twelve up.
- **Mechanism:** `loader_other_bright_tail` 0.01 (signature passed). Saturated share 1.01% of
  fluid pixels and 0.98% of the others, so the clipping lands at its nominal rate on the sampled
  crops. `input_fluid_std` 0.225 and `input_other_std` 0.294 after augmentation, against about
  0.20 and 0.28 expected on the unchanged recipe.
- **Per-epoch ×1 against the reference:** −0.0005, +0.0015, +0.0016, −0.0008, −0.0003, −0.0009,
  −0.0001, n/a, +0.0001, +0.0002: no separation at any epoch.
- **Lesson:** the bone and ligament labels rise and the three fluid labels (Effusion, Synovitis,
  Baker's) fall, the opposite of what the seed expected from more fluid contrast: clipping the
  brightest 1% saturates fluid itself. A positive macro delta just short of the rule; 0.98 is
  the next point on the same knob.

### 2. `loader-upper-quantile-098` — rejected (2026-10-05)

`loader-upper-quantile-098-20261004T215130Z-4fce85a8`: `data.upper_quantile` 0.98 in place of
0.995, nothing else.

- **Result:** ×4 0.927998 against 0.927579, **+0.000420**, 80% [−0.000230, +0.001087], 95%
  [−0.000577, +0.001403]; ×1 0.922524 against 0.921891 (+0.0006); soft AUC +0.000195. Rejected
  by the rule. Against run 1 (0.99): −0.00002 at ×4. Training 1,126 s.
- **Per label (×4 AUC delta):** MCL +0.0040, Lateral OA +0.0035, ACL +0.0018, PF OA +0.0011,
  Fracture +0.0009, Medial Meniscus +0.0007, Lateral Meniscus +0.0001, Medial OA −0.0004,
  Contusion −0.0007, Effusion −0.0008, Synovitis −0.0017, Baker's −0.0035 (95%
  [−0.0071, −0.0004]). Seven of twelve up.
- **Mechanism:** `loader_other_bright_tail` 0.02 (signature passed). Saturated share 2.02% of
  fluid pixels and 1.98% of the others. `input_fluid_std` 0.255 and `input_other_std` 0.305
  (0.225 and 0.294 at 0.99).
- **Lesson:** the same split as run 1, wider: bone and ligament labels gain more, the three
  fluid labels lose more (Baker's −0.0013 then −0.0035), and the macro stays at +0.0004. The
  two effects cancel in the macro; the knob is not flat, it trades labels.
- **Keeper seed added:** `other-type-upper-quantile` (0.98 for the non-fluid series, 0.995 for
  the fluid-sensitive ones; `keeper-ideas-1.yaml`), the mirror of `fluid-type-upper-quantile`,
  queued right behind it so the pair shows which group carries the gains and which the losses.

### 3. `heated-weights` — rejected (2026-10-05)

`heated-weights-20261004T221145Z-817b28af`: `model.pretrained_weights` `radimagenet_policy`,
backbone LR 0.0002, anchor strengths 0.00088 and 0.000088; nothing else.

- **Result:** ×4 0.928618 against 0.927579, **+0.001039**, 80% [−0.000650, +0.002647], 95%
  [−0.001546, +0.003620]; ×1 0.923477 against 0.921891 (+0.0016); soft AUC +0.001549. Rejected
  by the rule: the largest delta so far and the widest interval (3.4 times that of the
  loader-quantile runs), because the predictions move much more per study. Training 1,123 s.
- **Per label (×4 AUC delta):** MCL +0.0096, ACL +0.0075, Lateral Meniscus +0.0053, Contusion
  +0.0036, Medial Meniscus +0.0001, PF OA −0.0003, Baker's −0.0006, Fracture −0.0014, Effusion
  −0.0018, Synovitis −0.0019, Medial OA −0.0022, Lateral OA −0.0055. Five of twelve up.
- **Mechanism:** `pretrained_conv_rms` 0.05305 (signature passed; 0.01576 on the standard
  weights).
- **Per-epoch ×1 against the reference:** −0.0056, +0.0023, −0.0003, −0.0014, +0.0017, +0.0005,
  +0.0014, +0.0021, n/a, +0.0016: behind in epoch 1 and ahead from epoch 5, as the seed expected.
- **Lesson:** third run of these weights with the same size and the same label pattern (runs 2
  and 3 of the previous campaign: +0.0012 and +0.0011 under the ImageNet normalization):
  ligaments, lateral meniscus and contusion up, the OA labels down. The checkpoint normalization
  did not change it. The pattern is the reverse of the loader clipping on Lateral OA (+0.0035 at
  0.98, −0.0055 here) and the same on MCL and ACL.

### 4. `slice-upper-quantile-098` — provisionally accepted (2026-10-05)

`slice-upper-quantile-098-20261004T223353Z-9b2352e9`: `model.slice_upper_quantile` 0.98 (each
slice divided by its own 98th percentile and clamped to 1, in training and prediction), loader
quantile unchanged at 0.995.

- **Result:** ×4 0.928294 against 0.927579, **+0.000716**, 80% [+0.000009, +0.001413], 95%
  [−0.000321, +0.001787]; ×1 0.922455 against 0.921891 (+0.0006); soft AUC +0.000478. Accepted
  by the rule with a margin of 0.000009; it is the incumbent for the later seeds. Against the
  loader 0.98 run: +0.0003 at ×4. Training 1,136 s.
- **Per label (×4 AUC delta):** MCL +0.0056, Lateral OA +0.0041, Lateral Meniscus +0.0037 (95%
  [+0.0004, +0.0076]), ACL +0.0021, Medial Meniscus +0.0004, PF OA −0.0002, Synovitis −0.0006,
  Medial OA −0.0006, Effusion −0.0012, Fracture −0.0014, Contusion −0.0015, Baker's −0.0020.
  Five of twelve up.
- **Mechanism:** `slice_quantile_rescaled_slices` 91,418 per epoch (signature passed). Saturated
  share 2.05% of fluid pixels and 2.11% of the others. `input_fluid_mean` 0.341 and
  `input_fluid_std` 0.263, `input_other_mean` 0.464 and `input_other_std` 0.313: more contrast
  than the loader 0.98 run (0.255 and 0.305) at the same saturated share, because every slice is
  stretched to its own range.
- **Lesson:** the same label split as the loader clipping (MCL, Lateral OA, ACL up; Effusion,
  Baker's down), with Lateral Meniscus added and Baker's losing less (−0.0020 against −0.0035).
  Three intensity runs now agree on the split; the fluid labels are the cost of every version.
  The per-type seeds that follow run on top of this per-slice rescaling.
- **Keeper seed added:** `slice-quantile-other-types` (`keeper-ideas-2.yaml`): the per-slice
  0.98 rescaling on the non-fluid series only, fluid-sensitive slices left as loaded. A small
  source change with exact edits (new option `model.slice_upper_quantile_types`, scalar
  `slice_quantile_skipped_slices`), queued behind `other-type-upper-quantile`.

### 5. `autoaugment-intensity` — rejected (2026-10-05)

`autoaugment-intensity-20261004T231503Z-4d75246d`: `data.autoaugment_intensity.probability` 1.0
and `data.intensity_augmentation.probability` 0.0, on the per-slice quantile incumbent.

- **Result:** ×4 0.925387 against 0.928294, **−0.002907**, 80% [−0.004085, −0.001671], 95%
  [−0.004752, −0.001048]; ×1 0.919449 against 0.922455 (−0.0030); soft AUC −0.002114.
  Training 1,336 s (1.15 of the reference; the policy loops over slices one at a time).
- **Per label (×4 AUC delta):** Fracture +0.0009, Medial Meniscus −0.0006, Medial OA −0.0008,
  Contusion −0.0008, ACL −0.0026, Effusion −0.0028, Lateral Meniscus −0.0037, Baker's −0.0038,
  PF OA −0.0044, Lateral OA −0.0049, Synovitis −0.0054, MCL −0.0061. One of twelve up.
- **Mechanism:** `autoaugment_intensity_slices` 67,533 per epoch of 93,569 slices (72%; the
  rest drew sub-policies whose steps were geometric or did not fire); `intensity_off_slices`
  93,569. `input_fluid_mean` 0.410 and `input_fluid_std` 0.295 against 0.341 and 0.263 on the
  incumbent: the policy's equalize, autocontrast, solarize and invert move the fluid series
  further from the incumbent's input than any clipping did.
- **Per-epoch ×1 against the incumbent:** −0.0068, −0.0051, −0.0077, −0.0015, −0.0049, −0.0037,
  −0.0025, −0.0028, −0.0022, −0.0030: behind from the first epoch and closing slowly, the curve
  of a run that is still under-trained at epoch 10.
- **Lesson:** the run changed two things at their extremes: the policy on every slice and our
  own gamma and contrast augmentation removed. The validation input has no such operations, so
  at probability 1.0 the model rarely trains on an input like the one it is scored on. The next
  point is a low probability with our intensity augmentation kept.
- **Keeper seed added:** `autoaugment-intensity-025` (`keeper-ideas-3.yaml`): probability 0.25,
  `data.intensity_augmentation` as the incumbent has it.

### 6. `fluid-type-upper-quantile` — rejected (2026-10-05)

`fluid-type-upper-quantile-20261004T231751Z-c19e605a`: `data.fluid_upper_quantile` 0.98 with
`data.upper_quantile` 0.995, on the per-slice quantile incumbent.

- **Result:** ×4 0.928047 against 0.928294, **−0.000247**, 80% [−0.000703, +0.000205], 95%
  [−0.000939, +0.000425]; ×1 0.922350 against 0.922455 (−0.0001); soft AUC −0.000276. Training
  1,135 s.
- **Per label (×4 AUC delta):** PF OA +0.0015, Contusion +0.0013, Synovitis +0.0008, Medial OA
  +0.0005, Fracture +0.0005, Lateral OA −0.0000, Effusion −0.0001, MCL −0.0002, Medial Meniscus
  −0.0006, ACL −0.0020 (95% [−0.0043, −0.0002]), Baker's −0.0022, Lateral Meniscus −0.0025 (95%
  [−0.0044, −0.0008]). Five of twelve up.
- **Mechanism:** `loader_fluid_bright_tail` 0.02 (signature passed). On top of the per-slice
  rescaling the loader clip adds little: `input_fluid_saturated_share` 2.55% against 2.05%,
  `input_fluid_std` 0.269 against 0.263; the non-fluid input is unchanged.
  `slice_quantile_rescaled_slices` fell from 91,400 to 65,900: on about 28% of the slices the
  loader clip already put the slice's 98th percentile at 1, so the per-slice step left them.
- **Lesson:** more clipping of the fluid series costs Baker's again (−0.0022, fourth run in a
  row) and now ACL and Lateral Meniscus, which are read on fluid-sensitive series too. The
  direction for the fluid series is less clipping, which the two seeds that follow test:
  `other-type-upper-quantile` and `slice-quantile-other-types`.

### 7. `other-type-upper-quantile` — rejected (2026-10-05, keeper seed)

`other-type-upper-quantile-20261004T234137Z-a7431a40`: `data.upper_quantile` 0.98 with
`data.fluid_upper_quantile` 0.995, on the per-slice quantile incumbent.

- **Result:** ×4 0.928078 against 0.928294, **−0.000217**, 80% [−0.000477, +0.000031], 95%
  [−0.000605, +0.000173]; ×1 0.922247 against 0.922455 (−0.0002); soft AUC −0.000127. Training
  1,137 s.
- **Per label (×4 AUC delta):** PF OA +0.0007, Fracture +0.0006, Effusion +0.0005, Medial OA
  +0.0002, Medial Meniscus −0.0001, Synovitis −0.0002, ACL −0.0003, MCL −0.0003, Contusion
  −0.0005, Lateral OA −0.0007, Baker's −0.0008, Lateral Meniscus −0.0018 (95%
  [−0.0032, −0.0004]). Four of twelve up.
- **Mechanism:** `loader_other_bright_tail` 0.02 (signature passed). The fluid input is the
  incumbent's exactly; the non-fluid input moves little: `input_other_saturated_share` 2.78%
  against 2.11%, `input_other_std` 0.316 against 0.313. `slice_quantile_rescaled_slices`
  78,700 against 91,400.
- **Lesson:** the seed was written before the per-slice rescaling became the incumbent. On top
  of it a loader clip of either group (runs 6 and 7) only adds saturation beyond 2% and both
  runs lose about 0.0002, with Lateral Meniscus down in both (−0.0025, −0.0018). The loader
  quantile no longer separates the groups; 2% per slice is at or past the useful amount of
  clipping. The group question is carried by `slice-quantile-other-types`, and the amount by a
  milder per-slice value (0.99), not yet queued.

### 8. `slice-quantile-other-types` — rejected (2026-10-05, keeper seed)

`slice-quantile-other-types-20261005T000148Z-7ace6ea2`: per-slice 0.98 rescaling on the
non-fluid series only (new option `model.slice_upper_quantile_types: other`), fluid-sensitive
slices as loaded.

- **Result:** ×4 0.927343 against 0.928294, **−0.000952**, 80% [−0.001677, −0.000240], 95%
  [−0.002091, +0.000129]; ×1 −0.0008 at epoch 9; soft AUC −0.000526. Against the historical
  control (0.927579): −0.0002. Training 1,133 s.
- **Per label (×4 AUC delta against the incumbent):** Baker's +0.0020, Effusion +0.0020 (95%
  [+0.0003, +0.0040]), Fracture +0.0014, Synovitis +0.0013, Contusion +0.0011, Medial OA
  +0.0009, PF OA −0.0004, Medial Meniscus −0.0004, ACL −0.0033, Lateral Meniscus −0.0033,
  Lateral OA −0.0056 (95% [−0.0096, −0.0016]), MCL −0.0070 (95% [−0.0153, −0.0002]). Six of
  twelve up.
- **Mechanism:** `slice_quantile_skipped_slices` 57,812 per epoch (signature passed),
  `slice_quantile_rescaled_slices` 34,470. Fluid input back at the reference
  (`input_fluid_saturated_share` 0.51%, `input_fluid_mean` 0.245, `input_fluid_std` 0.201);
  non-fluid input as in the incumbent.
- **Lesson:** the cleanest reading of the campaign so far. Taking the rescaling off the fluid
  series returns exactly the incumbent's label split with the sign reversed: the fluid labels
  come back and MCL, Lateral OA, Lateral Meniscus and ACL lose what they had gained. So the
  whole gain of the incumbent comes from stretching the fluid-sensitive series, on which those
  four labels are read; rescaling the non-fluid series is worth about nothing (−0.0002 against
  the historical control). The same stretch that helps ligament, meniscus and cartilage on a
  fluid-sensitive slice costs the fluid labels, because the brightest 2% it clips is the fluid.
- **Keeper seed added:** `slice-quantile-soft-knee` (`keeper-ideas-4.yaml`): keep the stretch,
  replace the hard clip above the 98th percentile by a smooth roll-off so fluid stays graded.

### 9. `fluid-brightness` — rejected (2026-10-05)

`fluid-brightness-20261005T002208Z-1fe8f3a5`: `model.fluid_brightness` 1.2 (the seed's rule for
an incumbent that already stretches the fluid series), on the per-slice quantile incumbent.

- **Result:** ×4 0.927216 against 0.928294, **−0.001079**, 80% [−0.001827, −0.000349], 95%
  [−0.002230, +0.000024]; ×1 −0.0009 at epoch 9; soft AUC −0.000734. Training 1,136 s.
- **Per label (×4 AUC delta):** Synovitis +0.0029 (95% [+0.0003, +0.0056]), Contusion +0.0010,
  PF OA +0.0001, Medial Meniscus +0.0000, Medial OA −0.0005, Lateral Meniscus −0.0006, ACL
  −0.0013, Effusion −0.0016, MCL −0.0019, Fracture −0.0023, Lateral OA −0.0033 (95%
  [−0.0064, −0.0003]), Baker's −0.0054. Four of twelve up.
- **Mechanism:** `fluid_brightness_slices` 57,812 per epoch (signature passed).
  `input_fluid_saturated_share` 5.74% against 2.05% on the incumbent, `input_fluid_mean` 0.400
  against 0.341, `input_fluid_std` 0.300 against 0.263.
- **Lesson:** a gain of 1.2 on an already stretched slice nearly triples the saturated share
  of the fluid series, and Baker's pays the most of any run (−0.0054; five clipping runs in a
  row). The saturated share orders the Baker's losses across the runs: 0.5% → reference, 1%
  −0.0013, 2% −0.0020 to −0.0035, 2.6% −0.0022 more, 5.7% −0.0054 more. The stretch past 2%
  also stops helping the ligament labels (Lateral OA −0.0033, MCL −0.0019). This supports the
  soft-knee seed: the saturation, not the stretch, carries the loss.

### 10. `no-bilateral-smoothing` — rejected (2026-10-05)

`no-bilateral-smoothing-20261005T004250Z-91a209ce`: `data.smoothing_augmentation.probability`
0.0 (from 0.2), on the per-slice quantile incumbent.

- **Result:** ×4 0.929197 against 0.928294, **+0.000902**, 80% [−0.000073, +0.001923], 95%
  [−0.000586, +0.002522]; ×1 +0.0006 at epoch 9; soft AUC +0.000517. Rejected by the rule: the
  80% interval reaches below zero by 0.00007. The highest ×4 of the campaign. Training 1,100 s
  (36 s less than the incumbent: the bilateral filter is gone).
- **Per label (×4 AUC delta):** Baker's +0.0057 (95% [+0.0006, +0.0116]), Fracture +0.0054 (95%
  [+0.0010, +0.0108]), MCL +0.0039, PF OA +0.0016, Synovitis +0.0011, Effusion +0.0011,
  Contusion +0.0003, Lateral Meniscus −0.0005, Medial OA −0.0007, Medial Meniscus −0.0009, ACL
  −0.0031, Lateral OA −0.0031. Seven of twelve up.
- **Mechanism:** `smoothing_off_slices` 93,569 per epoch (signature passed).
- **Per-epoch ×1 against the incumbent:** −0.0014, +0.0023, −0.0066, +0.0015, −0.0023, −0.0015,
  −0.0003, −0.0000, +0.0006: noisier early, ahead only at the end.
- **Lesson:** a positive prospect just short of the rule, with a wide interval (twice that of
  the clipping runs). The three fluid labels all rise, Baker's the most: edge-preserving
  smoothing flattens exactly the fine bright detail the fluid labels are read from, and it is
  the augmentation prior work measured as the largest late-layer shift. A candidate for a
  combination with another positive prospect once the single seeds are through.

### 11. `slice-quantile-soft-knee` — rejected (2026-10-05, keeper seed)

`slice-quantile-soft-knee-20261005T010301Z-bad3f2c1`: the per-slice 0.98 stretch with values
above 0.8 rolled off by a tanh instead of clamped (new option `model.slice_quantile_soft_knee`
0.8), on the per-slice quantile incumbent.

- **Result:** ×4 0.928319 against 0.928294, **+0.000025**, 80% [−0.000266, +0.000314], 95%
  [−0.000413, +0.000493]; ×1 +0.0005 at epoch 9; soft AUC +0.000128. Training 1,136 s.
- **Per label (×4 AUC delta):** Fracture +0.0010, Medial OA +0.0008, Effusion +0.0008, ACL
  +0.0007, Baker's +0.0003, MCL +0.0003, Contusion +0.0001, Medial Meniscus −0.0001, PF OA
  −0.0001, Synovitis −0.0008, Lateral OA −0.0009, Lateral Meniscus −0.0017 (95%
  [−0.0036, −0.0001]). Seven of twelve up, all small.
- **Mechanism:** `slice_soft_knee_pixels` 6.05e8 per epoch (signature passed). Saturated share
  0.004% of fluid pixels and 0.0003% of the others (2.05% and 2.11% on the incumbent) at the
  same contrast (`input_fluid_std` 0.261 against 0.263, `input_other_std` 0.310 against 0.313).
- **Lesson:** the mechanism did what it was built for and the score did not move: the gains of
  the stretch are held (MCL, ACL level), but the fluid labels did not come back (Baker's
  +0.0003, Effusion +0.0008, Synovitis −0.0008, against +0.0020, +0.0020, +0.0013 when the
  stretch was taken off the fluid series in run 8). So removing the flat top is not enough;
  the roll-off still squeezes everything above 0.8 of the slice's 98th percentile into a fifth
  of the range. Two readings remain open, both guesses: the fluid labels need the top of the
  range at full gain (a lower knee with a gentler roll-off, or a stretch with a smaller
  factor), or they need the brightness of a slice relative to the rest of its series, which a
  per-slice divisor removes whatever its shape (a per-series divisor with a roll-off would
  keep it). The option stays in the candidate's source only; it is not in the incumbent.
- **Keeper seed added:** `series-quantile-stretch` (`keeper-ideas-5.yaml`): one stretch factor per
  series (the mean of its slices' 98th percentiles), which keeps the slices of a series in
  proportion; it tests the second reading.

### 12. `no-contrast-mixing` — provisionally accepted (2026-10-05)

`no-contrast-mixing-20261005T012244Z-811efd8a`: `data.contrast_mixing_augmentation.probability`
0.0 (from 0.3), on the per-slice quantile incumbent.

- **Result:** ×4 0.930247 against 0.928294, **+0.001952**, 80% [+0.000994, +0.002938], 95%
  [+0.000348, +0.003448]; ×1 0.923998 against 0.922455 (+0.0015); soft AUC +0.001184. Accepted
  by the rule with room; the new incumbent, +0.002668 over the historical control. Training
  1,106 s.
- **Per label (×4 AUC delta):** Fracture +0.0066 (95% [+0.0025, +0.0115]), Lateral OA +0.0046,
  Baker's +0.0044, Contusion +0.0039, Medial Meniscus +0.0025, Lateral Meniscus +0.0022, MCL
  +0.0020, PF OA +0.0012, Medial OA +0.0010, Effusion −0.0003, Synovitis −0.0014, ACL −0.0032.
  Nine of twelve up.
- **Mechanism:** `contrast_mixing_off_slices` 93,569 per epoch (signature passed). Input
  statistics as the incumbent's (`input_fluid_std` 0.264, `input_other_std` 0.318).
- **Per-epoch ×1 against the previous incumbent:** −0.0008, +0.0027, −0.0051, +0.0009, +0.0008,
  +0.0014, +0.0012, +0.0015, +0.0020, +0.0015: ahead from epoch 4 on and widening.
- **Lesson:** the largest accepted effect of the campaign, and the first whose 95% interval
  excludes zero. Mixing the contrasts of two series of a study produces inputs unlike anything
  the backbone was pretrained on and unlike the validation input. Fracture, Baker's and MCL
  rise as in `no-bilateral-smoothing` (run 10: +0.0054, +0.0057, +0.0039), and ACL falls by the
  same 0.003 in both: the two switch-offs share a label pattern, so they may overlap rather
  than add.
- **Keeper seed added:** `no-bilateral-smoothing-on-no-mixing` (`keeper-ideas-6.yaml`): run 10's
  change on this incumbent.

### 13. `no-gin` — rejected (2026-10-05)

`no-gin-20261005T020921Z-64aa9ce9`: `data.gin_augmentation.probability` 0.0, on the incumbent
without contrast mixing.

- **Result:** ×4 0.929686 against 0.930247, **−0.000560**, 80% [−0.001830, +0.000706], 95%
  [−0.002522, +0.001278]; ×1 −0.0011 at epoch 9; soft AUC −0.000149. Training 1,091 s.
- **Per label (×4 AUC delta):** Synovitis +0.0042, Baker's +0.0011, MCL +0.0011, ACL +0.0009,
  Effusion +0.0008, PF OA +0.0006, Lateral OA −0.0004, Contusion −0.0009, Medial Meniscus
  −0.0010, Medial OA −0.0014, Lateral Meniscus −0.0034, Fracture −0.0083 (95%
  [−0.0190, −0.0008]). Six of twelve up.
- **Mechanism:** `gin_off_slices` 93,569 per epoch (signature passed).
- **Per-epoch ×1 against the incumbent:** +0.0055, +0.0019, +0.0055, +0.0018, −0.0024, −0.0011,
  +0.0002, −0.0011, −0.0011: ahead for four epochs, then behind. Without GIN the model learns
  faster and ends lower, the curve of an augmentation that regularizes.
- **Lesson:** the macro delta is one label: Fracture −0.0083 (59 positives) accounts for
  −0.0007 of the −0.0006. The other eleven sum to about zero, with the three fluid labels up
  again. Unlike smoothing and contrast mixing, GIN is still doing useful work at ten epochs; a
  lower probability or blend is the open point, not its removal.

### 14. `no-bilateral-smoothing-on-no-mixing` — rejected (2026-10-05, keeper seed)

`no-bilateral-smoothing-on-no-mixing-20261005T021255Z-e09913ba`:
`data.smoothing_augmentation.probability` 0.0 on the incumbent without contrast mixing.

- **Result:** ×4 0.928298 against 0.930247, **−0.001948**, 80% [−0.002988, −0.000937], 95%
  [−0.003499, −0.000348]; ×1 0.922261 against 0.923680 (−0.0014); soft AUC −0.001416. Against
  the per-slice quantile incumbent both switch-offs were measured on (0.928294): +0.000004.
  Training 1,072 s.
- **Per label (×4 AUC delta):** Synovitis +0.0045, PF OA +0.0018, Effusion −0.0001, Medial OA
  −0.0002, ACL −0.0005, Medial Meniscus −0.0008, Lateral OA −0.0022, Baker's −0.0035, Lateral
  Meniscus −0.0038, Contusion −0.0039, Fracture −0.0047 (95% [−0.0101, −0.0003]), MCL −0.0099.
  Two of twelve up.
- **Mechanism:** `smoothing_off_slices` and `contrast_mixing_off_slices` both logged
  (signature passed).
- **Per-epoch ×1 against the incumbent:** +0.0036, +0.0002, +0.0036, +0.0015, −0.0025, −0.0031,
  n/a, −0.0021, −0.0016, −0.0014: ahead for four epochs, behind from the fifth, the same curve
  as `no-gin`.
- **Lesson:** the two switch-offs do not add: +0.0009 and +0.0020 alone, +0.0000 together,
  all against the same reference. Three readings, in the order the evidence supports them.
  (1) The smoothing gain was not established: its 80% interval reached below zero and at ×1
  it is +0.0002 over the base (0.9227 against 0.9225); only the ×4 prediction showed +0.0009.
  (2) The incumbent may sit high (MCL −0.0099 and Fracture −0.0047 here, the pattern of runs
  13 to 20), which would account for about half of the −0.0019 against it; `incumbent-rerun`
  tests that. (3) Less regularization: weak evidence for this pair (final train loss 0.5777
  against the incumbent's 0.5801), clear only for `no-gin` (0.5710 with a lower validation
  score). An earlier version of this note put (3) first; the train losses do not support that.
  Final ×1 and train loss: base 0.9225 / 0.5804, smoothing off 0.9227 / 0.5811, mixing off
  0.9237 / 0.5801, both off 0.9223 / 0.5777.

### 15. `series-quantile-stretch` — rejected (2026-10-05, keeper seed)

`series-quantile-stretch-20261005T023230Z-dddceb71`: one stretch factor per series (the mean of
its slices' 98th percentiles; new option `model.slice_upper_quantile_scope: series`), on the
incumbent without contrast mixing.

- **Result:** ×4 0.929686 against 0.930247, **−0.000560**, 80% [−0.001196, +0.000060], 95%
  [−0.001627, +0.000439]; ×1 0.923448 against 0.923680 (−0.0002); soft AUC −0.000527. Training
  1,118 s.
- **Per label (×4 AUC delta):** PF OA +0.0006, Contusion +0.0005, Lateral OA +0.0003, Effusion
  +0.0002, Medial OA −0.0000, ACL −0.0001, MCL −0.0004, Medial Meniscus −0.0006, Fracture
  −0.0014, Lateral Meniscus −0.0018, Baker's −0.0019, Synovitis −0.0021. Four of twelve up.
- **Mechanism:** `slice_quantile_series_groups` 18,855 per epoch (signature passed). Saturated
  share 2.23% of fluid pixels and 3.08% of the others (2.05% and 2.11% per slice): slices
  brighter than their series mean clip more. Contrast as the incumbent's (`input_fluid_std`
  0.259, `input_other_std` 0.319).
- **Lesson:** keeping the slices of a series in proportion did not bring the fluid labels back
  (Baker's −0.0019, Synovitis −0.0021, Effusion +0.0002), so the second reading of run 11 is
  not supported either. Three shapes of the same stretch now sit within 0.0006 of each other
  (per slice with a clip, per slice with a roll-off, per series with a clip), and the only
  version that returned the fluid labels is no stretch at all on the fluid series (run 8),
  which gives up the ligament gains. On this evidence the trade is in the gain itself: the
  fluid labels are read better at the loader's original scale of the fluid-sensitive series
  and the ligament, meniscus and cartilage labels at the stretched one. One input cannot hold
  both scales; two can (the stretched and the unstretched fluid series side by side), which is
  a change of what the model sees per study and beyond a one-option seed.

### 16. `rotation-30` — rejected (2026-10-05)

`rotation-30-20261005T025145Z-c672e0c4`: `data.affine_augmentation.rotation_degrees` 30.0 (from
10.0), on the incumbent without contrast mixing.

- **Result:** ×4 0.928482 against 0.930247, **−0.001765**, 80% [−0.002582, −0.000974], 95%
  [−0.003047, −0.000546]; ×1 0.922138 against 0.923680 (−0.0015); soft AUC −0.001150. Training
  1,116 s.
- **Per label (×4 AUC delta):** ACL +0.0029, Effusion +0.0011, Synovitis +0.0003, Fracture
  −0.0005, Contusion −0.0013, Medial Meniscus −0.0022 (95% [−0.0041, −0.0005]), Baker's
  −0.0023, Lateral Meniscus −0.0023, Medial OA −0.0030 (95% [−0.0056, −0.0006]), Lateral OA
  −0.0033, PF OA −0.0033, MCL −0.0073. Three of twelve up.
- **Mechanism:** `affine_rotation_max_degrees` 29.998 (signature passed; 10 on the incumbent).
- **Per-epoch ×1 against the incumbent:** −0.0026, −0.0046, −0.0053, −0.0023, n/a, −0.0016,
  −0.0024, −0.0018, −0.0015, −0.0015: behind at every epoch, closing slowly, the curve of
  `autoaugment-intensity`.
- **Lesson:** the pretraining rotated by up to 30°, but our validation input is upright and
  the attention pooling uses positions in the knee frame; the compartment labels (medial and
  lateral OA, the menisci, MCL), which depend on where a finding sits, lose the most. Matching
  the pretraining's augmentation strength is not the same as matching its input: both
  pretraining augmentations tried at full strength (runs 5 and 16) slowed training and ended
  lower. The mild AutoAugment seed that follows tests the low-strength end.

### 17. `autoaugment-intensity-025` — rejected (2026-10-05, keeper seed)

`autoaugment-intensity-025-20261005T031147Z-e70f6c11`: `data.autoaugment_intensity.probability`
0.25 with our intensity augmentation kept, on the incumbent without contrast mixing.

- **Result:** ×4 0.926527 against 0.930247, **−0.003720**, 80% [−0.005028, −0.002470], 95%
  [−0.005716, −0.001921]; ×1 −0.0043 at epoch 9; soft AUC −0.001842. Training 1,170 s.
- **Per label (×4 AUC delta):** PF OA +0.0010, Effusion −0.0002, Lateral OA −0.0011, ACL
  −0.0015, Medial OA −0.0016, Synovitis −0.0024, Medial Meniscus −0.0027, Contusion −0.0027,
  Fracture −0.0047, Lateral Meniscus −0.0060, Baker's −0.0080, MCL −0.0148. One of twelve up.
- **Mechanism:** `autoaugment_intensity_slices` 16,931 per epoch (18% of the slices; signature
  passed).
- **Per-epoch ×1 against the incumbent:** −0.0033, +0.0013, +0.0011, −0.0008, −0.0021, −0.0033,
  −0.0030, −0.0049, −0.0043: level for four epochs, then falling behind and widening.
- **Lesson:** a quarter of the dose cost as much as the full dose (run 5: −0.0029 on its own
  reference), and the curve is the opposite of run 5's: level early, worse late, where run 5
  was behind early and closing. The loss does not scale with the share of slices, so it is not
  simply "too much augmentation for ten epochs". Two things differ from our own augmentations
  and are untested: four of the policy's operations (invert, solarize, posterize, equalize)
  produce slices unlike any validation input, whatever their share; and a batch that mixes
  such slices with plain ones gives the backbone's batch normalization bimodal statistics.
  Both are guesses.
- **Keeper seed added:** `autoaugment-mild-operations` (`keeper-ideas-7.yaml`): the same 0.25
  with invert, solarize, posterize and equalize skipped, leaving brightness, contrast,
  sharpness and autocontrast.

### 18. `study-cutmix` — rejected (2026-10-05)

`study-cutmix-20261005T033142Z-4d7e04f5`: `data.study_cutmix_augmentation.probability` 0.5, on
the incumbent without contrast mixing.

- **Result:** ×4 0.926329 against 0.930247, **−0.003917**, 80% [−0.005064, −0.002773], 95%
  [−0.005682, −0.002165]; ×1 −0.0033 at epoch 9; soft AUC −0.002856. Training 1,130 s.
- **Per label (×4 AUC delta):** Effusion −0.0000, Medial OA −0.0013, ACL −0.0018, Medial
  Meniscus −0.0023, Synovitis −0.0027, Contusion −0.0037, Lateral OA −0.0038, MCL −0.0043, PF OA
  −0.0058 (95% [−0.0103, −0.0012]), Lateral Meniscus −0.0061, Fracture −0.0069, Baker's −0.0084
  (95% [−0.0167, −0.0012]). None of twelve up.
- **Mechanism:** `study_cutmix_slices` 46,836 per epoch (half the slices; signature passed).
- **Per-epoch ×1 against the incumbent:** −0.0009, −0.0034, −0.0009, −0.0004, −0.0035, −0.0030,
  −0.0039, −0.0033, −0.0033: behind throughout, not closing.
- **Lesson:** pasting a box from another slice of the same series puts anatomy at a position
  it does not occupy; the pooling reads positions in the knee frame, so the pasted box carries
  the wrong position, and the focal labels (Baker's, Fracture, Lateral Meniscus) lose the most.
  A smaller probability or box is the mild point.
- **Reading across runs 13 to 18:** all six candidates measured against the
  `no-contrast-mixing` incumbent are negative, including two whose mechanism is nearly neutral
  on the input (`no-gin` −0.0006, `series-quantile-stretch` −0.0006). One run of the incumbent
  carries its own luck: if its 0.930247 sits above the recipe's typical score, every later
  delta is shifted down by the same amount. A guess, and about 0.0005 at most by those two
  runs; it does not change any decision so far (the other four are below −0.0017).

### 19. `union-bag-loss` — rejected (2026-10-05)

`union-bag-loss-20261005T035255Z-eac9bf78`: `trainer.union_bag_loss_weight` 0.5, on the
incumbent without contrast mixing.

- **Result:** ×4 0.926540 against 0.930247, **−0.003706**, 80% [−0.005447, −0.001899], 95%
  [−0.006477, −0.000971]; ×1 −0.0072 at epoch 9 (the ×4 prediction recovers half of the ×1
  deficit); soft AUC −0.002935. Training 1,134 s.
- **Per label (×4 AUC delta):** Medial OA +0.0057, Lateral OA −0.0008, Baker's −0.0009,
  Fracture −0.0018, Medial Meniscus −0.0026, ACL −0.0028, PF OA −0.0040, Effusion −0.0053 (95%
  [−0.0097, −0.0012]), Synovitis −0.0062, Contusion −0.0069, Lateral Meniscus −0.0075, MCL
  −0.0113. One of twelve up.
- **Mechanism:** `union_bags` 1,747 per epoch (signature passed), `union_bag_positive_rate`
  0.334, `union_bag_loss` 0.774 falling to about 0.71 in two epochs. The pairing and the OR
  target were read in the code and are consistent.
- **Per-epoch ×1 against the incumbent:** −0.0073, −0.0092, −0.0093, −0.0098, −0.0081, −0.0094,
  −0.0091, −0.0072 from epoch 2: a constant deficit from the start, not a slower curve.
- **Lesson:** the added loss asks the attention to find a finding among the tokens of two
  knees; at weight 0.5 it competes with the study loss for the same attention and head, and
  the single-study scores pay. The deficit is flat over the epochs, so more training would not
  close it; a much smaller weight (0.1) is the mild point of the mechanism. The five chimera
  and transplant seeds that follow use the same re-pooling at their own weights.

### 20. `input-mask-centred-one` — rejected (2026-10-05)

`input-mask-centred-one-20261005T041305Z-78a77d88`: random smooth input mask with `low` 0.5
(cells uniform in [0.5, 1.5], mean 1), all epochs, `model.input_mask_expectation` true, on the
incumbent without contrast mixing.

- **Result:** ×4 0.927354 against 0.930247, **−0.002893**, 80% [−0.004008, −0.001777], 95%
  [−0.004615, −0.001171]; ×1 −0.0024 at epoch 9; soft AUC −0.001876. Training 1,118 s.
- **Per label (×4 AUC delta):** ACL +0.0013, Contusion −0.0007, Synovitis −0.0010, Effusion
  −0.0012, Lateral Meniscus −0.0013, Medial Meniscus −0.0026, Baker's −0.0026, PF OA −0.0028,
  Medial OA −0.0035, Fracture −0.0043, Lateral OA −0.0057, MCL −0.0104. One of twelve up.
- **Mechanism:** `input_mask_mean` 1.00002 (signature passed; 0.5 with `low` 0).
- **Per-epoch ×1 against the incumbent:** −0.0049, n/a, −0.0002, +0.0025, −0.0063, −0.0026,
  −0.0022, −0.0031, −0.0024: level until epoch 4, behind after.
- **Lesson:** a mask centred at 1 keeps the mean input but multiplies the normalized slice by
  up to 1.5 locally, a contrast change the validation input never has.
- **The reference, across runs 13 to 20:** all eight candidates measured against the
  `no-contrast-mixing` incumbent are negative, and the same two small labels lose in almost
  every one, whatever the mechanism: MCL (48 positives) −0.0099, −0.0073, −0.0148, −0.0043,
  −0.0113, −0.0104 and +0.0011, −0.0004; Fracture (59 positives) −0.0083, −0.0047, −0.0005,
  −0.0047, −0.0069, −0.0018, −0.0043, −0.0014. The incumbent gained +0.0020 on MCL and +0.0066
  on Fracture over its own reference. Unrelated mechanisms losing the same amount on the same
  two labels reads as those two labels sitting high in the incumbent's single run rather than
  as eight separate effects. If so, every delta against this incumbent carries a bias of about
  −0.001 on the macro (two labels of twelve at about −0.006 each). A guess from the pattern;
  it would not rescue runs 16 to 20 (all below −0.0017) but it is the size of the acceptance
  margin for milder ideas.

### 21. `heated-weights-mask-release` — rejected (2026-10-05)

`heated-weights-mask-release-20261005T043322Z-661c0df7`: the heated weights (backbone LR 0.0002,
anchors 0.00088 and 0.000088) with the random input mask (`low` 0.0) for the first three epochs,
then released, on the incumbent without contrast mixing.

- **Result:** ×4 0.928095 against 0.930247, **−0.002152**, 80% [−0.003888, −0.000445], 95%
  [−0.004839, +0.000378]; ×1 −0.0009 at epoch 9; soft AUC −0.000622. Training 1,115 s.
- **Per label (×4 AUC delta):** ACL +0.0027, MCL +0.0015, Effusion +0.0008, Contusion +0.0005,
  Synovitis −0.0003, Lateral Meniscus −0.0005, PF OA −0.0009, Medial Meniscus −0.0020, Baker's
  −0.0030, Medial OA −0.0054, Fracture −0.0058, Lateral OA −0.0133 (95% [−0.0240, −0.0038]).
  Four of twelve up.
- **Mechanism:** `pretrained_conv_rms` 0.05305 (signature passed); the mask ran for three
  epochs and was released.
- **Per-epoch ×1 against the incumbent:** −0.0047, n/a, −0.0014, +0.0016, −0.0017, +0.0004,
  +0.0001, −0.0006, −0.0009: level from epoch 4, the best ×1 curve of the runs on this
  incumbent.
- **Lesson:** the label pattern of the heated weights for the fourth time: ligaments up (ACL,
  MCL), the OA labels down, Lateral OA the most (−0.0055 in run 3, −0.0133 here). The run mixes
  two mechanisms and the first one has no run of its own on this incumbent, so the mask's
  share cannot be read off; `heated-weights` alone on the current incumbent is the missing
  comparison.

### 22. `split-knee-bag-loss` — rejected (2026-10-05)

`split-knee-bag-loss-20261005T045320Z-0a6425d0`: `trainer.split_knee_bag_loss_weight` 0.5 with
the other seven labels left out, on the incumbent without contrast mixing.

- **Result:** ×4 0.925328 against 0.930247, **−0.004919**, 80% [−0.006380, −0.003556], 95%
  [−0.007290, −0.002872]; ×1 −0.0043 at epoch 9; soft AUC −0.002141. Training 1,138 s.
- **Per label (×4 AUC delta):** Effusion +0.0006, Medial OA +0.0000, ACL −0.0006, Medial
  Meniscus −0.0019, Synovitis −0.0023, Lateral OA −0.0040, Lateral Meniscus −0.0043, Contusion
  −0.0048, Baker's −0.0076, PF OA −0.0079, Fracture −0.0087, MCL −0.0175 (95%
  [−0.0349, −0.0016]). Two of twelve up.
- **Mechanism:** `split_knee_bags` 1,615 per epoch (signature passed), as the seed expected.
- **Per-epoch ×1 against the incumbent:** −0.0126, n/a, −0.0041, −0.0056, −0.0063, −0.0034,
  −0.0053, −0.0053, −0.0043: a constant deficit, as in `union-bag-loss`.
- **Lesson:** the five side labels the loss trains did not gain (Medial OA level, the menisci
  and Lateral OA down, MCL the largest loss of the campaign), and the seven it leaves out lost
  as much, so the cost is not a labelling error of the chimera; a second objective on
  re-pooled tokens at weight 0.5 pulls the shared attention away from the study loss, as the
  union bags did (−0.0037). Both bag runs point at the weight; 0.1 is the mild point.
- **Review note:** the next variant (`split-knee-bag-average`) was sent back once: the
  reviewer wanted `split_knee_bags` to count only bags with at least one supervised target.
  The developer made that change in the candidate and it was approved without intervention.
- **Keeper seed added:** `union-bag-loss-01` (`keeper-ideas-8.yaml`): the union bags at weight 0.1.

### 23. `split-knee-bag-average` — rejected (2026-10-05)

`split-knee-bag-average-20261005T051613Z-a09ba656`: `trainer.split_knee_bag_loss_weight` 0.5
with the other seven labels set to the mean of the two donors, on the incumbent without
contrast mixing. The candidate also counts only supervised bags in `split_knee_bags` (the
reviewer's repair).

- **Result:** ×4 0.927596 against 0.930247, **−0.002651**, 80% [−0.003977, −0.001359], 95%
  [−0.004683, −0.000782]; ×1 0.921713 against 0.923680 (−0.0020); soft AUC −0.001331. Against
  run 22 (other labels left out): +0.0023. Training 1,135 s.
- **Per label (×4 AUC delta):** ACL +0.0038, Medial OA +0.0016, Synovitis +0.0006, Effusion
  +0.0002, Lateral OA +0.0001, Medial Meniscus −0.0018, Contusion −0.0022, Lateral Meniscus
  −0.0032, Fracture −0.0051, PF OA −0.0059, Baker's −0.0092 (95% [−0.0185, −0.0019]), MCL
  −0.0105. Five of twelve up.
- **Mechanism:** `split_knee_bags` 1,615 per epoch (signature passed).
- **Per-epoch ×1 against the incumbent:** −0.0032, −0.0004, −0.0024, −0.0024, −0.0042, n/a,
  −0.0035, −0.0037, −0.0023, −0.0020: about half the deficit of run 22 at every epoch.
- **Lesson:** giving the chimera a target for all twelve labels halves the cost of the bag
  loss (−0.0027 against −0.0049), so part of run 22's loss came from a bag that trains five
  labels and lets the other seven drift. The two side OA labels are now level or up; PF OA and
  Baker's, the front and back labels a medial-lateral cut tells nothing about, lose the most,
  which is what an averaged target of two knees would do to them.

### 24. `split-knee-bag-or` — rejected (2026-10-05)

`split-knee-bag-or-20261005T053612Z-ccada898`: `trainer.split_knee_bag_loss_weight` 0.5 with
the other seven labels set to the OR of the two donors, on the incumbent without contrast
mixing.

- **Result:** ×4 0.926869 against 0.930247, **−0.003378**, 80% [−0.004968, −0.001745], 95%
  [−0.005900, −0.000872]; ×1 −0.0035 at epoch 9; soft AUC −0.002359. Against the three
  variants: none −0.0049, average −0.0027, OR −0.0034. Training 1,135 s.
- **Per label (×4 AUC delta):** ACL +0.0053, Medial OA +0.0006, Synovitis −0.0002, Lateral OA
  −0.0011, Effusion −0.0011, Fracture −0.0017, Lateral Meniscus −0.0017, Medial Meniscus
  −0.0023, Contusion −0.0040, PF OA −0.0082 (95% [−0.0155, −0.0011]), Baker's −0.0131 (95%
  [−0.0254, −0.0027]), MCL −0.0131. Two of twelve up.
- **Lesson:** the three variants order as none < OR < average, and PF OA and Baker's lose in
  all three (−0.0079 and −0.0076, −0.0059 and −0.0092, −0.0082 and −0.0131): half of a knee
  cut medial to lateral often lacks the patellofemoral joint or the popliteal fossa, so any
  target for those two labels on the chimera is often wrong, the OR most of all. ACL rises in
  the two variants that give the other labels a target (+0.0038, +0.0053). The front-to-back
  cut that follows assigns exactly PF OA and Baker's by side of the cut.

### 25. `front-back-bag-loss` — rejected (2026-10-05)

`front-back-bag-loss-20261005T055612Z-6ae1ac22`: the chimera cut front to back
(`trainer.front_back_bag_loss_weight` 0.5, other ten labels by OR; PF OA from the anterior donor, Baker's from the
posterior one), on the incumbent without contrast mixing.

- **Result:** ×4 0.925888 against 0.930247, **−0.004358**, 80% [−0.006103, −0.002634], 95%
  [−0.006986, −0.001724]; ×1 −0.0049 at epoch 9; soft AUC −0.003179. Training 1,142 s.
- **Per label (×4 AUC delta):** Medial OA +0.0023, ACL +0.0005, Fracture +0.0004, PF OA
  −0.0031, Medial Meniscus −0.0037, Synovitis −0.0037, Contusion −0.0040, Baker's −0.0043,
  Effusion −0.0046, Lateral OA −0.0079, Lateral Meniscus −0.0079, MCL −0.0163 (95%
  [−0.0301, −0.0023]). Three of twelve up.
- **Mechanism:** `front_back_bags` 1,615 per epoch (signature passed).
- **Lesson:** the two labels the cut assigns by side did not gain (PF OA −0.0031, Baker's
  −0.0043), though they lose less than under the medial-lateral cut (−0.0079 and −0.0076 in
  run 22). Five bag runs at weight 0.5 now sit between −0.0027 and −0.0049 with a constant
  ×1 deficit, whatever the cut and the labels; the weight is the common factor, and
  `union-bag-loss-01` (queued) tests it at 0.1.

### 26. `transplant-bag-loss` — rejected (2026-10-05)

`transplant-bag-loss-20261005T061449Z-e8e6c933`: `trainer.transplant_bag_loss_weight` 0.5 (the
donor's top 2% tokens per label pooled with the whole host, OR labels), on the incumbent
without contrast mixing.

- **Result:** ×4 0.926691 against 0.930247, **−0.003555**, 80% [−0.005228, −0.001847], 95%
  [−0.006109, −0.000888]; ×1 0.918557 against 0.923680 (−0.0051); soft AUC −0.002417. Training
  1,143 s.
- **Per label (×4 AUC delta):** Medial OA +0.0034, Effusion +0.0020, ACL +0.0002, Synovitis
  −0.0009, Medial Meniscus −0.0025, Fracture −0.0029, Lateral OA −0.0033, PF OA −0.0039, Baker's
  −0.0046, Contusion −0.0062 (95% [−0.0119, −0.0016]), Lateral Meniscus −0.0068, MCL −0.0172.
  Three of twelve up.
- **Mechanism:** `transplant_bags` 1,747 per epoch (signature passed).
- **Lesson:** the sixth bag run at weight 0.5 and the sixth between −0.0027 and −0.0049 with a
  constant ×1 deficit. Medial OA rises in five of the six (+0.0057, +0.0000, +0.0016, +0.0006,
  +0.0023, +0.0034), the one label the bags consistently help.

### 27. `learned-mask-policy` — aborted at epoch 6 by the user (2026-10-05)

`learned-mask-policy-20261005T063516Z-6b81db7e`: a policy network (ResNet-10t, two heads) that
outputs the Beta parameters of an 8 × 8 mask in [0.5, 1.5], trained by REINFORCE with a batch
baseline and a per-cell mean; prediction multiplies by 1. Stopped with `stop --abort` at 09:13
CEST on the user's order, to start the incumbent rerun; no ×4 score.

- **×1 against the incumbent, epochs 1 to 6:** −0.0090, −0.0096, −0.0141, −0.0089, −0.0080,
  −0.0104: no closing trend.
- **Mechanism:** the policy learns, which the seed set out to fix: `input_mask_alpha_mean`
  1.20, 1.51, 2.63 and `input_mask_beta_mean` 1.18, 1.48, 3.36 over three epochs,
  `input_mask_policy_grad_norm` 0.45 to 0.74 (1.000 and 0.0001 in the previous campaign),
  `input_mask_alpha_slice_std` 0.77 to 2.10 (the mask depends on the image).
- **Why it trails (measured, then a guess):** `input_mask_std` 0.29 rising to 0.37, against
  0.079 for the random 64 × 64 mask of run 20 over the same [0.5, 1.5] range: cells of an
  8 × 8 grid are larger than the blur, so whole regions train at 0.5 or 1.5 times the
  contrast while prediction sees 1. Run 20 lost 0.0029 with a mask a quarter as strong. A
  guess on top: the policy is rewarded for masks that lower the training loss, so the
  classifier can lean on a pattern that prediction does not have.
- **Time:** about 2.4 minutes per epoch, projected 24 to 26 minutes against the 1,568 s limit;
  the policy network runs on every slice at 224.
- **Next point if the line is continued:** the same grid with a narrow range (0.85 to 1.15,
  spread about 0.08 to 0.1), and the policy fed a smaller image.

### 28. `incumbent-rerun` — the same-seed noise floor (2026-10-05, ordered by the user)

`incumbent-rerun-20261005T065543Z-f0ce2e48`: the incumbent recipe again at the same seed (the
campaign pins it); `data.fluid_upper_quantile` written as 0.995, which changes no pixel.

- **Result:** ×4 0.930127 against 0.930247, **−0.000119**, 80% [−0.000346, +0.000124], 95%
  [−0.000472, +0.000237]; soft AUC −0.000042. Recorded as a rejection by the rule; read as a
  measurement.
- **Per-epoch ×1 against the incumbent:** −0.000003, +0.0001, n/a, +0.0003, +0.0001, +0.0001,
  −0.0003, −0.00002, −0.0001: the two runs drift apart slowly and stay within 0.0003.
- **Per label (×4 AUC delta):** Lateral OA +0.0003, Medial Meniscus +0.0003, Baker's +0.0002,
  MCL +0.0002, Lateral Meniscus +0.0001, PF OA +0.0000, Medial OA +0.0000, ACL −0.0001,
  Effusion −0.0004, Contusion −0.0007, Synovitis −0.0007, Fracture −0.0008. All within 0.0008.
- **What it settles:** two runs of one recipe at one seed differ by about 0.0001 on the macro
  and by under 0.001 on any label. So the deltas of this campaign are not GPU noise, down to
  about 0.0003 on the macro; below that (`slice-quantile-soft-knee` +0.00003) a delta is
  indistinguishable from a rerun. The MCL and Fracture losses of runs 13 to 26 (−0.004 to
  −0.017) are far outside this floor.
- **What it does not settle:** whether this seed is a favourable one for the incumbent on MCL
  and Fracture. Candidates that change the random stream (bag pairings, AutoAugment draws) or
  the input are perturbed more than a rerun is; only a run of the incumbent recipe at another
  seed measures that, and the campaign cannot run one.

### Equal-weight blends with the incumbent (2026-10-05, measurement, no training)

Fold-0 ×4 predictions of two runs averaged with equal weights and scored with the same
targets (the incumbent reproduces at 0.930247). No weight was tuned. One fold, one run each.

| Second member | Alone | Blend with the incumbent | Blend − incumbent |
|---|---:|---:|---:|
| `heated-weights-mask-release` (run 21) | 0.928095 | 0.931455 | **+0.001208** |
| `no-gin` (run 13) | 0.929686 | 0.930739 | +0.000492 |
| `series-quantile-stretch` (run 15) | 0.929686 | 0.930174 | −0.000073 |
| `incumbent-rerun` (run 28) | 0.930127 | 0.930234 | −0.000013 |

- **Heated blend per label:** MCL +0.0040, ACL +0.0031, Synovitis +0.0029, PF OA +0.0024,
  Lateral Meniscus +0.0023, Contusion +0.0017, Baker's +0.0017, Effusion +0.0013, Fracture
  +0.0003, Medial Meniscus +0.0000, Medial OA −0.0014, Lateral OA −0.0039. Ten of twelve at or
  above the incumbent.
- **Lesson:** the heated run is 0.0022 below the incumbent alone and adds 0.0012 in a blend,
  2.5 times what a standard-weights run 0.0006 below adds, and a same-seed rerun adds nothing.
  The two checkpoints make different errors; the heated one is worth more as a second member
  (or a second teacher) than its own score says. The two OA labels it is weakest on are the
  two the blend loses.

### Per-view reading of the stretch on the fluid series (2026-10-05, measurement, no training)

Single-view validation AUC (`val_auc_view_*`, mean of the last three epochs), run 8 (no stretch
on the three fluid-sensitive planes) minus run 4 (stretch everywhere). One pair of runs; a
same-seed rerun moves single-view rows by up to about 0.006.

| Label | Axial fluid | Coronal fluid | Sagittal fluid |
|---|---:|---:|---:|
| Baker's | +0.0011 | +0.0156 | **+0.0798** |
| MCL | +0.0016 | **−0.0232** | **−0.0204** |
| Medial OA | −0.0023 | +0.0042 | +0.0111 |
| Lateral OA | +0.0101 | −0.0060 | −0.0013 |
| ACL | −0.0091 | −0.0038 | −0.0037 |
| Synovitis | −0.0003 | −0.0043 | −0.0088 |
| Effusion | +0.0034 | +0.0055 | +0.0016 |
| Contusion | −0.0012 | +0.0080 | +0.0016 |
| Fracture | +0.0079 | −0.0003 | −0.0002 |

Level of each fluid view alone in the stretched run: MCL 0.859 axial, 0.854 coronal, 0.733
sagittal; Baker's 0.938 axial, 0.765 coronal, 0.740 sagittal; ACL 0.745, 0.862, 0.908.

- **Lesson:** the trade is by plane. The fluid labels' loss under the stretch sits on the
  sagittal fluid series (Baker's +0.080 without it); MCL's gain sits on the coronal and
  sagittal ones, and MCL is read much better on the coronal view than on the sagittal one. The
  axial fluid series, which `slice-quantile-keep-axial-fluid` leaves unstretched, moves no label
  by more than 0.010.
- **Keeper seed added:** `slice-quantile-keep-sagittal-fluid` (`keeper-ideas-12.yaml`): the
  stretch everywhere except the sagittal fluid-sensitive series.

### 29. `slice-quantile-keep-axial-fluid` — rejected (2026-10-05, keeper seed, ordered first by the user)

`slice-quantile-keep-axial-fluid-20261005T071345Z-9f3c887a`: the per-slice stretch everywhere
except the axial fluid-sensitive series, on the incumbent without contrast mixing.

- **Result:** ×4 0.929877 against 0.930247, **−0.000370**, 80% [−0.000980, +0.000236], 95%
  [−0.001378, +0.000559]; ×1 0.924046 against 0.923680 (+0.0004); soft AUC +0.000005. Training
  1,117 s. Same-seed rerun floor: 0.0001 on the macro, 0.0008 on a label.
- **Per label (×4 AUC delta):** Baker's +0.0049 (95% [+0.0014, +0.0087]), Lateral Meniscus
  +0.0021 (95% [+0.0002, +0.0043]), Synovitis +0.0014, Medial OA −0.0001, Medial Meniscus
  −0.0002, Lateral OA −0.0005, PF OA −0.0005, Effusion −0.0006, ACL −0.0012, Fracture −0.0013,
  Contusion −0.0017, MCL −0.0068 (95% [−0.0152, −0.0003]). Three of twelve up.
- **Mechanism:** `slice_quantile_kept_slices` 22,806 per epoch (signature passed);
  `input_fluid_saturated_share` 1.44% against 2.05%. On the axial fluid view alone at epoch 7:
  Fracture +0.0074, Synovitis +0.0061, Baker's +0.0059, PF OA +0.0049, MCL −0.0234 against the
  incumbent's view (rerun: +0.0002, +0.0009, +0.0003, −0.0022, −0.0026).
- **Lesson:** leaving one fluid plane unstretched is enough to return Baker's (+0.0049, the
  largest Baker's gain of any clipping variant) and it costs MCL alone (−0.0068); the other ten
  labels stay within 0.002. So the trade can be cut down to two labels by plane. It also
  contradicts one row of the per-view table above: there the axial fluid view showed no MCL
  change between runs 8 and 4 (+0.0016), here it shows −0.023, so single-view rows from one
  pair of runs are not reliable for MCL (48 positives). The sagittal version runs next.

### 30. `heated-weights-on-incumbent` — rejected (2026-10-05, ordered by the user)

`heated-weights-on-incumbent-20261005T073414Z-c2371b78`: the heated checkpoint (backbone LR
0.0002, anchors 0.00088 and 0.000088) on the incumbent without contrast mixing, no mask.

- **Result:** ×4 0.928577 against 0.930247, **−0.001669**, 80% [−0.003376, −0.000016], 95%
  [−0.004223, +0.000862]; ×1 0.922155 against 0.923680 (−0.0015); soft AUC −0.000312. Against
  run 21 (the same weights with the three-epoch mask): +0.0005, so the mask did not help them.
- **Per label (×4 AUC delta):** ACL +0.0052, Lateral Meniscus +0.0029, Contusion +0.0004, PF OA
  +0.0004, Synovitis −0.0001, Effusion −0.0008, Fracture −0.0014, Medial Meniscus −0.0017,
  Medial OA −0.0036, MCL −0.0044, Baker's −0.0044, Lateral OA −0.0124 (95%
  [−0.0225, −0.0026]). Four of twelve up.
- **Per-epoch ×1 against the incumbent:** −0.0035, +0.0050, +0.0003, +0.0017, n/a, −0.0008,
  −0.0000, −0.0018, −0.0022, −0.0015: ahead early, behind from epoch 8.
- **Equal-weight blend with the incumbent:** 0.931334, **+0.001087** over the incumbent (run
  21's blend: +0.001208): ACL +0.0042, Lateral Meniscus +0.0040, PF OA +0.0031, Synovitis
  +0.0029, Contusion +0.0016, Fracture +0.0012; Lateral OA −0.0041. The blend with a
  standard-weights run of the same campaign (`slice-quantile-keep-axial-fluid`): +0.00001.
- **Lesson:** fifth run with the heated label pattern (ACL and Lateral Meniscus up, the OA
  labels down, Lateral OA −0.012). On the first recipe these weights were +0.0010 over their
  reference; on the incumbent they are −0.0017: the two accepted input changes helped the
  standard weights and not these. Their value is as a second view: two different heated runs
  each add about 0.0011 to 0.0012 in a blend, a standard-weights run adds nothing. This run's
  checkpoint is the teacher of `heated-teacher-distillation` (`keeper-ideas-13.yaml`).

### 31. `slice-quantile-keep-sagittal-fluid` — rejected (2026-10-05, keeper seed)

`slice-quantile-keep-sagittal-fluid-20261005T075411Z-a9179b06`: the per-slice stretch
everywhere except the sagittal fluid-sensitive series, on the incumbent without contrast
mixing.

- **Result:** ×4 0.929312 against 0.930247, **−0.000934**, 80% [−0.001449, −0.000422], 95%
  [−0.001695, −0.000158]; ×1 0.923345 against 0.923680 (−0.0003); soft AUC −0.000369. Training
  1,108 s.
- **Per label (×4 AUC delta):** ACL +0.0014, Medial OA +0.0009, Contusion +0.0003, Effusion
  −0.0001, Synovitis −0.0003, Medial Meniscus −0.0004, Lateral OA −0.0008, Lateral Meniscus
  −0.0018, PF OA −0.0019, Baker's −0.0027 (95% [−0.0056, −0.0002]), MCL −0.0028, Fracture
  −0.0030 (95% [−0.0059, −0.0006]). Three of twelve up.
- **Mechanism:** `slice_quantile_kept_slices` 17,266 per epoch (signature passed).
- **Lesson:** the per-view table predicted Baker's up on this plane (+0.080 on the sagittal
  fluid view between runs 8 and 4) and Baker's fell (−0.0027). The prediction was wrong, and
  with the MCL row already contradicted by run 29 the table should not be used to pick planes:
  a view's standalone score depends on how the shared head and attention were trained across
  all views in that run, so the difference between two runs is not the effect of that view's
  input. The two plane runs themselves are the evidence: leaving the **axial** fluid series
  unstretched returns Baker's (+0.0049) at MCL's cost (−0.0068); leaving the **sagittal** one
  unstretched returns nothing and costs Baker's, Fracture and MCL. Baker's is read on the
  axial fluid series (0.938 alone, its best view by far), which is where the clip hurts it.

### 32. `heated-teacher-distillation` — rejected, level with the incumbent (2026-10-05, asked by the user)

`heated-teacher-distillation-20261005T081428Z-ec8a2b04`: the standard-weights student with the
trained heated model of run 30 as a frozen teacher (extra BCE against the teacher's
probabilities on all labels, weight 0.5), on the incumbent without contrast mixing.

- **Result:** ×4 0.930351 against 0.930247, **+0.000105**, 80% [−0.000486, +0.000680], 95%
  [−0.000825, +0.000969]; ×1 0.9244 against 0.9237 at epoch 10; soft AUC +0.000187. The first
  of sixteen candidates on this incumbent that is not below it; the macro delta is the size of
  the same-seed rerun floor. Training 1,219 s (1.10 of the incumbent's 1,106 s).
- **Per label (×4 AUC delta):** MCL +0.0039, ACL +0.0034, Lateral Meniscus +0.0011, Medial
  Meniscus +0.0004, Baker's −0.0001, Lateral OA −0.0001, Synovitis −0.0005, Effusion −0.0005,
  Fracture −0.0010, Medial OA −0.0011, Contusion −0.0014, PF OA −0.0029. Four of twelve up.
- **Mechanism:** `teacher_distillation_loss` 0.513 falling to 0.439; `teacher_student_gap`
  0.141 falling to 0.050 (signature passed).
- **Per-epoch ×1 against the incumbent:** +0.0052, +0.0052, +0.0034, +0.0031, +0.0012, +0.0015,
  +0.0011, +0.0007, +0.0006: ahead at every epoch, the lead shrinking as the student converges.
- **Lesson:** the transfer happened where it was wanted: the student took the heated side's
  ligament gains (MCL +0.0039, ACL +0.0034, Lateral Meniscus +0.0011; the teacher alone: −0.0044,
  +0.0052, +0.0029) and kept Lateral OA (−0.0001 against the teacher's −0.0124). What it paid
  is elsewhere (PF OA −0.0029, Contusion, Medial OA), so the macro is level, against +0.0011
  for the blend of the two models. The student imitates the heated model on every label,
  including the ones where the heated model is the worse of the two; the blend does not. The
  next step is to make the teacher the blend itself.
- **Keeper seed added:** `blend-teacher-distillation` (`keeper-ideas-14.yaml`): the teacher is
  the mean of the incumbent's and the heated model's probabilities.

### 33. `slice-window-channels` — rejected (2026-10-05, keeper seed, asked by the user)

`slice-window-channels-20261005T083424Z-68f235ff`: the three input channels as stretched, as
loaded, stretched (the stretch applied after augmentation), on the incumbent without contrast
mixing.

- **Result:** ×4 0.929201 against 0.930247, **−0.001046**, 80% [−0.001748, −0.000385], 95%
  [−0.002092, −0.000049]; ×1 +0.0004 at epoch 9; soft AUC −0.000555. Training 1,113 s.
- **Per label (×4 AUC delta):** Effusion +0.0010, Synovitis +0.0008, Baker's +0.0003, ACL
  −0.0003, Medial OA −0.0007, Contusion −0.0007, Medial Meniscus −0.0007, Lateral Meniscus
  −0.0007, Lateral OA −0.0013, Fracture −0.0029, PF OA −0.0034 (95% [−0.0067, −0.0001]), MCL
  −0.0040. Three of twelve up: the three fluid labels.
- **Mechanism:** `slice_window_channel_gap` 0.088 (signature passed); the diff matches the seed.
- **Lesson:** both scales in the input moved the labels in the direction of less clipping (the
  fluid labels slightly up, MCL down), about a third of the way to the axial-plane run, and
  the macro lost 0.001. The pretrained first layer sums its three channels with nearly equal
  weights (the checkpoint was trained on grey images repeated three times), so at the start it
  sees the average of the two scales, a weaker stretch, not two separate windows; ten epochs
  at a backbone rate of 6e-5 under the L2-SP anchor did not turn it into a window-aware stem.
  The run also moved the stretch after the augmentation, a second change. Every way of
  reducing the clip on the fluid series now shows the same trade in proportion (runs 8, 11,
  15, 29, 31, 33); none has kept MCL.

### 34. `feature-sparsity-penalty` (weight 0.05) — scored, decision not recorded (2026-10-05, asked by the user)

`feature-sparsity-penalty-20261005T085611Z-0ab381d6`: an L1 penalty on the final feature map
(`optimizer.feature_sparsity_weight` 0.05), on the incumbent without contrast mixing. The
campaign was stopped on the user's order when this run ended, so the comparison exists and the
orchestrator's decision does not (the rule's verdict is reject).

- **Result:** ×4 0.929427 against 0.930247, **−0.000819**, 95% [−0.001641, −0.000013]; ×1
  0.9233 against 0.9237 at epoch 10; soft AUC −0.000566. Training 1,115 s.
- **Per label (×4 AUC delta):** Lateral Meniscus +0.0031, Synovitis +0.0002, PF OA −0.0002,
  Medial Meniscus −0.0002, Lateral OA −0.0006, Medial OA −0.0012, MCL −0.0012, Effusion −0.0013,
  ACL −0.0014, Contusion −0.0017, Baker's −0.0027, Fracture −0.0028. Two of twelve up.
- **Mechanism:** `feature_sparsity_penalty` 0.021 falling to 0.014; `feature_zero_share` 0.459,
  0.435, 0.436, 0.429, 0.423, 0.420, 0.418, 0.418, 0.418, 0.418.
- **Lesson:** at this weight the penalty shrank the activations by a third and the share of
  zero features fell instead of rising (0.46 to 0.42; the heated checkpoint is at 0.88), so the
  run did not produce the property it was meant to test. A mean-activation penalty scales
  features down before it switches any off. The 0.5 run was launched by hand to see whether a
  ten times stronger pull changes that.

## Campaign stopped (2026-10-05, about 11:33 CEST)

Stopped by the user after run 34. At the stop: 33 decided (2 accepted), incumbent
`no-contrast-mixing` at ×4 0.930247 (+0.002668 over the historical control). Not run, by the
user's decision: `blend-teacher-distillation` (prepared; it needs both teachers trained first),
`anchor-toward-heated`, `autoaugment-mild-operations` and `union-bag-loss-01` (removal requests
left in the inbox), and no researcher refill. `feature-sparsity-penalty-05` was run by hand
instead: `experiments/feature-sparsity-penalty-05-manual-20261005/`, a copy of run 34's frozen
source with the weight set to 0.5, launched with the supervisor's command, sandbox and
environment (`manual-launch.sh`, `manual-spec.json`).

### 35. `feature-sparsity-penalty-05` (weight 0.5) — manual run, stopped at epoch 4 by the user (2026-10-05)

`experiments/feature-sparsity-penalty-05-manual-20261005/`: run 34's frozen source with
`optimizer.feature_sparsity_weight` 0.5, launched by hand with the supervisor's command,
sandbox and environment after the campaign stopped. Stopped at 11:42 CEST on the user's order
during the fourth epoch; no ×4 score. (A first launch failed at config load because the job
environment, `PYTHONPATH` to the experiment's source among others, had been left out.)

| Epoch | Zero share | Penalty | Train loss | ×1 | Incumbent ×1 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.347 | 0.136 | 0.852 | 0.8501 | 0.8761 |
| 2 | 0.327 | 0.068 | 0.721 | 0.8727 | 0.8998 |
| 3 | 0.391 | 0.052 | 0.693 | 0.8784 | 0.9058 |

- **Lesson:** ten times the weight pushed the features the wrong way harder: the share of zero
  final features dropped to 0.33 before turning up to 0.39 (0.46 at the start on the standard
  weights, 0.88 on the heated checkpoint), and the ×1 score sat 0.026 to 0.027 below the
  incumbent for three epochs with no closing. Runs 34 and 35 agree: a penalty on the mean
  activation is paid by shrinking every feature (the penalty fell from 0.136 to 0.052), which
  BatchNorm and the head can undo, not by switching features off. It does not reproduce the
  heated checkpoint's sparsity. A penalty that targets zeros would have to act on the share
  of active features itself, or the sparsity has to come from where the heated checkpoint got
  it, a mask on the input.

### 36. `anchor-toward-heated` — manual run, stopped after epoch 2 by the user (2026-10-05)

`experiments/anchor-toward-heated-manual-20261005/`: the incumbent's frozen source with the
seed's edits applied by the keeper (the L2-SP reference of each anchored backbone parameter is
the heated checkpoint's parameter, rescaled to the standard layer's norm; new option
`optimizer.anchor_target_weights`, scalar `anchor_target_distance`), anchor strengths 0.001 and
0.0001. Compiled, linted, diff checked against the incumbent; no reviewer pass (campaign
stopped). Launched with the supervisor's command, sandbox and environment; stopped at 11:47
CEST on the user's order after the second epoch; no ×4 score.

| Epoch | Anchor penalty | Train loss | ×1 | Incumbent ×1 |
|---|---:|---:|---:|---:|
| 1 | 0.572 | 1.272 | 0.8761 | 0.8761 |
| 2 | 0.528 | 1.155 | 0.9008 | 0.8998 |

- **Mechanism:** `anchor_target_distance` 49.25. The penalty was larger than the task loss
  share (about 0.57 of a 1.27 total) and fell by 8% in an epoch, so the weights were moving
  toward the reference slowly; at this rate ten epochs would not bring them close.
- **Lesson:** two epochs level with the incumbent (+0.0000, +0.0010) under a pull that
  dominated the loss: no early sign of a transfer, and none of harm. Not a candidate for
  adoption on the user's criterion (only if clearly better), and it would in any case give up
  what the anchor was adopted for: the reference is no longer the starting weights, so the
  term stops limiting drift from the pretrained features that the site-generalization
  campaign kept L2-SP for, and fold 0 cannot measure that.

## Checkpoints removed (2026-10-05, about 12:08 CEST)

On the user's order all 72 checkpoint files of this campaign (36 runs, 5.7 GB) were deleted
from `outputs/autoresearch/pretraining-input/experiments/*/training/fold_0/`. Predictions,
metrics, configs, frozen sources and comparisons remain (1.5 GB). The campaign's comparison
tool reads a run's selected checkpoint, so new comparisons against these runs, a resume of the
campaign and the teacher seeds (which load the incumbent's and the heated run's checkpoints)
need those runs retrained first.

## ResNet-18 with the two adopted changes (2026-10-05, manual run asked by the user)

`outputs/training/20261005-094913-rn18-slice-quantile-no-mixing`: the previous campaign's
ResNet-18 incumbent config (`radimagenet-input-statistics-20261004T103433Z-ff2f3786`, the
checkpoint normalization, seed 1533697213, fold 0, ten epochs) with `model.slice_upper_quantile`
0.98 and `data.contrast_mixing_augmentation.probability` 0.0, trained from this worktree's
build. Reference: that incumbent run, trained from the earlier source without the build's
options. Comparison in `outputs/analysis/rn18-slice-quantile-no-mixing-vs-rn18-reference/`.

- **Result:** ×4 0.931789 against 0.933058, **−0.001269**, 95% [−0.003347, +0.000695]; soft AUC
  −0.000749. One run. On the ResNet-10t the same two changes were +0.002668 over its reference.
- **Per label (×4 AUC delta):** MCL +0.0076, Baker's +0.0019, Effusion +0.0004, Medial Meniscus
  −0.0005, Lateral OA −0.0014, Synovitis −0.0019, Contusion −0.0021, ACL −0.0024, Lateral
  Meniscus −0.0025, PF OA −0.0030, Medial OA −0.0034, Fracture −0.0080. Three of twelve up.
- **Per-epoch ×1 against the reference:** +0.0005, +0.0016, −0.0014, +0.0021, −0.0003, −0.0003,
  −0.0027, −0.0013, −0.0013, −0.0015 (the ResNet-10t: −0.0006, +0.0034, −0.0031, +0.0006,
  +0.0013, +0.0005, +0.0013, +0.0023, +0.0025, +0.0018).
- **Lesson:** the two changes did not carry over to the ResNet-18 in this run: the macro is
  0.0013 lower with an interval that includes zero, and the label pattern is not the 10t's.
  MCL rises on both; Fracture, the largest gain of `no-contrast-mixing` on the 10t (+0.0066),
  is the largest loss here (−0.0080), and Lateral OA, Lateral Meniscus and ACL do not gain.
  That fits the earlier suspicion that part of the 10t incumbent's gain sat on small labels in
  one run. What it leaves open: the two changes were not run one at a time on the ResNet-18,
  and the reference comes from the earlier source (the build is assumed equivalent with its
  options off; the 10t campaign's reference check covered that for the 10t only).

### ResNet-18 with the per-slice quantile alone (2026-10-05, manual run)

`outputs/training/20261005-101600-rn18-slice-quantile-098`: the same ResNet-18 reference config
with `model.slice_upper_quantile` 0.98 only (contrast mixing on at 0.3). Comparison in
`outputs/analysis/rn18-slice-quantile-098-vs-rn18-reference/`. A run with contrast mixing off
alone was started first and stopped after two minutes on the user's choice, because that
question already has a ResNet-18 answer (`rn18-native-contrast-preservation`, +0.000551, 80%
[−0.001057, +0.002199], in the `radimagenet-rn18` campaign).

- **Result:** ×4 0.932317 against 0.933058, **−0.000741**, 95% [−0.001987, +0.000450]; soft AUC
  −0.000454. One run.
- **Per label (×4 AUC delta):** Fracture +0.0048, Lateral Meniscus +0.0024, MCL +0.0011,
  Effusion +0.0002, Contusion −0.0003, Medial Meniscus −0.0004, Lateral OA −0.0012, Synovitis
  −0.0013, Medial OA −0.0024, ACL −0.0037, Baker's −0.0040, PF OA −0.0040. Four of twelve up.
- **Per-epoch ×1 against the reference:** −0.0015, −0.0008, −0.0007, −0.0010, −0.0015, −0.0012,
  −0.0010, −0.0011, −0.0012, −0.0012: a constant deficit of about 0.001 from the first epoch.
- **Lesson:** the per-slice stretch does not carry over to the ResNet-18 either: below the
  reference at every epoch, −0.0007 at ×4 with an interval that includes zero. Of its ResNet-10t
  signature (MCL, Lateral OA, Lateral Meniscus, ACL up; Baker's and Effusion down) the ResNet-18
  shows MCL and Lateral Meniscus up and Baker's down, but ACL and Lateral OA down. With the
  combined run (−0.0013) and the earlier contrast-mixing run (+0.0006, not separated), neither
  ResNet-10t acceptance is supported on the ResNet-18; the recipe on main keeps only the
  checkpoint normalization.
