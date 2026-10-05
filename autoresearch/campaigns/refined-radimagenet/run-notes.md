# Refined-RadImageNet run notes

Per-run details the decision summary does not hold: per-label deltas, mechanism readings and the
lesson of each run. Every row is one run and weak evidence. ×4 macro AUC on fold 0 (912 studies),
80% paired report-group interval, 2,000 bootstrap draws.

## Keeper log

- **2026-10-04 12:26 CEST, first start paused at calibration**: "Baseline data/preprocessing
  identity differs from historical control". The build had edited `modeling/model.py`, a hashed
  preprocessing source. The file was restored to main's bytes and the new weight entries moved
  behind `PRETRAINED_WEIGHTS` in the unhashed `modeling/radimagenet.py`; seeds 2, 3 and 12 now
  carry the model change themselves. The failed start's folder is
  `outputs/autoresearch/refined-radimagenet.failed-calibration-20261004` (no results). Restarted
  12:32 CEST with Codex.

## Runs

### 1. `radimagenet-input-statistics` — accepted (2026-10-04)

`radimagenet-input-statistics-20261004T103433Z-ff2f3786`: `data.mean` 0.3162 and `data.std`
0.3213 in place of the ImageNet values, nothing else.

- **Result:** ×4 0.933058 against 0.932477, **+0.000581**, 80% [+0.000182, +0.001016];
  ×1 0.929948 against 0.929026 (+0.0009); soft AUC +0.000335. Accepted by the rule; it is the
  incumbent for the later seeds. Training 1,266 s against 1,181 s (the developer for the next
  seed ran beside it).
- **Per label (×4 AUC delta):** ACL +0.0028, Medial OA +0.0017, MCL +0.0016, Lateral OA +0.0009,
  PF OA +0.0008, Synovitis +0.0008, Lateral Meniscus +0.0005, Contusion +0.0003, Effusion
  +0.0002, Medial Meniscus +0.0001, Baker's −0.0003, Fracture −0.0024. Ten of twelve up.
- **Curve (×1 against the reference by epoch):** −0.0001, +0.0001, +0.0002, +0.0003, +0.0011,
  −0.0002, +0.0010, +0.0009, +0.0008, +0.0009. Level for four epochs, ahead from epoch 7.
- **Reading:** small and consistent across labels, as expected if BatchNorm in training mode
  absorbs most of an input mean and scale change (keeper's reasoning, not measured). The large
  difference seen with the frozen network (knee-class mass 0.088 against 0.978) uses stored
  BatchNorm statistics and does not carry over in size to fine-tuning.
- **Follow-up outside the campaign:** main's recipe and the ResNet-50 config still use the
  ImageNet values; moving them is the user's decision.

### 2. `rn10t-policy-weights` — measurement (2026-10-04)

`rn10t-policy-weights-20261004T110216Z-700bbacd`: ResNet-10t started from the authors' heated
(mask-trained) checkpoint, backbone LR 0.0002, L2-SP 0.00088 / 0.000088, ImageNet normalization
as its reference. No mask or policy in our training.

- **Against the ResNet-10t reference** (`20261004-094416-rn10t-view-label-bias`, standard
  weights): ×4 0.928777 against 0.927543, **+0.001235**, 95% paired report-group interval
  [−0.0015, +0.0039] (`outputs/analysis/refined-radimagenet/rn10t-policy-weights-vs-rn10t-reference`);
  ×1 0.923666 (epoch 9) against 0.921848 (+0.0018); soft AUC +0.0016. Training 1,132 s against
  1,141 s. One run: a positive start, not separated from zero.
- **Against the ResNet-18 incumbent** (the runtime's comparison): 0.928777 against 0.933058,
  −0.0043; a ResNet-10t is not expected to reach it.
- **Per label against the ResNet-10t reference (×4 AUC delta):** MCL +0.0160, Lateral Meniscus
  +0.0056, ACL +0.0048, Contusion +0.0043, Medial Meniscus +0.0004, Baker's +0.0003, Effusion
  −0.0014, PF OA −0.0017, Medial OA −0.0023, Synovitis −0.0027, Fracture −0.0032, Lateral OA
  −0.0052. Ligaments and menisci up (meniscus pair +0.0032), the three OA labels down (−0.0038).
- **By vendor:** Philips +0.0019, Siemens +0.0006, GE −0.0010.
- **Curve (×1 against the reference by epoch):** +0.0026 at epoch 2, +0.0003, −0.0014, +0.0028,
  +0.0016, +0.0021, +0.0022, +0.0023 at epoch 9: ahead from epoch 5 on.
- **Confound:** the rate (3.3× the reference's) and the anchor strengths were set from the
  checkpoint's weight scale, so the pair differs in the starting weights and in those three
  values together.
- **Open:** the same swap on the ResNet-50 (both mask-trained ResNet-50 files are cached and
  registered) is the version that would matter for submissions.

### 3. `rn10t-policy-ft-weights` — measurement (2026-10-04)

`rn10t-policy-ft-weights-20261004T110619Z-53b34de3`: ResNet-10t started from the authors' heated
checkpoint after their own fine-tuning without the mask, backbone LR 0.0002, L2-SP 0.00093 /
0.000093, ImageNet normalization as its reference.

- **Against the ResNet-10t reference:** ×4 0.928671 against 0.927543, **+0.001128**, 95% paired
  report-group interval [−0.0016, +0.0038]
  (`outputs/analysis/refined-radimagenet/rn10t-policy-ft-weights-vs-rn10t-reference`); ×1
  0.923769 against 0.921848 (+0.0019); soft AUC +0.0014. Training 1,113 s.
- **Against seed 2 (the heated checkpoint):** 0.928671 against 0.928777, −0.0001: the two
  starting points end at the same place.
- **Against the ResNet-18 incumbent:** −0.0044.
- **Per label against the ResNet-10t reference:** MCL +0.0144, Lateral Meniscus +0.0061, ACL
  +0.0055, Contusion +0.0049, Medial Meniscus +0.0004, Fracture −0.0014, Effusion −0.0014,
  Medial OA −0.0018, PF OA −0.0022, Synovitis −0.0026, Baker's −0.0027, Lateral OA −0.0057.
  Meniscus pair +0.0034, OA trio −0.0038.
- **By vendor:** Siemens +0.0009, Philips +0.0007, GE +0.0002.
- **Reading of seeds 2 and 3 together:** both mask-trained starts give about +0.0012 over the
  standard weights with the same per-label pattern (MCL, ACL, lateral meniscus and contusion
  up; the OA labels down). The two runs share their origin, their rate and their anchor
  strengths, so the repeated pattern is one piece of evidence, not two. Not separated from
  zero on either run.
- **Open question:** whether the pattern comes from the weights or from the weaker absolute
  anchor and the higher rate that came with them. A standard-weights ResNet-10t at an anchor
  weakened by the same relative amount would separate the two.
- **Next on this line:** the ResNet-50 files (`radimagenet_policy`, `radimagenet_policy_ft`)
  against the Refined ResNet-50, the swap that matters for submissions; outside this
  ResNet-18 campaign unless the user asks.

### 4. `backbone-eval-pretrained-statistics` — rejected (user seed, 2026-10-04)

`backbone-eval-pretrained-statistics-20261004T112644Z-24ec85b7`: `model.backbone_eval: true` on
the incumbent (checkpoint normalization), nothing else.

- **Result:** ×4 0.929939 against 0.933058, **−0.003119**, 80% [−0.004573, −0.001732] (95%
  [−0.0053, −0.0010]); ×1 0.925110 against 0.929948 (−0.0048); soft AUC −0.0025. Training
  1,151 s.
- **Mechanism active:** `backbone_running_statistics_changed` 0 in every epoch.
- **Per label (×4 AUC delta):** Fracture −0.0109, ACL −0.0107, MCL −0.0091, Synovitis −0.0068,
  Medial Meniscus −0.0035, Effusion −0.0023, Baker's −0.0012, PF OA −0.0003, Contusion +0.0014,
  Medial OA +0.0016, Lateral OA +0.0019, Lateral Meniscus +0.0024. The OA labels and the lateral
  meniscus gain; ligaments and fracture lose most.
- **By vendor:** Philips −0.0067, Siemens −0.0020, GE −0.0016.
- **Curve (×1 against the incumbent by epoch):** −0.0107, −0.0147, n/a, −0.0058, −0.0078,
  −0.0093, −0.0067, −0.0057, −0.0054, −0.0048. The gap halves between epoch 2 and the end and
  is still closing at epoch 10.
- **It fits more slowly, it does not overfit:** training loss is above the incumbent's in every
  epoch (0.5738 against 0.5714 at the end) and the backbone moves less (anchor penalty 0.0104
  against 0.0172).
- **Lesson and next steps (keeper's reading, one run):** with fixed statistics the backbone
  takes smaller effective steps at the same rate, so this run is the mechanism at an
  under-trained setting. Three follow-ups, each one change on this run: a higher backbone rate;
  eval mode only in the stem and layer1, where the statistics are most generic; DSU off, since
  its feature-statistics perturbation at stride 8 is no longer re-normalized by the following
  BatchNorm layers.

### 5. `policy-mask-rn10t` — rejected (2026-10-04)

`policy-mask-rn10t-20261004T114633Z-b68939c7`: policy-gradient input mask, ResNet-10t policy from
scratch, mask on for all ten epochs, expected mask at validation and prediction.

- **Result:** ×4 0.930766 against 0.933058, **−0.002292**, 80% [−0.003594, −0.001087] (95%
  [−0.0043, −0.0004]); ×1 0.925979 against 0.929948 (−0.0040); soft AUC −0.0019. Training
  1,524 s (1.29× the reference), inside the range.
- **The policy did not learn; the mask was a random uniform one.** Mean Beta parameters stayed at
  1.000 / 1.000 from epoch 2 (1.050 in epoch 1), the dataset-level logs went from 0.047 to 0.000,
  the policy gradient norm fell from 0.0022 to 0.0001. Mask mean 0.500, standard deviation 0.079
  over pixels, expectation buffer 0.500 everywhere, sample clamp hits 0.02%, parameter clamp
  hits 0. All 93,569 slices per epoch were masked. So this run measured "multiply the
  normalized input by a smooth random field around 0.5", which is what `random-smooth-mask`
  will run without the policy.
- **Why the policy got no signal (keeper's analysis, not measured beyond the scalars above):**
  the authors' loss weights the reward by `logsumexp` over the 4,096 cells of the log-density.
  At Beta(1, 1) its gradient with respect to a cell's parameter is `(log x + 1) / 4096`, whose
  mean over cells is zero, and the reward (the study loss) has no baseline, so the update is
  noise around zero; the head also enters at weight 0.1 against the running dataset term. The
  paper reports the same end state ("almost uniform").
- **Per label (×4 AUC delta):** Lateral Meniscus +0.0041, Lateral OA +0.0009, PF OA −0.0008,
  Effusion −0.0010, Contusion −0.0022, MCL −0.0027, ACL −0.0028, Medial Meniscus −0.0030,
  Synovitis −0.0039, Baker's −0.0045, Medial OA −0.0051, Fracture −0.0063.
- **By vendor:** GE −0.0024, Philips −0.0025, Siemens −0.0010.
- **Curve (×1 against the incumbent by epoch):** −0.0035, −0.0032, −0.0043, −0.0001, −0.0026,
  −0.0031, −0.0040, −0.0033, −0.0040, −0.0040. Training loss above the incumbent's throughout
  (0.5779 against 0.5714 at the end); backbone movement equal (anchor penalty 0.0173 against
  0.0172). The mask acts as a regularizer the model does not recover from within ten epochs
  while it stays on.
- **Next steps on this line:** (1) `policy-mask-two-stage`, running next, removes the mask for
  the last four epochs, the authors' own scheme; (2) a policy that can learn: reward minus its
  batch mean as baseline, the sum of the cell log-densities in place of `logsumexp`, and the
  head at full weight, one run; (3) a weaker mask (mean near 1 instead of 0.5), since halving
  the normalized input is the largest part of what this mask does.

### 6. `policy-mask-two-stage` — below the incumbent (2026-10-04)

`policy-mask-two-stage-20261004T120731Z-e6d8019f`: the mask of run 5 for epochs 1–6, then four
epochs on clean input (expectation buffer set to ones).

- **Result:** ×4 0.931325 against 0.933058, **−0.001733**, 95% [−0.0032, −0.0004]; ×1 0.926923
  against 0.929948 (−0.0030); soft AUC −0.0011. Training 1,384 s (1.17×).
- **Against run 5 (mask on for all epochs):** 0.931325 against 0.930766, +0.0006; removing the
  mask for the last four epochs recovers about a quarter of the loss.
- **Mask stage as in run 5:** the policy stayed uniform (mean Beta parameters 1.000 / 1.000 from
  epoch 4, mask mean 0.500), 93,569 slices masked per epoch in epochs 1–6 and none after; the
  expectation buffer was 0.500 through epoch 6 and 1.000 from epoch 7.
- **Stem BatchNorm follows the input scale:** mean `bn1` running variance 1.145 in the
  pretrained checkpoint, 0.21 under the mask (the input scaled by about 0.5), 0.81 within one
  epoch of the mask going off and 0.79 at the end. On clean input our slices sit below the
  pretraining variance (0.79 against 1.145).
- **Per label (×4 AUC delta):** Lateral Meniscus +0.0034, PF OA +0.0013, Baker's −0.0001,
  Effusion −0.0003, Lateral OA −0.0011, Medial OA −0.0023, Medial Meniscus −0.0024, Contusion
  −0.0024, Synovitis −0.0033, ACL −0.0034, Fracture −0.0050, MCL −0.0054.
- **By vendor:** GE −0.0003, Siemens −0.0015, Philips −0.0018.
- **Curve (×1 against the incumbent by epoch):** −0.0049, −0.0023, −0.0033, +0.0005, −0.0019,
  −0.0036, then without the mask −0.0030, −0.0025, −0.0028, −0.0030. The gap stops growing when
  the mask goes off but does not close in four epochs at the tail of the cosine schedule.
- **Pattern across runs 5 and 6:** Lateral Meniscus is up in both (+0.0041, +0.0034); Fracture,
  ACL, MCL and Synovitis are down in both.
- **Lesson:** both runs so far tested a uniform random mask, not a learned one. What they say
  about the mask itself is that scaling the normalized input by a random field around 0.5 for
  six or ten epochs costs 0.002. The learned-policy question is still open
  (run 5, next steps 2 and 3).

### 7. `policy-mask-pretrained-policy` — rejected (2026-10-04)

`policy-mask-pretrained-policy-20261004T123425Z-876070db`: as run 5, with the policy's ResNet-10t
started from the Refined-RadImageNet ResNet-10t at 6e-5.

- **Result:** ×4 0.930468 against 0.933058, **−0.002590**, 80% [−0.003837, −0.001417] (95%
  [−0.0045, −0.0008]); ×1 0.925552 against 0.929948 (−0.0044); soft AUC −0.0025. Training
  1,524 s.
- **The policy stayed uniform again, and image-independent:** mean Beta parameters 1.000 / 1.000
  from epoch 3, `input_mask_conditional_std` 0.0003 in epoch 1 and 0.0000 after, policy gradient
  norm 0.0027 falling to 0.0001, mask mean 0.500 and standard deviation 0.079. Pretrained policy
  features did not change that: the limit is the loss's signal, not the policy's features
  (run 5, analysis).
- **Runs 5 and 7 are the same experiment twice** (a uniform random mask for ten epochs with
  different random draws): 0.930766 and 0.930468, 0.0003 apart, −0.0023 and −0.0026 against the
  incumbent. That difference is a measured run-to-run yardstick for this recipe with the mask
  on.
- **Per label (×4 AUC delta):** Lateral Meniscus +0.0030, Lateral OA −0.0007, Effusion −0.0011,
  MCL −0.0014, Contusion −0.0014, Medial Meniscus −0.0017, PF OA −0.0018, Fracture −0.0032, ACL
  −0.0042, Synovitis −0.0057, Baker's −0.0063, Medial OA −0.0066.
- **Stable across the three mask runs (5, 6, 7):** Lateral Meniscus up (+0.0041, +0.0034,
  +0.0030); ACL, Synovitis, Fracture and Medial OA down in all three.
- **By vendor:** GE −0.0018, Philips −0.0020, Siemens −0.0018.

### 8. `random-smooth-mask` — below the incumbent (2026-10-04)

`random-smooth-mask-20261004T125851Z-36cbce69`: uniform cells on a 64 × 64 grid, resized and
blurred (kernel 13, sigma 6), multiplied into the normalized input for all ten epochs, constant
0.5 at validation and prediction; no policy network.

- **Result:** ×4 0.930733 against 0.933058, **−0.002325**, 95% [−0.0042, −0.0005]; ×1 0.926527
  (epoch 9) against 0.929948; soft AUC −0.0023. Training 1,180 s, the incumbent's cost.
- **Mask readings:** mean 0.500, standard deviation 0.0789 in every epoch, identical to runs 5
  and 7.
- **Three replicates of the same mask** (runs 5, 7, 8; policy inert in the first two): ×4
  0.930766, 0.930468, 0.930733, mean 0.930656, range 0.0003; against the incumbent −0.0023,
  −0.0026, −0.0023. The loss from this mask is established at about −0.0024, and the spread of
  three runs that differ only in random draws is 0.0003.
- **Per label (×4 AUC delta):** Lateral Meniscus +0.0019, MCL −0.0005, PF OA −0.0012, Lateral OA
  −0.0016, Contusion −0.0021, Effusion −0.0025, Baker's −0.0026, ACL −0.0030, Synovitis −0.0030,
  Medial Meniscus −0.0034, Fracture −0.0034, Medial OA −0.0065.
- **Stable across the four mask runs (5–8):** Lateral Meniscus up (+0.0041, +0.0034, +0.0030,
  +0.0019); Medial OA, ACL, Synovitis and Fracture down in all four.
- **By vendor:** GE −0.0010, Philips −0.0033, Siemens −0.0013.
- **Where the mask family stands:** a uniform multiplicative field around 0.5 costs about 0.0024
  when on for ten epochs and 0.0017 when removed for the last four. Untested: a mask centred
  near 1 (no halving of the input), and a policy with a usable learning signal (run 5, next
  steps 2 and 3). The lateral-meniscus gain is the one consistent positive.

### 9. `knee-logit-token-features` — rejected (2026-10-04)

`knee-logit-token-features-20261004T132549Z-b0f2608e`: the 18 pretrained knee-class logits of
every token, through a zero-initialized Linear(18, 256), added to the head's input.

- **Result:** ×4 0.931562 against 0.933058, **−0.001496**, 80% [−0.002371, −0.000728] (95%
  [−0.0028, −0.0004]); ×1 0.928557 against 0.929948 (−0.0014); soft AUC −0.0009. Training
  1,183 s.
- **Mechanism active:** the new layer's weight norm grows from 0.54 to 1.17 and flattens from
  epoch 7. The token logits are large (standard deviation 3.46 in epoch 1) and shrink to 2.76 as
  the backbone moves away from the features the frozen classifier rows were trained on. The
  most frequent top classes per token are chondral abnormality (85) and meniscal abnormality
  (91), about 25% each, then soft-tissue fluid collection (99), 13%.
- **Fit:** training loss equal to the incumbent's (0.5719 against 0.5714 at the end; slightly
  lower in epoch 1, 0.6900 against 0.6928); backbone movement equal (anchor penalty 0.0171
  against 0.0172). ×1 led at epoch 1 (+0.0017) and trailed by 0.0012–0.0025 from epoch 2.
- **Per label (×4 AUC delta):** Lateral Meniscus +0.0020, Fracture +0.0004, Effusion +0.0001,
  Medial Meniscus −0.0004, Baker's −0.0005, Contusion −0.0008, ACL −0.0017, MCL −0.0022, Medial
  OA −0.0027, PF OA −0.0033, Lateral OA −0.0040, Synovitis −0.0048. The three OA labels lose
  most, while chondral abnormality is the most frequent top class.
- **By vendor:** GE −0.0007, Philips −0.0011, Siemens −0.0022.
- **Lesson and next steps (keeper's reading, one run):** the logits enter unscaled, at several
  times the size of the 256-wide projected features, and they drift with the backbone. Two
  one-change follow-ups: normalize the 18 logits per token before the new layer (LayerNorm or a
  softmax over the knee classes); take them from a frozen copy of the pretrained backbone, which
  `pretrained-slice-teacher` builds, so they keep their pretrained meaning.

### 10. `knee-classifier-readout` — tie (2026-10-04)

`knee-classifier-readout-20261004T134652Z-0fda241c`: a second readout on the raw backbone
features, its rows initialized from the pretrained knee-class rows mapped to our targets, added
to each study logit through a per-target scale that starts at zero.

- **Result:** ×4 0.933106 against 0.933058, **+0.000048**, 95% [−0.0006, +0.0006]; ×1 0.929632
  (epoch 9) against 0.929948; soft AUC +0.0001. Training 1,191 s. The narrowest interval of the
  campaign: the run is the incumbent plus a small term.
- **The scales stayed small.** Final per-target scale: ACL 0.038, MCL 0.029, Lateral OA 0.026,
  Medial Meniscus 0.023, PF OA 0.022, Fracture 0.019, Lateral Meniscus 0.016, Contusion 0.016,
  Medial OA 0.012, Effusion 0.003, Synovitis 0.002, Baker's −0.002. Mean absolute contribution
  to a study logit 0.060 at the end (ACL 0.144, MCL 0.102, Medial Meniscus 0.103; Effusion,
  Synovitis and Baker's below 0.004). The readout rows moved 34% from their pretrained values.
- **The scales rank the mappings as the zero-shot check did, without having seen it:** positive
  and growing for the ligament, meniscus, OA, contusion and fracture mappings, and zero for the
  three targets mapped to soft-tissue fluid collection, whose zero-shot AUC was 0.53–0.60
  (prior-work.md). The pretrained directions carry signal for nine targets and the fluid
  mapping carries none.
- **Per label (×4 AUC delta):** Fracture +0.0016, Lateral OA +0.0004, Medial Meniscus +0.0003,
  Lateral Meniscus +0.0002, Effusion +0.0002, MCL +0.0001, Synovitis +0.0001, Contusion +0.0001,
  Baker's −0.0001, Medial OA −0.0002, PF OA −0.0010, ACL −0.0012.
- **Fit:** training loss equal to the incumbent's in every epoch (0.5710 against 0.5714 at the
  end). ×1 within ±0.0005 of the incumbent at every epoch.
- **Lesson and next steps (keeper's reading, one run):** the term was too small to matter: a
  scale that starts at zero and moves at the head rate reaches 0.04 in ten epochs. Follow-ups,
  one change each: start the scale at a value where the pretrained evidence counts from the
  first step, or give the scales their own higher rate; drop the fluid mapping for Effusion,
  Synovitis and Baker's (their scale stayed at zero on training data alone).

### 11. `pretrained-slice-teacher` — positive, not separated (2026-10-04)

`pretrained-slice-teacher-20261004T140749Z-4cc3f307`: a frozen copy of the pretrained ResNet-18
with its 18 knee rows gives every training slice a target distribution; the student's pooled
backbone feature predicts it through a training-only head; KL term at weight 0.1.

- **Result:** ×4 0.933391 against 0.933058, **+0.000333**, 80% [−0.000588, +0.001210] (95%
  [−0.0010, +0.0016]); ×1 0.929104 against 0.929948 (−0.0008); soft AUC +0.0002. A reject by
  the rule (the interval includes zero) and the best ×4 of the campaign after the incumbent's
  own run. Training 1,292 s (1.09×).
- **Mechanism active:** all 93,569 slices per epoch distilled; KL 0.145 falling to 0.110;
  student and teacher agree on the top knee class for 76% of slices at the start and 79% at the
  end; teacher entropy 1.25 nats over 18 classes; teacher knee-class mass 0.88–0.98 on the
  augmented training slices (sampled once per epoch).
- **Curve (×1 against the incumbent by epoch):** +0.0051, +0.0020, +0.0026, +0.0028, +0.0018,
  +0.0004, −0.0014, −0.0008, −0.0009, −0.0008. Ahead for six epochs, behind for the last four:
  the teacher helps early and holds the model back late. The backbone moved slightly less than
  the incumbent's (anchor penalty 0.0166 against 0.0172).
- **Per label (×4 AUC delta):** Lateral OA +0.0067, PF OA +0.0023, Fracture +0.0021, Baker's
  +0.0009, Contusion +0.0008, Lateral Meniscus −0.0002, Effusion −0.0007, Synovitis −0.0011,
  Medial Meniscus −0.0013, Medial OA −0.0013, ACL −0.0020, MCL −0.0020.
- **By vendor:** GE +0.0015, Siemens +0.0004, Philips −0.0013.
- **Lesson and next steps (keeper's reading, one run):** a prospect under the contract's
  definition (positive delta, clear early advantage, a clear per-label lift on the OA labels).
  The curve names the change: keep the teacher while it helps and remove it afterwards. One
  change each: the teacher weight decayed to zero over the first six epochs (linear or with the
  cosine schedule); a larger weight with the same decay. Its per-label pattern (OA up, ligaments
  down) is the mirror of the mask-trained ResNet-10t weights (ligaments up, OA down).

### 12. `knee-logit-attention-prior` — positive, not separated (2026-10-04)

`knee-logit-attention-prior-20261004T142904Z-704694f5`: each target's token attention score gains
`gain[target] × (detached pretrained logit of its mapped knee class)`, gains starting at zero.

- **Result:** ×4 0.933355 against 0.933058, **+0.000297**, 95% [−0.0003, +0.0009]; ×1 0.929798
  (epoch 9) against 0.929948; soft AUC +0.0002. Training 1,175 s, the incumbent's cost.
- **Mechanism active and used for every target:** all twelve gains ended positive, 0.018–0.058.
  Largest: Effusion 0.058, Baker's 0.053, MCL 0.033, PF OA 0.032, Lateral Meniscus 0.031.
  Smallest: Contusion 0.018, Medial Meniscus 0.019. The attention moved by 4.5–5.0% (total
  variation against the attention without the term) from epoch 2 on.
- **The fluid class is useful as a pointer though not as evidence:** Effusion and Baker's, mapped
  to soft-tissue fluid collection, took the largest attention gains here, while in run 10 the
  same class got a readout scale of zero for them. The pretrained fluid logit tells where to
  look, not whether the label is present.
- **Per label (×4 AUC delta):** Fracture +0.0017, MCL +0.0011, Synovitis +0.0010, ACL +0.0008,
  Contusion +0.0007, Lateral Meniscus +0.0004, Lateral OA +0.0003, Medial Meniscus +0.0000,
  Effusion −0.0003, Medial OA −0.0006, Baker's −0.0007, PF OA −0.0010. Eight of twelve up, all
  small.
- **By vendor:** GE +0.0005, Philips +0.0005, Siemens −0.0003.
- **Fit:** training loss equal to the incumbent's (0.5710 against 0.5714); ×1 within ±0.0004 of
  the incumbent at every epoch.
- **Lesson and next steps (keeper's reading, one run):** a prospect under the contract's
  definition, with the narrowest positive interval so far. The gains saturate early (most are
  flat from epoch 3), so the term is small by choice of the model, not by slow learning as in
  run 10. One change each: the prior from a frozen copy of the pretrained backbone (as in run
  11) instead of the drifting live features; all 18 knee logits through a learned 18-to-12 map
  instead of one mapped class per target. Candidate pair for a combination run: this prior
  (attention path, ligaments and fracture up) with the slice teacher of run 11 (backbone
  features, OA labels up).

### 13. `crossd-resnet18-drop-in` — invalid on time, first attempt (2026-10-04)

`crossd-resnet18-drop-in-20261004T145156Z-13700c8f`: the Cross-D ResNet-18 per slice, using
`modeling/crossd.py` as built (a separate kernel per slice in each of the 20 convolutions, in
chunks of 16 slices, recomputed in the backward pass).

- **Stopped by the supervisor at the 2,362 s training bound**, inside its first epoch. No score;
  no evidence about the mechanism.
- **Measured cost:** about 100–150 of 437 steps in the first 10 minutes, so 35–45 minutes per
  epoch against the incumbent's 2; peak GPU memory 11.4 GB. The cost is the per-slice kernels
  (about 208 slices per step), not memory.
- **Returned to the developer as a repair** (at most two repairs, each bounded by the same
  2,362 s).
- **A cheaper exact form exists (keeper's derivation, not implemented or measured):** the
  Fourier shift is separable per axis, so the per-slice kernel is a per-slice mix of fixed
  tensors. Unfold the input into its K × K taps, mix the taps per slice with a K² × K² matrix
  built from the two in-plane shifts, apply one ordinary convolution with the K depth slices
  stacked as K × out channels, and sum the K groups with the per-slice depth weights. Cost about
  3× a plain 3 × 3 convolution and 7× the stem, no per-slice kernels; estimated 53 minutes of
  training, over the 39-minute bound.
- **Open decision for the user (asked 2026-10-04 17:25):** skip this seed in favour of
  `crossd-3d-stem`, raise the bound for it, or leave it to the two repairs.

### 14. `crossd-3d-stem` — faulty seed, no evidence (2026-10-04)

`crossd-3d-stem-20261004T151316Z-f5d98e17`: ×1 0.613, 0.713, 0.714, 0.744 in its first four
epochs against 0.886, 0.908, 0.915, 0.915 for the incumbent.

- **Cause: the seed, written by the keeper, used the checkpoint's kernels without their shift.**
  In 2D use the Cross-D layer shifts its 3D kernel by about one voxel per axis before taking the
  middle depth slice (measured mean shifts 0.15–1.81, most within 0.05 of 1). The seed took the
  unshifted middle slice for layer1–layer4 and the unshifted 3D kernel for the stem. Checked
  label-free on held-out slices with the frozen network and its 165-class head: knee-class mass
  0.949–1.000 for the authors' dynamic network, **0.000 for the unshifted middle slices**, 0.190
  for a shift of exactly one voxel, 0.85–0.91 for each layer's own mean shift. The run trained
  from scrambled kernels.
- **A second defect in the same seed:** even with correct kernels, the stem's side depth slices
  applied ungated to neighbouring slices break the pretrained function (three identical slices:
  knee mass 0.016).
- **Not evidence against Cross-D or against adjacent slices.** It must not count toward the
  family's closure.
- **Repair:** `crossd-static-kernels` and `crossd-gated-depth-stem`
  ([seeds](crossd-static-ideas.yaml)), forced by the keeper on 2026-10-04.

### Cross-D static kernels (label-free measurements, 2026-10-04)

- **Per-layer mean shifts** on 240 training slices (40 per series type, no labels), stored as
  `CROSSD_MEAN_SHIFTS` in `modeling/crossd.py`. Per-slice standard deviation per axis: 0.001 in
  the stem, 0.01–0.10 in layer1–layer4, 0.31 in `layer4.1.conv2`. Between series types 0.03–0.05,
  within a type 0.03–0.04.
- **Fidelity of the static network to the dynamic one**, frozen, on held-out validation slices:

  | Kernels | Knee-class mass | Top-class agreement | Logit correlation |
  |---|---:|---:|---:|
  | dynamic (authors' layer) | 0.949–0.985 | 1.00 | 1.000 |
  | one mean shift per layer | 0.85–0.91 | 0.57–0.60 | 0.63 |
  | mean shift per layer and series type | 0.927 | 0.65 | 0.711 |
  | shift of exactly one voxel | 0.190 | 0.00 | 0.146 |
  | no shift | 0.000 | 0.00 | 0.210 |

- **Static kernel scale:** convolution RMS 0.2166 (17.4× the incumbent's).
- **Cost:** a plain ResNet-18. The dynamic layer costs 5.5 s per step (run 13).
- **Not built:** per-type shifts (a small fidelity gain for six kernel sets), and the exact
  dynamic layer in `layer4.1.conv2` alone, where the per-slice spread is largest.

### Keeper log, continued

- **2026-10-04 18:00 CEST, pause on the orchestrator's block of `crossd-resnet18-drop-in`:**
  "Report rejected 3 consecutive times … Ranking must contain each eligible live queue entry
  once, within the target". The developer returned `blocked` (no repair fits the bound without
  cutting epochs or input); the orchestrator blocked and parked the seed with `ranked_ids: null`,
  and the runtime read a null ranking as the whole queue including the parked idea, which can
  never validate. Fixed in `autoresearch/supervisor/results.py`: a null ranking is the queue
  without the parked ideas. Stop requested 18:01; the supervisor finishes stopping when the
  running training ends.

- **2026-10-04 18:06 CEST, user instruction:** stop at once, fix both Cross-D seeds, resume.
  `stop --abort` cancelled `crossd-3d-stem` (recorded invalid); the two repaired seeds were
  imported as forced ideas and moved ahead of `acs-stem-layer1`; resumed 18:07 with the ranking
  fix. The orchestrator's block of `crossd-resnet18-drop-in` then applied. Standing instruction
  from the user: when a run is obviously wrong and fixable, cancel, fix and resume without
  asking.

### 15. `crossd-static-kernels` — rejected (keeper seed, 2026-10-04)

`crossd-static-kernels-20261004T161234Z-c0aeebae`: the Cross-D checkpoint as a plain ResNet-18,
each layer's kernel shifted by its mean shift; backbone LR 0.0002, L2-SP 0.000033 / 0.0000033.

- **Result:** ×4 0.929155 against 0.933058, **−0.003903**, 80% [−0.005739, −0.002093] (95%
  [−0.0067, −0.0012]); ×1 0.924120 against 0.929948 (−0.0058); soft AUC −0.0038. Training
  1,175 s, the incumbent's cost: the cost problem of run 13 is solved.
- **The repair works:** ×1 0.858 after epoch 1 and 0.905 after epoch 2 (the faulty seed had
  0.613 and 0.713); all 93,569 slices per epoch processed; convolution RMS 0.2166 throughout.
- **Curve (×1 against the incumbent by epoch):** −0.0281, −0.0033, n/a, −0.0002, −0.0041,
  −0.0055, −0.0046, −0.0052, −0.0058, −0.0058. Level at epoch 4, then behind as the incumbent
  keeps improving.
- **The backbone moved with the wrong profile across depth.** Relative drift from the starting
  weights at the end, stem and layer1–layer4: 0.024, 0.039, 0.049, 0.046, 0.046, against the
  incumbent's 0.010, 0.024, 0.081, 0.096, 0.110. The early stages moved 1.6–2.4× as far as the
  incumbent's and the late stages less than half as far: a checkpoint with uniform weight
  scale moves uniformly under one rate, where the incumbent's scale lets its late stages adapt
  most. The same pattern as the original RadImageNet ResNet-50's first run.
- **Training loss equal to the incumbent's** (0.5724 against 0.5714), so the gap is in what the
  features generalize to, not in fit.
- **Per label (×4 AUC delta):** Lateral OA +0.0063, Medial Meniscus +0.0021, PF OA +0.0003,
  Effusion −0.0021, Medial OA −0.0029, Synovitis −0.0032, Baker's −0.0033, Contusion −0.0040,
  ACL −0.0050, Lateral Meniscus −0.0074, Fracture −0.0084, MCL −0.0193.
- **By vendor:** GE +0.0019, Philips −0.0030, Siemens −0.0079.
- **Lesson and next steps (keeper's reading, one run):** the first scored run of this
  checkpoint, on a static approximation (frozen fidelity 0.85 knee mass against 0.95) and with
  one rate for all depths. One change each: per-stage backbone rates that reproduce the
  incumbent's drift profile (about 0.4× in the stem and layer1, 2× in layer2–layer4); the exact
  per-slice layer in `layer4.1.conv2` only, where the shift spread is 0.31; mean shifts per
  series type (fidelity 0.93).

### ACS at the start (label-free check, 2026-10-04 18:55 CEST)

The frozen pretrained ResNet-18 with its 165-class head on 48 held-out slices, three identical
slices as depth: plain 2D weights knee-class mass 0.990; **ACS in the stem and layer1 as
`acs-stem-layer1` specifies it, 0.256**, top-class agreement 0.00, logit correlation 0.387.
Reorienting two thirds of the channels of five pretrained convolutions at step 0 breaks the
network, the same failure as `crossd-3d-stem`. The keeper cancels `acs-stem-layer1` at launch and
replaces it with `acs-gated-stem-block1` ([seed](acs-gated-ideas.yaml)): every depth-plane channel
starts as its in-plane self behind a gate at zero, in the stem and the first block of layer1
(the full stem and layer1 with both forms would cost about 2.3× the backbone, at the time bound).

### 16. `crossd-gated-depth-stem` — rejected (keeper seed, 2026-10-04)

`crossd-gated-depth-stem-20261004T161947Z-4bb697d6`: run 15 plus the stem's two neighbouring depth
slices applied to the adjacent slices through a per-channel gate that starts at zero.

- **Result:** ×4 0.928858 against 0.933058, **−0.004200**, 80% [−0.006117, −0.002313]; ×1
  0.923623 (epoch 9) against 0.929948; soft AUC −0.0039. Training 1,392 s (1.18×).
- **Against run 15 (the same network without the neighbours):** 0.928858 against 0.929155,
  **−0.0003**, inside the 0.0003 run-to-run spread measured on the mask replicates. The
  neighbouring slices added nothing measurable here.
- **The gates opened only a little:** mean absolute gate 0.012 after epoch 1 and 0.026 at the
  end, largest channel 0.10; the gated side term was 1.4% of the middle term at the start and
  3.1% at the end. Mechanism active, used weakly.
- **Curve (×1 against run 15 by epoch):** +0.0067, −0.0029, n/a, −0.0000, +0.0008, +0.0006,
  −0.0005, −0.0004, −0.0003: indistinguishable from run 15 after the first epoch.
- **Per label (×4 AUC delta against the incumbent):** Lateral OA +0.0043, Baker's +0.0005,
  Medial Meniscus −0.0004, Medial OA −0.0020, Effusion −0.0020, PF OA −0.0028, ACL −0.0037,
  Contusion −0.0040, Synovitis −0.0055, Lateral Meniscus −0.0061, Fracture −0.0072, MCL −0.0215.
  The same pattern as run 15 (Lateral OA up, MCL far down): the result is carried by the
  Cross-D base, not by the stem.
- **Lesson and next steps (keeper's reading, one run):** on a base that is 0.004 below the
  incumbent, a gate trained at the head rate from zero stays near zero, as the readout scale
  did in run 10. The adjacent-slice question is better asked on the incumbent's own weights
  (`acs-gated-stem-block1`, next) and, for this stem, with gates that start open at a small
  value or train faster.

### Keeper log, continued

- **2026-10-04 19:05 CEST, `acs-stem-layer1` cancelled at launch** by an armed `stop --abort`
  (checked every 10 s), because the label-free check showed it would start from a broken
  network; recorded invalid, no GPU time spent. Resumed 19:06 with `acs-gated-stem-block1` at
  the head of the queue.

### 17. `acs-gated-stem-block1` — cancelled on cost after one epoch (keeper seed, 2026-10-04)

`acs-gated-stem-block1-20261004T170533Z-7802eb51`.

- **The gated form is function-preserving, as designed:** ×1 0.885681 after epoch 1 against the
  incumbent's 0.885713.
- **Cost:** 250 s for the first epoch (19:16:07 to 19:20:17), 2.1× the incumbent's 118 s and
  about 2,500 s for ten epochs against the 2,362 s bound; the keeper's estimate (1.3–1.5×) was
  wrong. The developer's implementation is sound (`modeling/depth_stem.py`, the last
  convolution at the middle depth only); the cost is the stem at three depths with 3D
  convolutions at 224 px. Peak GPU memory 12.8 GB.
- **Cancelled by the keeper at 19:22 CEST** (`stop --abort`), recorded invalid; no evidence
  about the mechanism beyond the equal start.
- **Replaced by `acs-gated-stem`** ([seed](acs-gated-stem-ideas.yaml)): the gated ACS form in the
  stem only, computed at the middle depth (one 2D stem convolution plus two 21-channel 3D
  convolutions), about 1.05× the backbone. Resumed 19:27.
- **Left for the researcher:** the block-level form at lower cost, with the stem run per slice in
  2D and gated ACS only inside layer1.

### 18. `acs-gated-stem` — inert mechanism, an incumbent replicate (keeper seed, 2026-10-04)

`acs-gated-stem-20261004T172410Z-54c7c519`.

- **Result:** ×4 0.933085 against 0.933058, +0.000027, 80% [−0.000328, +0.000374]; training
  loss equal to the incumbent's in every epoch; training 1,379 s (1.17×).
- **The gates died at zero:** `acs_gate_max` 0.0001 in epochs 1–2 and 0.0000 after,
  `acs_gate_mean` 0.0000, `acs_blend_ratio` 0.0 throughout. The seed, written by the keeper,
  clamped the gate to [0, 1] in the forward pass: a gate whose first updates take it below zero
  is clamped to zero and gets no gradient again. The ACS term never contributed. The cancelled
  `acs-gated-stem-block1` had the same clamp.
- **What the run is worth:** a replicate of the incumbent through a different code path: ×4
  within 0.00003, per-label deltas within ±0.0012 (Lateral Meniscus +0.0012, Baker's +0.0012,
  Medial OA −0.0012). No evidence about adjacent slices; it must not count toward the ACS
  family.
- **Repair:** `acs-gated-stem-signed-gate` ([seed](acs-gate-fix-ideas.yaml)), the same run
  without the clamp, forced by the keeper at 19:57 CEST. Valid only if the mean absolute gate
  exceeds 0.001 from epoch 2.
- **Lesson:** a gate that starts at zero must not be clamped at zero; check the gate scalars of
  a gated seed before reading its score.

### 19. `knee-normal-contrast-attention` — tie (researcher, 2026-10-04)

`knee-normal-contrast-attention-20261004T175206Z-5bdf1182`: the attention prior of run 12 with
the evidence replaced by a bounded pathology-minus-normal class difference (normal class 93).

- **Result:** ×4 0.933153 against 0.933058, **+0.000095**, 95% [−0.0004, +0.0006]; soft AUC
  −0.00004. Training 1,173 s.
- **Against run 12 (the plain mapped-class prior, +0.000297):** −0.0002; the contrast form did
  not improve on the plain logit.
- **Mechanism active but weak:** the attention moved by 0.11% in epoch 1 and 0.37% at the end,
  against 4.5–5.0% in run 12. 37% of the prior values sat at the bound (saturation fraction
  0.35–0.37). Gains ended signed: MCL +0.037, Contusion +0.014, Lateral Meniscus +0.014,
  Effusion +0.010; Lateral OA −0.029, Medial OA −0.025, Synovitis −0.025, PF OA −0.022. For
  the OA labels the model turned attention away from tokens that look like chondral pathology
  relative to normal.
- **Per label (×4 AUC delta):** Baker's +0.0010, Effusion +0.0009, MCL +0.0008, Medial Meniscus
  +0.0006, Lateral Meniscus +0.0005, Synovitis +0.0004, Fracture −0.0001, Lateral OA −0.0003,
  Contusion −0.0003, ACL −0.0004, PF OA −0.0008, Medial OA −0.0013. All within the ±0.0012 seen
  in the inert replicate of run 18.
- **Lesson (keeper's reading):** bounding the evidence cut its effect on the attention by more
  than tenfold; the plain logit of run 12 remains the stronger form of this prior.

### 20. `acs-gated-stem-signed-gate` — tie, mechanism active (keeper seed, 2026-10-04)

`acs-gated-stem-signed-gate-20261004T180215Z-190fb6a3`: run 18 with the gate unclamped. The
campaign was stopped before the orchestrator decided it; the comparison is on disk.

- **Result:** ×4 0.933132 against 0.933058, **+0.000074**, 95% [−0.0005, +0.0006]; ×1 0.929777
  against 0.929948 (−0.0002); soft AUC +0.00005. Training 1,396 s (1.18×).
- **Mechanism active this time:** mean absolute gate 0.0056 after epoch 1 and 0.021 at the end,
  largest 0.08, 57% of the gates positive; the ACS term was 0.55% of the stem's in-plane signal
  after epoch 1 and 1.7% at the end. Valid by the seed's own condition (above 0.001 from
  epoch 2).
- **Per label (×4 AUC delta):** Lateral OA +0.0011, Lateral Meniscus +0.0007, Effusion +0.0007,
  MCL +0.0005, Baker's +0.0004, Medial Meniscus +0.0003, Synovitis +0.0003, Contusion +0.0003,
  Fracture −0.0005, ACL −0.0006, Medial OA −0.0010, PF OA −0.0012. All inside the ±0.0012 of
  the inert replicate (run 18).
- **Reading (one run):** with the gate free, the model lets under 2% of depth-oriented signal
  into the stem and the score does not move. Together with run 16 (Cross-D gated depth stem,
  −0.0003 against its base, gates at 0.026), two gated ways of feeding the neighbouring slices
  into the stem both stayed near zero and changed nothing. The block-level form (ACS inside
  layer1 at lower cost) is the untested variant.

### Keeper log, continued

- **2026-10-04 20:36 CEST, stop requested by the user** ("no more runs after the current one");
  the supervisor stopped at 20:43 after `acs-gated-stem-signed-gate` finished, before its
  decision was recorded. Queue at the stop: `slice-teacher-early-release` (prepared),
  `radimagenet-bounded-contrast-calibration`, `slice-teacher-with-knee-attention`,
  `random-mask-moment-restoration`.
- **2026-10-04 20:47 CEST, `slice-teacher-early-release` launched by hand** at the user's
  request, mimicking the supervisor: the prepared experiment's frozen source as working
  directory and import path, its `configs/training.yaml` with only the name and output root
  changed, `scripts.pipeline.train … --validation.folds [0] --run-id training`, output in
  `experiments/slice-teacher-early-release-20261004T181824Z-fc8876d1/manual-run/`. Not
  registered in the campaign state.

### 21. `slice-teacher-early-release` (release after epoch 6) — manual run, positive, not separated (2026-10-04)

`experiments/slice-teacher-early-release-20261004T181824Z-fc8876d1/manual-run`: the researcher's
idea as prepared and reviewed in the campaign, trained by hand after the stop: the slice teacher
of run 11 for epochs 1–6, no teacher pass and no distillation term in epochs 7–10.

- **Result:** ×4 0.933436 against 0.933058, **+0.000378**, 95% [−0.0009, +0.0016] (80% about
  [−0.0005, +0.0012] from the same bootstrap spread); ×1 0.929045 against 0.929948 (−0.0009);
  soft AUC +0.0004. Training 1,259 s (1.07×).
- **Against run 11 (teacher for all ten epochs, 0.933391):** +0.00005. Releasing the teacher
  after epoch 6 changed nothing.
- **Mechanism as specified:** 93,569 slices distilled in each of epochs 1–6 and none after;
  KL 0.145 to 0.128; agreement 0.76–0.78.
- **Curve (×1 against the incumbent by epoch):** +0.0053, +0.0021, +0.0028, +0.0030, +0.0015,
  +0.0004, then released: −0.0012, −0.0012, −0.0010, −0.0009. Through epoch 6 it reproduces
  run 11 within 0.0003; after the release it stays on run 11's trajectory (run 11: −0.0014,
  −0.0008, −0.0009, −0.0008).
- **Per label (×4 AUC delta):** Lateral OA +0.0065, Fracture +0.0017, Baker's +0.0009, Lateral
  Meniscus +0.0006, Contusion +0.0005, PF OA +0.0003, Synovitis −0.0000, Effusion −0.0006,
  Medial Meniscus −0.0009, MCL −0.0011, ACL −0.0014, Medial OA −0.0020. Lateral OA and Fracture
  repeat run 11's gains (+0.0067, +0.0021).
- **By vendor:** GE +0.0013, Philips −0.0001, Siemens −0.0000.
- **Reading:** the lead was already down to +0.0004 at epoch 6 with the teacher on, and the
  learning rate was at about 40% of its peak at the release, so a release there had little to
  protect and little rate left to use (user's observation, 2026-10-04). Two runs of the teacher
  now agree on ×4 (+0.0003, +0.0004) and on the Lateral OA and Fracture gains.
- **Next:** release after epoch 1, launched by hand at the user's request
  (`manual-run-release-after-1`, `active_epochs: 1`, same frozen source).

### 22. Slice teacher released after epoch 1 — manual run, tie (2026-10-04)

`experiments/slice-teacher-early-release-20261004T181824Z-fc8876d1/manual-run-release-after-1`:
run 21's frozen source with `trainer.slice_teacher.active_epochs: 1` (user's choice).

- **Result:** ×4 0.932975 against 0.933058, **−0.000083**, 95% [−0.0013, +0.0010]; ×1 0.929567
  against 0.929948 (−0.0004); soft AUC +0.0002. Training 1,221 s (1.03×).
- **Mechanism as specified:** 93,569 slices distilled in epoch 1 and none after.
- **Curve (×1 against the incumbent by epoch):** +0.0054, +0.0027, +0.0026, +0.0014, +0.0005,
  −0.0000, −0.0007, −0.0002, −0.0004, −0.0004. The head start of epoch 1 fades over four
  epochs without the teacher and the run ends level with the incumbent.
- **Against the release after epoch 6 (×1 by epoch):** +0.0001, +0.0006, −0.0002, −0.0017,
  −0.0010, −0.0004, +0.0006, +0.0011, +0.0006, +0.0005. Behind it in epochs 4–6, where the
  teacher was still sustaining that run, and ahead in epochs 7–10, where that run dipped.
- **Per label (×4 AUC delta):** Synovitis +0.0034, Lateral OA +0.0031, Contusion +0.0012,
  Baker's +0.0007, MCL +0.0006, Effusion −0.0002, Medial Meniscus −0.0008, Medial OA −0.0011, PF
  OA −0.0011, Lateral Meniscus −0.0015, Fracture −0.0022, ACL −0.0029. Lateral OA keeps about
  half of the teacher's gain; Fracture's gain is gone.

### Slice teacher: the three durations together

| Teacher on for | ×4 | Δ vs incumbent | ×1 Δ at the end | Lateral OA Δ |
|---|---:|---:|---:|---:|
| epoch 1 only | 0.932975 | −0.0001 | −0.0004 | +0.0031 |
| epochs 1–6 | 0.933436 | +0.0004 | −0.0009 | +0.0065 |
| all 10 epochs | 0.933391 | +0.0003 | −0.0008 | +0.0067 |

- **What the three runs establish (each one run, so weak):** the teacher gives a head start of
  about +0.005 on ×1 after epoch 1 that decays whether or not the teacher stays; keeping it on
  for six epochs or ten gives the same ×4 (+0.0003 to +0.0004, within the 0.0003 run-to-run
  spread of each other and of the incumbent); the one label that moves consistently is Lateral
  OA, and its gain grows with the teacher's duration.
- **What they do not support:** that the teacher holds the model back late. Removing it early
  did not lift the final score; ×4 went down by 0.0005 against the longer durations, inside the
  noise.
- **×4 and ×1 disagree in sign for the longer durations** (×4 +0.0003 to +0.0004, ×1 −0.0008 to
  −0.0009); the campaign's rule reads ×4.
