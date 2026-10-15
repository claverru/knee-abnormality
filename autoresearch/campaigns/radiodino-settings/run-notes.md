# RadioDINO settings: run notes

Kept by the keeper. Times in CEST.

## Start (2026-10-15 01:29)

Started with Codex after the user's four `resnet50-silver` folds finished. The first start paused at
the seal: the baseline replay's input identity differed from the reference's in
`preprocessing_identity.sources_sha256` and `preprocessing_identity.spatial_tokens.view_label_bias`
only ([prior work](prior-work.md#the-reference-against-todays-source)). Stopped, the runtime moved to
`outputs/autoresearch/radiodino-settings.failed-seal-20261015T0140`, both added as reference identity
exceptions, restarted 01:36; sealed 01:40.

## Seed 1, `radiodino-on-mamba-recipe` (2026-10-15 02:28, adopted by the user's decision)

Training 2,701 s (0.87 of the reference's 3,106 s), peak 11.8 GB against 12.7 GB, 22.0M model
parameters with the heads against 31.7M.

**The comparison first failed:** it reads the reference's checkpoint, which the checkpoint wipe of
2026-10-14 had deleted. The exported copy of that checkpoint (`model.ckpt` of bundle
`mambavision-t-combined-f2-x4-check`, sha256 `da49cf7d…`, the manifest's `exported_sha256`; epoch 11,
step 5,304) was downloaded from the Kaggle dataset `claverru/knee-profile-mambavision-t-f2` and put
back as `outputs/training/mambavision-t-combined-12-epochs/fold_2/best-epoch=11-val_macro_auc=0.92988.ckpt`.
With the campaign stopped, the experiment was sent back to `needs_comparison` (state backed up as
`state.before-recompare-20261015.json`, the failed evaluation kept as
`evaluation.failed-missing-reference-checkpoint.json`) and resumed; no retraining.

**Result against the MambaVision-T reference:** ×4 0.929220 against 0.937801 (−0.008581), ×1
0.923642 against 0.929880 (−0.006238), 80% interval of ×4 [−0.010865, −0.006304], soft AUC
0.791241 against 0.797376. Runtime verdict reject; adopted by hand at 02:34 under the contract
(decision `user-adopt-20261015T003410Z-a3ec3bcc`). The comparison also carries the mask decoder's
removal since the reference.

Validation macro AUC per epoch: 0.8485, 0.8839, 0.8942, 0.9030, 0.9070, 0.9118, 0.9176, 0.9196,
0.9208, 0.9198, 0.9227, 0.9236 (reference 0.8156, 0.8903, 0.9073, 0.9110, 0.9168, 0.9178, 0.9213,
0.9263, 0.9284, 0.9278, 0.9291, 0.9299). Validation BCE 0.466 falling to 0.404 (reference 0.539 to
0.400). Ahead after epoch 1 (+0.033), behind from epoch 2 on (−0.004 to −0.013), still rising at the
last epoch.

Per label at ×4 against the reference, largest gain to largest loss: MCL +0.0031, Synovitis −0.0010,
Baker's −0.0012, Contusion −0.0024, Effusion −0.0049, ACL −0.0087, Medial Meniscus −0.0096, Lateral
OA −0.0097, Medial OA −0.0128, Lateral Meniscus −0.0165, PF OA −0.0182, Fracture −0.0212.

**What it shows:** diffuse labels (Synovitis, Baker's, Effusion, Contusion) are within 0.005 of
MambaVision-T; the losses sit on small, located structures (Fracture, the menisci, the OA
compartments) and on the stride-16 reads' targets, which seeds 2 (rate on the last blocks), 4
(feature statistics) and 5 (middle-read depth) address. The fast first epoch and the later slower
climb fit a model whose early blocks move as much as its late ones at one rate (seed 2).

## Seed 2, `layer-rate-decay-from-radimagenet-profile` (2026-10-15 03:27, reject)

4e-4 on block 12, 0.85 per block towards the stem. Training 2,714 s, peak 11.8 GB. ×4 0.929210
against 0.929220 (−0.000010), ×1 0.925191 (epoch 11) against 0.923642 (+0.001549), 80% interval of
×4 [−0.001221, +0.001196]. Not adopted: level at ×4.

Validation macro AUC per epoch: 0.8597, 0.8895, 0.9003, 0.9095, 0.9134, 0.9177, 0.9218, 0.9209,
0.9230, 0.9226, 0.9252, 0.9248; ahead of seed 1 at every epoch, by +0.011 at epoch 1, +0.006 through
epoch 6, then +0.001 to +0.003.

Per label at ×4 against seed 1: Lateral OA +0.0028, MCL +0.0021, Baker's +0.0017, Synovitis
+0.0013, Lateral Meniscus +0.0010, Medial OA +0.0005, ACL −0.0004, PF OA −0.0006, Effusion −0.0006,
Medial Meniscus −0.0011, Contusion −0.0034, Fracture −0.0035.

**What it shows:** the profile speeds learning (the lead is largest early) and lifts ×1, but at ×4
the gains on the OA, collateral and diffuse labels are offset by Contusion and Fracture, the two
bone-routed labels, the ones furthest from MambaVision-T after seed 1 (Fracture −0.021). Candidate
causes: the stem and early blocks now move at 5.7e-5 to 1.2e-4, so the low-level patch features that
bone signal depends on adapt less; or the last blocks at 4e-4 overfit the located bone labels by the
last epochs, where the lead narrowed.

## Seed 3, `anchors-by-weight-scale` (2026-10-15 04:13, reject)

Anchors 0.002 early and 0.00002 late. Training 2,712 s, peak 11.8 GB. ×4 0.926783 against 0.929220
(−0.002436), ×1 0.923575 (epoch 11) against 0.923642 (−0.000067), 80% interval of ×4
[−0.003585, −0.001292].

Validation macro AUC per epoch: 0.8502, 0.8826, 0.8934, 0.9038, 0.9109, 0.9123, 0.9184, 0.9195,
0.9209, 0.9221, 0.9236, 0.9234; within 0.004 of seed 1 at every epoch.

Per label at ×4 against seed 1: MCL +0.0032, Lateral Meniscus −0.0008, Baker's −0.0011, Medial OA
−0.0014, Medial Meniscus −0.0014, Synovitis −0.0019, PF OA −0.0021, Effusion −0.0023, Lateral OA
−0.0039, ACL −0.0039, Fracture −0.0061, Contusion −0.0075.

**What it shows:** level at ×1 and lower at ×4, again led by Contusion and Fracture, the two labels
seed 2 also lost. Both runs held the early layers closer to the pretrained weights (seed 2 by a lower
stem rate, seed 3 by a stronger early anchor), so the bone-routed labels point to the stem and first
blocks needing to move more, not less: the direction of the queued rate ideas 1 (stem kept at 2e-4,
top 4e-4) and 5 (the position embedding's own rate), and of the early anchor following the rate the
subagent noted. MCL gained in seeds 1, 2 and 3 alike.

## Seed 4, `feature-statistics-before-position-embedding` (2026-10-15 05:01, reject)

The augmentation on the patch projection, the position embedding added after it. Training 2,687 s,
peak 11.8 GB. ×4 0.928834 against 0.929220 (−0.000385), ×1 0.923995 (epoch 12) against 0.923642
(+0.000353), 80% interval of ×4 [−0.001339, +0.000601].

Validation macro AUC per epoch: 0.8530, 0.8830, 0.8942, 0.9044, 0.9083, 0.9133, 0.9200, 0.9205,
0.9217, 0.9234, 0.9239, 0.9240; ahead of seed 1 from epoch 4 on by +0.001 to +0.004.

Per label at ×4 against seed 1: MCL +0.0051, Lateral Meniscus +0.0037, Medial OA +0.0015, Baker's
+0.0003, Lateral OA +0.0001, ACL −0.0010, Medial Meniscus −0.0010, Synovitis −0.0013, Effusion
−0.0016, Contusion −0.0017, PF OA −0.0039, Fracture −0.0048.

**What it shows:** level at both multipliers. Fracture leads the losses again: seeds 2, 3 and 4 all
lost it against seed 1 (−0.0035, −0.0061, −0.0048) and all gained MCL (+0.0021, +0.0032, +0.0051).
What the three share against seed 1: each changes the early layers (lower stem rate, stronger early
anchor, the augmentation moved onto the patch projection), and each ran on the same code as seed 1
apart from that. Candidate causes: Fracture's bone-routed read depends on low-level patch features
that seed 1's unchanged early layers already fit best; or each change shifts the shared backbone
towards the soft-tissue reads (MCL, the menisci) at the cost of the bone route, which the opposite
signs of MCL and Fracture in all three runs fit. The rate ideas now running change the early
layers in the other direction (stem kept at 2e-4 or more), which separates the two.

## Rate idea 1, `stage-rates-4e-4-top-to-2e-4-stem` (2026-10-15 05:47, reject)

Backbone rate 4e-4 with stage decay 0.87: the final norm 4e-4, blocks 9 to 12 3.5e-4, down to the
stem at 2.0e-4. Forced on the user's message of 03:54. Training 2,711 s, peak 11.8 GB. ×4 0.929056
against 0.929220 (−0.000164), ×1 0.924465 (epoch 12) against 0.923642 (+0.000822), 80% interval of
×4 [−0.001284, +0.001028].

Validation macro AUC per epoch: 0.8564, 0.8846, 0.8934, 0.9052, 0.9092, 0.9144, 0.9166, 0.9214,
0.9210, 0.9228, 0.9237, 0.9245.

Per label at ×4 against seed 1: Fracture +0.0052, Lateral Meniscus +0.0025, Baker's +0.0023, Medial
Meniscus +0.0021, Effusion +0.0006, ACL +0.0003, Medial OA −0.0002, Synovitis −0.0014, PF OA −0.0024,
Lateral OA −0.0025, Contusion −0.0037, MCL −0.0046.

**What it shows:** level overall, but the label pattern of seeds 2 to 4 reverses: with the stem kept at
2e-4 and the upper blocks raised, Fracture gains (+0.0052) and MCL loses (−0.0046), where the three
runs that slowed or held the early layers lost Fracture and gained MCL. This supports the first
candidate cause noted under seed 4 (the bone-routed read needs the early layers to move) and points
at a trade between the bone route and the collateral-ligament read through the shared upper blocks:
the runs that raised the upper-block rate (seed 2, this one) are the two that lost MCL or gained it
least among the rate changes. Contusion lost in all four runs (−0.0034, −0.0075, −0.0017, −0.0037).

## Rate idea 2, `backbone-held-for-the-head-epoch` (2026-10-15 06:35, reject)

Backbone rate zero in epoch 1 while the heads fit, then its warm-up and linear decay. Training 2,714 s,
peak 11.8 GB. ×4 0.929279 against 0.929220 (+0.000060), ×1 0.923615 (epoch 12) against 0.923642
(−0.000027), 80% interval of ×4 [−0.001331, +0.001445]. Not adopted: not strictly above at ×1.

Validation macro AUC per epoch: 0.8262, 0.8692, 0.8888, 0.9012, 0.9055, 0.9116, 0.9173, 0.9195,
0.9201, 0.9220, 0.9233, 0.9236; −0.022 after the frozen epoch, level from epoch 6.

Per label at ×4 against seed 1: Lateral OA +0.0049, Medial OA +0.0031, Synovitis +0.0025, MCL
+0.0023, PF OA +0.0008, ACL +0.0002, Lateral Meniscus +0.0002, Effusion −0.0003, Medial Meniscus
−0.0008, Baker's −0.0015, Fracture −0.0047, Contusion −0.0058.

**What it shows:** level overall; holding the backbone first lifts the three OA labels and Synovitis
and again costs Fracture and Contusion, the pattern of the runs that slowed the early layers (seeds
2 to 4). Across the five decided changes the two bone-routed labels and MCL move in opposite
directions, and the run that kept the early layers fastest (rate idea 1) is the only one that gained
Fracture. Contusion has now lost in all five.

## Rate idea 3, `top-down-backbone-warm-up` (2026-10-15 07:23, provisional accept, the new incumbent)

Each backbone stage group reaches its peak rate one epoch after the group above it: the last stage at
the end of epoch 1, the stem at epoch 5 (new `optimizer.backbone_warmup_epochs_per_stage`, 1). Training
2,737 s, peak 11.8 GB. ×4 0.929641 against 0.929220 (+0.000422), ×1 0.924256 (epoch 12) against
0.923642 (+0.000614), 80% interval of ×4 [−0.000555, +0.001392]. Strictly above seed 1 at both
multipliers; adopted by the runtime under the shared rule. Still 0.008160 below the MambaVision-T
reference at ×4.

Validation macro AUC per epoch: 0.8610, 0.8850, 0.8934, 0.9050, 0.9092, 0.9141, 0.9183, 0.9197,
0.9204, 0.9214, 0.9234, 0.9243; +0.0125 at epoch 1, +0.002 through epoch 6, then within +0.002.

Per label at ×4 against seed 1: Baker's +0.0025, Lateral Meniscus +0.0017, ACL +0.0011, Medial OA
+0.0009, Synovitis +0.0009, Medial Meniscus +0.0008, MCL +0.0006, Effusion +0.0000, Fracture −0.0001,
PF OA −0.0003, Lateral OA −0.0008, Contusion −0.0024.

**What it shows:** the first change that keeps Fracture and MCL both level while gaining on most
labels: the early layers still reach their full rate, only later (unlike seeds 2 to 4 and rate idea
2, which slowed or held them), and the upper blocks are not raised (unlike rate idea 1). Contusion
lost again, the sixth run of six. The remaining rate ideas 4 and 5 and seeds 5 and 6 now build on
this incumbent.

## Rate idea 4, `backbone-rate-1-4e-4-by-relative-step` (2026-10-15 08:11, reject)

One backbone rate of 1.4e-4 on the top-down warm-up incumbent. Training 2,728 s. ×4 0.929187 against
0.929641 (−0.000455), ×1 0.923724 against 0.924256 (−0.000532), 80% interval of ×4
[−0.001361, +0.000497].

Validation macro AUC per epoch: 0.8550, 0.8847, 0.8966, 0.9047, 0.9076, 0.9137, 0.9176, 0.9200,
0.9200, 0.9221, 0.9229, 0.9237; within 0.006 of the incumbent at every epoch.

Per label at ×4 against the incumbent: Fracture +0.0034, MCL +0.0021, PF OA +0.0015, Synovitis
+0.0010, Lateral OA +0.0008, Contusion +0.0005, Effusion −0.0012, Medial OA −0.0014, Medial Meniscus
−0.0017, ACL −0.0017, Lateral Meniscus −0.0036, Baker's −0.0051.

**What it shows:** a lower rate is level to slightly below; with rate ideas 1 and 4 and seed 2, rates
from 1.4e-4 to 4e-4 and two profiles over depth all land within 0.0005 of each other at ×4, so the
level of the backbone rate is not what holds RadioDINO 0.008 below MambaVision-T. On the user's
message of 08:12 a subagent is looking for a defect in the adapter or the pipeline.

## Diagnosis of the gap to MambaVision-T (subagent, user's message of 2026-10-15 08:12)

No defect found in the adapter or the pipeline:

- **Weights:** the incumbent's anchor buffers equal the RadioDino-s16 file exactly (difference 0.0);
  they differ from ImageNet DINO by 0.25 to 0.35.
- **Normalisation:** ImageNet mean and std, as in the model card's own example.
- **Config:** LayerNorm eps 1e-6 is DINO's own value (the 1e-12 is the Hugging Face conversion's);
  qkv bias present; bf16 as for MambaVision-T.
- **Geometry:** origins 7.5 and 15.5, strides 16 and 32, derived from the layers; MambaVision-T's final
  map is 7×7 too.
- **CLS token:** small effect, mean cosine of the final patch tokens with and without it 0.979 (lowest
  0.836); attention to it 6–16% in blocks 1–3, at most 0.6% from block 5.

**A third of the ×4 gap is the mask decoder.** The reference trained with the mask decoder's
supervision, removed by 9a93087. `mamba-silver-fold-2`, MambaVision-T on the same config and fold
without it, scores ×4 0.935160 (−0.0026 from the reference), ×1 0.9301. Against it, RadioDINO's gap
is ×4 −0.0055 and ×1 −0.0058. Seed 1's per-label comparison with the reference carries that change:
against the decoder-free run Contusion is +0.0046 and Fracture −0.0061. The comparisons between
RadioDINO runs (seed 2 on) are unaffected.

**The rest sits on the five soft-tissue-routed labels.** Lateral Meniscus −0.0154, PF OA −0.0154,
Medial OA −0.0102, Lateral OA −0.0077 and Medial Meniscus −0.0076 carry 85% of the summed deficit
against the decoder-free MambaVision-T; the other seven average −0.0015. Anatomy share error 0.0789
against 0.0760 (Swin-T 0.0706), meniscus validation loss 0.5955 against 0.5876 (Swin-T 0.5755).

**The adapter's one structural difference from both models that reached 0.937:** MambaVision-T and
Swin-T reach the 7×7 final map through a learned step (a stride-2 convolution, patch merging);
RadioDINO uses a fixed 2×2 average, which removes 41% of the spatial variation of its final tokens.

**Under-fitting, not over-fitting:** training loss falls 0.094 from epoch 2 to 12 against 0.114 for
MambaVision-T, validation loss ends level (0.404 against 0.401); stem and block-1 gradients are 7 to 17
times smaller than MambaVision-T's, and under the early anchor the stem and block 1 lose 40% and 54% of
their peak drift by the last epoch (19% and 35% for MambaVision-T).

Two forced seeds from this ([keeper-ideas-fixes.yaml](keeper-ideas-fixes.yaml)), queued after seed 5:
`final-read-by-learned-two-by-two-merge` (a learned 2×2 stride-2 merge from 384 to 768 channels
started as the average) and `early-anchor-by-gradient-scale` (early anchor 0.00013). Outside the
contract, for the user: restoring the mask decoder's supervision (about +0.0026 at ×4 on
MambaVision-T) and judging the campaign against `mamba-silver-fold-2`.

## Rate idea 5, `position-embedding-rate-by-weight-scale` (2026-10-15 09:00, reject)

The position embedding at 1.86 times the backbone rate (3.7e-4 at peak), on the top-down warm-up
incumbent. Training 2,740 s. ×4 0.929093 against 0.929641 (−0.000549), ×1 0.924444 against 0.924256
(+0.000188), 80% interval of ×4 [−0.001443, +0.000340].

Validation macro AUC per epoch: 0.8585, 0.8851, 0.8954, 0.9021, 0.9103, 0.9117, 0.9202, 0.9209,
0.9211, 0.9231, 0.9235, 0.9244.

Per label at ×4 against the incumbent: MCL +0.0059, Lateral OA +0.0038, Contusion −0.0001, Effusion
−0.0003, Medial Meniscus −0.0004, Fracture −0.0010, PF OA −0.0013, Medial OA −0.0014, Lateral
Meniscus −0.0027, Synovitis −0.0030, ACL −0.0031, Baker's −0.0032.

**What it shows:** level at ×1, slightly below at ×4; the five rate and schedule ideas end within
0.0006 of each other at ×4 apart from the adopted top-down warm-up (+0.0004). With the diagnosis, the
gap to MambaVision-T sits in the soft-tissue-routed labels and the final read, not in the rates.

## Seed 5, `middle-read-at-block-10` (2026-10-15 09:48, reject)

The stride-16 middle read after block 10 (stages 1, 2, 7, 2) on the top-down warm-up incumbent.
Training 2,742 s. ×4 0.929126 against 0.929641 (−0.000515), ×1 0.924548 (epoch 11) against 0.924256
(+0.000292), 80% interval of ×4 [−0.001511, +0.000518].

Validation macro AUC per epoch: 0.8569, 0.8860, 0.8954, 0.9039, 0.9100, 0.9122, 0.9185, 0.9194,
0.9220, 0.9239, 0.9245, 0.9245.

Per label at ×4 against the incumbent: MCL +0.0019, Contusion +0.0018, Lateral OA +0.0012, Medial OA
+0.0003, PF OA +0.0001, Lateral Meniscus −0.0005, Synovitis −0.0005, Medial Meniscus −0.0015, Baker's
−0.0017, Effusion −0.0018, Fracture −0.0020, ACL −0.0036.

**What it shows:** the deeper middle read leaves the menisci and OA labels, the ones the diagnosis
found behind MambaVision-T, where they were (within 0.0015); the read's depth is not what holds them.

## `final-read-by-learned-two-by-two-merge` blocked at review (2026-10-15 09:48)

The orchestrator blocked the learned merge: "No linked authorization permits the new learned readout
structure" (the contract's out-of-bounds list names new readout structure). The candidate was
correctly built (average-initialised merge, projection columns kept, geometry 15.5/32). Left for the
user to authorise; the keeper does not widen the contract.

## Fix 2, `early-anchor-by-gradient-scale` (2026-10-15 10:36, reject)

Early anchor 0.00013 on the top-down warm-up incumbent. Training 2,727 s. ×4 0.928977 against 0.929641
(−0.000664), ×1 0.922905 against 0.924256 (−0.001351), 80% interval of ×4 [−0.001559, +0.000219].

Validation macro AUC per epoch: 0.8592, 0.8849, 0.8940, 0.9065, 0.9067, 0.9118, 0.9182, 0.9199,
0.9207, 0.9220, 0.9223, 0.9229.

Per label at ×4 against the incumbent: Fracture +0.0020, MCL +0.0017, Lateral Meniscus +0.0012,
Lateral OA +0.0009, Medial OA +0.0003, Synovitis −0.0011, Effusion −0.0012, Contusion −0.0015,
Baker's −0.0021, PF OA −0.0023, Medial Meniscus −0.0028, ACL −0.0031.

**What it shows:** freeing the stem and block 1 from the anchor is slightly below at both multipliers;
with seed 3 (anchors doubled, −0.0024), the inherited early anchor 0.001 sits between the two tried
values and both directions lose. The gap's remaining lead from the diagnosis is the blocked learned
final merge. Forced seeds done at 10:36; the researcher took over.

## Learned final merge, `final-read-by-learned-two-by-two-merge` (2026-10-15 12:10, reject)

Run after the scope note; the campaign stopped gracefully around it at the user's order. Training
2,745 s. ×4 0.927756 against 0.929641 (−0.001885), ×1 0.922937 (epoch 11) against 0.924256
(−0.001319), 80% interval of ×4 [−0.003148, −0.000671].

Per label at ×4 against the incumbent: Fracture +0.0060, Lateral Meniscus +0.0054, Lateral OA
+0.0006, Baker's −0.0012, Medial Meniscus −0.0018, Effusion −0.0026, Contusion −0.0031, Synovitis
−0.0037, Medial OA −0.0042, ACL −0.0054, MCL −0.0058, PF OA −0.0070.

**What it shows:** a learned step to stride 32 moves single labels both ways but loses overall;
the stride-32 read is not where the missing detail is. With the middle read at block 10 also level,
re-reading RadioDINO's own tokens does not close the gap; the next run adds detail the tokens lack.

## Spatial prior queued first (2026-10-15 12:12)

`convolutional-spatial-prior-injection` ([keeper-ideas-spatial-prior.yaml](keeper-ideas-spatial-prior.yaml)),
the user's choice: written and CPU-verified by the keeper as [spatial-prior.patch](spatial-prior.patch)
(outputs equal the incumbent's with the gates at zero, branch weights equal the RadImageNet file,
ResNet layers at 5e-5). Imported forced while stopped; the blockwise idea's claim released (it had
not been developed) and the spatial prior put at the queue head (state backed up as
`state.before-spatial-prior-first-20261015.json`).

## Spatial prior, first attempt: invalid on the budget (2026-10-15 12:27)

`convolutional-spatial-prior-injection` with the pretrained ResNet layers training at 5e-5: all four
gates opened after epoch 1 (0.0014, 0.0010, 0.0007, 0.0005), validation 0.8587 against 0.8610, but GPU
memory reached 22 GB and epoch 2 took about 340 s against 264 s for epoch 1, consistent with the known
spill into shared memory above about 19 GB; projected 4,072 s against the 3,417 s ceiling, stopped as
invalid. The developer's repair was blocked by the seed's own "apply exactly, change nothing" wording.

Replaced by `convolutional-spatial-prior-frozen-branch`
([keeper-ideas-spatial-prior-frozen.yaml](keeper-ideas-spatial-prior-frozen.yaml),
[spatial-prior-frozen.patch](spatial-prior-frozen.patch)): the pretrained ResNet layers frozen
(no_grad, BatchNorm in eval), the downsampling convolution, norm, projections and gates training;
keeper's CPU checks as before (outputs equal the incumbent's with the gates at zero). Campaign stopped
gracefully at 12:35 to put it first after the blockwise run.

## Researcher's `blockwise-top-down-warm-up` (2026-10-15 13:14, reject)

The top-down warm-up per block instead of per stage. ×4 −0.000187, ×1 +0.000077 against the incumbent:
level. Campaign stopped gracefully after it; the frozen spatial prior put at the queue head (the
researcher's block-ten idea's undeveloped claim released; state backed up as
`state.before-spatial-prior-frozen-first-20261015.json`).

## Frozen spatial prior: invalid on the budget, repaired to stride 4 (2026-10-15 13:35)

`convolutional-spatial-prior-frozen-branch` (frozen stem to layer2): memory fixed (13.3 GB against 22),
full GPU speed, epoch 1 at 0.8580 against 0.8610, but projected 3,513 s against the 3,417 s ceiling
(+2.8%, over the 2% tolerance); stopped as invalid. Its automatic repair was pointed at
[spatial-prior-stride-four.patch](spatial-prior-stride-four.patch): frozen stem and layer1 only (stride
4, 256 channels) brought to the token grid by a 5×5 stride-4 convolution, the new adapter parameters
in the stem's rate group, anchor registration skipping frozen weights; keeper's CPU checks as before.

By the user's order the queue holds only the locality line: the spatial-prior repair, then
`overlapping-patch-embedding` ([keeper-ideas-overlap.yaml](keeper-ideas-overlap.yaml)), then
`depthwise-convolution-in-block-mlps` ([keeper-ideas-local-mlp.yaml](keeper-ideas-local-mlp.yaml)).
Retired: block-ten-read-with-fixed-optimizer-ownership, feature-statistics-preserving-token-moments,
adam-second-moment-memory-0-95, global-gradient-clip-0-3 (contract.md, Current line).

## Spatial prior, stride 4 (2026-10-15 14:36, reject)

`convolutional-spatial-prior-frozen-branch` repaired to the frozen stem and layer1: training 3,263 s
(inside the ceiling), 13.6 GB. ×4 0.929497 against 0.929641 (−0.000144), ×1 0.923938 (−0.000318),
80% interval of ×4 [−0.000997, +0.000690].

Per label at ×4 against the incumbent: Lateral OA +0.0048, Fracture +0.0018, Medial OA +0.0004, MCL
+0.0004, PF OA +0.0002, Contusion +0.0001, Medial Meniscus +0.0000, Effusion −0.0014, ACL −0.0018,
Baker's −0.0019, Lateral Meniscus −0.0019, Synovitis −0.0024.

**The gates barely opened:** mean absolute gamma per stage 0.0003 to 0.0002 after epoch 1, at most
0.0007 (stage 0) at the end. The prior entered the tokens at about a thousandth of its scale, so the run
measures the incumbent more than the mechanism. Candidate causes: zero-initialised gates at the stem
group's rate 2e-4 reaching full rate only around epoch 5 under the top-down warm-up; a gradient on the
gates that changes sign. Next in the line: the gates started at a nonzero value (LayerScale-style, e.g.
0.1) or at the head's rate, so the prior takes part from epoch 1.

## Overlapping patch embedding (2026-10-15 15:26, reject)

`overlapping-patch-embedding`: 32×32 stride-16 projection started from the pretrained kernel. ×4
0.928038 against 0.929641 (−0.001604), ×1 0.923088 (−0.001168), 80% interval of ×4
[−0.002746, −0.000504].

Per label at ×4 against the incumbent: Synovitis +0.0024, Contusion +0.0019, Lateral OA +0.0001, MCL
−0.0000, Lateral Meniscus −0.0003, Medial OA −0.0010, Medial Meniscus −0.0017, Effusion −0.0020,
Baker's −0.0031, Fracture −0.0032, PF OA −0.0060, ACL −0.0062.

Validation behind the incumbent at every epoch, most at epoch 4 (−0.0075) as the stem group reached
full rate under the top-down warm-up, then −0.0012 at the end. Border weight RMS 0.0004 after epoch 1,
peak 0.0017 at epoch 6, 0.0012 at the end, against the centre kernel's 0.046: the border reached about
3.5% of the kernel's scale and was pulled back in the second half, when the early anchor (0.001, on
stem.proj, zeros as its origin for the border) outweighed the decaying rate. The five target labels did
not gain (PF OA −0.0060, Medial Meniscus −0.0017, Medial OA −0.0010).

## Depthwise convolution in the block MLPs (2026-10-15 16:22, reject)

`depthwise-convolution-in-block-mlps`: training 3,186 s, 16.1 GB. ×4 0.929615 against 0.929641
(−0.000026), ×1 0.925436 (epoch 12) against 0.924256 (+0.001180), 80% interval of ×4
[−0.000926, +0.000830]. Not adopted: level at ×4.

Per label at ×4 against the incumbent: MCL +0.0041, Fracture +0.0034, Lateral OA +0.0032, Synovitis
+0.0006, Medial Meniscus +0.0006, Contusion −0.0003, Lateral Meniscus −0.0004, Baker's −0.0006,
Effusion −0.0010, Medial OA −0.0013, PF OA −0.0027, ACL −0.0058.

Gate mean absolute value per epoch rising steadily from 0.0015 to 0.0085 (against at most 0.0007 for
the spatial prior's gates): the depthwise path was used, more each epoch, and still growing at the end.
Ahead of the incumbent from epoch 7 on in validation (+0.0021 at epoch 7, +0.0012 at epoch 12), the
best ×1 of the locality line; at ×4 the gains on MCL, Fracture and Lateral OA are offset by ACL
(−0.0058) and PF OA (−0.0027).

## Closed (2026-10-15 ~17:00 CEST, user's order)

Stopped with `--abort` during `spatial-prior-direct-zero-projection` (epoch 6, level with the
incumbent), which is not scored. Final incumbent `top-down-backbone-warm-up` (×4 0.929641, ×1
0.924256), 0.008160 below the MambaVision-T reference at ×4 and about 0.0055 below a decoder-free
MambaVision-T (`mamba-silver-fold-2`, ×4 0.935160). No code adopted: the adapter
([radiodino-adapter.patch](radiodino-adapter.patch), the worktree's diff against main) and every
candidate stay as records only. The worktree, branch and runtime artifacts were removed. The next
campaign tries other backbone families (SAM2.1 Hiera-T, PVTv2-b2, HRNet-W18 and others).
