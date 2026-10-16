# Backbone families: run notes

Kept by the campaign keeper, one section per run. ×1 is validation macro AUC at the epoch; the
reference is `mamba-silver-fold-2` at the same epoch.

## 1. sam2-hiera-tiny (aborted at epoch 7 on the user's order, invalid)

SAM2.1 Hiera-T, rate 1e-4, anchors 0.01 and 0.0001, no padding. Adapter as specified: stage ends
0, 2, 9, 11; mapping origins 6 (stride 16) and 14 (stride 32) derived from the patch embedding and
query pooling and verified against the layers. 4 min 46 s per epoch (about 57 min for twelve,
inside the window).

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hiera-T ×1 | 0.7009 | 0.8365 | 0.8609 | 0.8783 | 0.8910 | 0.8962 | 0.9021 |
| Reference ×1 | 0.8217 | 0.8885 | 0.9065 | 0.9059 | 0.9164 | 0.9177 | 0.9234 |
| Gap | −0.1209 | −0.0520 | −0.0456 | −0.0276 | −0.0254 | −0.0215 | −0.0213 |

The user stopped it at epoch 7 (18:58 CEST) when the gap stopped closing; aborted, so no ×4 and no
decision.

- **Training loss** epoch 1: 0.810 against 0.738; the side head lagged most (0.59 against 0.43).
- **Token maps** at epoch 3: anatomy (bones) level or better (anatomy loss 0.699 against 0.718;
  kneecap overlap 0.709 against 0.652, fibula 0.714 against 0.682, centre error front 2.42 against
  2.86 mm). Meniscus maps behind: loss 0.737 against 0.646; medial and lateral meniscus overlap
  0.408 and 0.407 against 0.541 and 0.498; medial and lateral tibial cartilage 0.389 and 0.439
  against 0.501 and 0.537.
- **Per label at epoch 7** (×1, Hiera-T against the reference at epoch 7), largest gain to largest
  loss: Baker's −0.0018, Contusion −0.0048, MCL −0.0056, Synovitis −0.0112, Effusion −0.0146,
  Fracture −0.0153, Medial OA −0.0158, Medial Meniscus −0.0174, ACL −0.0295, Lateral OA −0.0374,
  Lateral Meniscus −0.0423, PF OA −0.0600.
- **Map scale** on a CPU test input: stride-16 map RMS 0.35 and stride-32 0.17, against 1.7 and 0.19
  for the ResNet-50 and 9.9 and 0.85 for the in12k ConvNeXt-T; no final norm in SAM2 Hiera.

Candidate causes (guesses): the loss sits on the thin structures (menisci, cartilage, OA
compartments, ACL), where Hiera has no convolution after the patch embedding and windowed
attention early; SAM2 trained at 1024 px, where those structures cover about 4.6 times more
pixels than at 224 px; the weak stride-16 map; the low rate and strong early anchor slowing the
early stages. Bone-scale localisation worked, so the input, adapter and mapping are not the cause.

## 2. hrnet-w18 (aborted after epoch 1 by the keeper, over budget)

HRNet-W18, rate 5e-5, anchors 0.005 and 0.0003, no padding (BatchNorm); feature-statistics
augmentation on the stride-8 branch; reads from the per-branch incre heads at strides 16 and 32
(512 and 1024 channels). Epoch 1 took 7 min 52 s with the GPU at 72 to 100% and 18.7 GB, 321 W in
P2 (no spill to Windows RAM): twelve epochs project to about 95 min against the 68 min upper bound,
so the runtime's projection check would have failed it after epoch 2. Aborted at 19:08 CEST
after epoch 1, which scored ×1 0.7576 against the reference's 0.8217 (−0.0641; Hiera-T was
−0.1209 there). Cause of the cost: the stride-4 and stride-8 streams run through every stage with
BatchNorm and fusion at full resolution, memory-bound work that its 4.3 GMAC do not show.

## 3. pvt-v2-b2 (harness failure, relaunched as pvt-v2-b2-retry)

Launched at 19:09 CEST, 30 s after the HRNet abort; the first training batch failed with `CUDA
driver error: device not ready` in a PVTv2 attention projection. The developer's repair found no
defect and edited nothing. A forward and backward pass of `pvt_v2_b2.in1k` on the GPU at 48, 37 and
64 images in bfloat16 ran cleanly (4.3 to 5.8 GB), so the failure was the GPU still releasing the
killed process. Requeued unchanged as the forced seed `pvt-v2-b2-retry`.

## 4. focalnet-tiny-srf (stopped by the runtime after epoch 2, projected over budget; repair requested)

FocalNet-T SRF on the recipe's settings, image counts padded to 16, final norm on the stride-32
map, `layers.0.` under the early anchor. ×1 0.8660 after epoch 1 (reference 0.8217, +0.0443) and
0.8903 after epoch 2 (reference 0.8885, +0.0018): the strongest start of the campaign. Epoch 1 took
6 min 47 s and epoch 2 5 min 50 s; the projection, 4,250 s, exceeded the 4,090 s bound by more than
the 2% tolerance (it needed at most 5 min 42 s per epoch). 23 GB on the GPU at 332 W in P2 (no
spill). Sent back to the developer for a cost repair within the idea.

## 5. rdnet-tiny (harness failure, relaunched as rdnet-tiny-retry)

Launched 7 s after the runtime killed the FocalNet run; the first convolution failed with the same
`CUDA driver error: device not ready` as PVTv2. Cause: the runtime's GPU check lists compute
processes with nvidia-smi, which lists none under WSL, so a training could start while the killed
one still held about 20 GB. Keeper fix in the worktree (`autoresearch/runtime/gpu.py`): the GPU also
counts as busy until its used memory is below 2,048 MiB; the job runner then returns GPU_BUSY and
the training waits.

## 6. focalnet-tiny-srf #2 (speed repair; provisionally accepted, the incumbent)

The cost repair ran FocalNet's 1×1 convolutions as linear layers over the channels-last view (same
weights and parameter names) and padded without a contiguous copy: epoch 1 6 min 20 s (from 6 min
47 s), later epochs 5 min 28 s. Training 4,067 s against the 4,090 s bound; peak 23.0 GB allocated
(332 W in P2, no spill).

×4 0.938816 against the reference's 0.935160: +0.0037 (80% interval +0.0017 to +0.0057, 95%
+0.0007 to +0.0066); ×1 0.931536 at epoch 12 (best 0.933241 at epoch 10) against 0.930081; soft ×4
+0.0011. Ahead of the reference at every epoch (+0.051 after epoch 1, +0.003 to +0.012 after).
Above MambaVision-T with the mask decoder (0.937801) and Swin-T (0.937176), 0.0008 below the
ResNet-50 (0.939634).

Per label at ×4, largest gain to largest loss: Fracture +0.0139, Lateral OA +0.0139, Contusion
+0.0064, MCL +0.0046, Lateral Meniscus +0.0042, Synovitis +0.0024, Baker's +0.0016, Medial Meniscus
+0.0016, PF OA +0.0014, ACL −0.0008, Medial OA −0.0010, Effusion −0.0045. The gains sit on the
labels where Hiera-T lost most (Lateral OA, Lateral Meniscus): FocalNet's gated depthwise context at
every stage keeps the local detail that Hiera's attention-only early stages lacked.

## 7. dinov3-convnext-tiny (aborted at epoch 2; relaunched at ten epochs)

Built on the FocalNet incumbent's code, the ConvNeXt adapter with the DINOv3 checkpoint, rate 5e-5,
anchors 0.01 and 0.0002. Epoch 1 5 min 30 s (the local 3090 speed of ConvNeXt-T: 4.7 min per
epoch of training measured on 2026-10-13; the 1,534 s ten-epoch silver ConvNeXt-T ran on the rented
RTX PRO 6000), 20.6 GB. ×1 0.8203 after epoch 1 (FocalNet 0.8729, reference 0.8217) and 0.8757
after epoch 2 (FocalNet 0.8943, −0.0187). Aborted at 21:02 CEST when the user made ten epochs the
default (twelve was MambaVision-T's recipe); relaunched as `dinov3-convnext-tiny-10ep`. The other
queued seeds, and the PVTv2 relaunch whose candidate had been built with twelve, now train ten.
The in12k ConvNeXt-T on its own ten-epoch recipe (`convnext-silver-only-imagenet-fold-2`) scored
×4 0.936123 on fold 2.

## 8. dinov3-convnext-tiny-10ep (rejected)

Ten epochs, rate 5e-5, anchors 0.01 and 0.0002. ×4 0.934199 against the FocalNet incumbent's
0.938816: −0.0046 (80% interval −0.0065 to −0.0029); ×1 0.927793 at epoch 10 against 0.933241
(−0.0054), and below the MambaVision reference (×4 0.935160). Behind FocalNet-T at every epoch:
−0.077 after epoch 1, −0.017 at epochs 2 and 3, closing to −0.004 at epochs 8 and 9. About 5 min
per epoch. Below the in12k ConvNeXt-T on its own ten-epoch recipe (×4 0.936123, fold 2).

## 9. The "device not ready" failures: out of GPU memory (correction)

The PVTv2 relaunch failed again at 22:02 CEST on its first batch with the GPU idle at 290 MB before
launch, so the earlier explanation (a training launched while a killed one still held memory) was
wrong; the GPU-wait fix in `runtime/gpu.py` stays as a harmless guard. Measured on the 3090 with the
backbone alone, 256 images (the mean training batch is 253 three-slice inputs), bf16, channels_last,
forward and backward: PVTv2-b2 22.3 GB, RDNet-T 20.4 GB, FocalNet-T 17.2 GB (which trained at a 23.0 GB
peak with the rest of the model). PVTv2 and RDNet therefore exceed the 24 GB card in training; under
WSL the spill surfaces as "CUDA driver error: device not ready". Activation checkpointing of PVTv2's
stage 0 brings it to 12.7 GB (0.401 to 0.417 s per 256 images) and of RDNet's dense stages 0 and 1 to
10.0 GB (0.471 to 0.459 s). The relaunches `pvt-v2-b2-retry` and `rdnet-tiny-retry` were dropped and
replaced by `pvt-v2-b2-checkpointed` and `rdnet-tiny-checkpointed`, ten epochs.

## 10. pvt-v2-b2-checkpointed (rejected)

Stage 0 under activation checkpointing in training, ten epochs, the recipe's settings. Trained
3,594 s at a 16.3 GiB peak (17.5 GiB predicted by the preflight run afterwards). ×4 0.933099 against
the FocalNet incumbent's 0.938816: −0.0057 (80% interval −0.0076 to −0.0039); ×1 0.9255 at epoch 10,
best 0.9270 at epoch 9 (FocalNet 0.9332). Within −0.001 to −0.005 of FocalNet-T at every epoch.

## 11. Preflight before every training (from 23:30 CEST)

`scripts/tools/preflight_training.py` wraps every experiment training: the candidate's real
training loop for 60 batches and 20 validation batches under a 22.5 GiB memory cap, peak memory
extrapolated to a batch 8% above the largest seen, and the training time projected as 1.13 × epochs ×
(442 median steps + validation) plus the first-epoch excess seen in the preflight; a failing
candidate returns to the developer with the numbers instead of training. Calibrated against the real
runs: FocalNet-T 12 epochs projects 4,246 s (real 4,067 s, rejected by the margin), PVTv2-b2 3,883 s
(real 3,594 s), RDNet-T 4,194 s (real projection 4,232 s, rejected). The first calibration, ×1.07
without the first-epoch excess, let RDNet-T through at 3,866 s.

## 12. rdnet-tiny-checkpointed (invalid: over the time bound after epoch 2)

Dense stages 0 and 1 checkpointed, its 1×1 convolutions as linear layers (the developer's repair
after the first preflight projected 4,163 s). Peak 13.8 GiB predicted. Epoch 1 560 s (cuDNN planning
for the depthwise convolutions on new image counts) and epoch 2 408 s; the runtime projected 4,221 s
against 4,090 s (+2% tolerance) and stopped it. ×1 0.8693 after epoch 1 (FocalNet 0.8729, −0.0036).
The preflight had projected 3,866 s with the ×1.07 calibration, since replaced. Back to the developer
for a cost repair.

## 13. focalnet-tiny-srf-10ep (rejected, level)

The incumbent at ten epochs (user's request). ×4 0.938154 against 0.938816: −0.0007 (80% interval
−0.0022 to +0.0009); ×1 0.932380 at epoch 10 against 0.933241 at epoch 10 of twelve. Within ±0.003 of
the twelve-epoch curve at every epoch (+0.003 after epoch 2, −0.001 after epoch 6, level from epoch
8). About five sixths of the twelve-epoch time. Level with the twelve-epoch incumbent at ×4 inside
its interval, not above it, so the runtime kept twelve epochs.

## 14. rdnet-tiny-checkpointed (blocked after the repair limit)

The developer's second cost repair cached the matrix-multiply and convolution outputs inside the
checkpointed stages; the orchestrator blocked it because the idea had used its one bounded repair
(repair count 2), not on evidence against RDNet: its ×1 trailed FocalNet-T by only 0.0036 after
epoch 1. Ways to bring it inside the bound: checkpoint dense stage 0 only (less recompute; memory
between the 13.8 GiB measured and the unbounded 20.4 GiB backbone), or nine epochs if the user wants.
The preflight's report now treats the expected validation completeness assertion of its truncated
pass as normal instead of listing it as an error, which the orchestrator had raised as a concern.

## 15. davit-tiny (rejected; above the MambaVision reference)

DaViT-T on the recipe's settings, ten epochs, padding to 16. Preflight projected 3,661 s and 20.0
GiB; trained 3,343 s at a 19.6 GiB peak. ×4 0.936669 against the FocalNet incumbent's 0.938816:
−0.0021 (80% interval −0.0038 to −0.0005), and +0.0015 above the MambaVision reference (0.935160).
×1 0.930357 at epoch 10 (FocalNet 0.933241): level or ahead of FocalNet-T from epoch 1 to 8 (+0.005
after epochs 2 and 5), behind only at epochs 9 and 10. Per label at ×4 against FocalNet-T, largest
gain to largest loss: Synovitis +0.0029, Effusion +0.0021, ACL +0.0013, Contusion +0.0004, MCL 0.0000,
Medial OA −0.0004, PF OA −0.0017, Medial Meniscus −0.0027, Baker's −0.0054, Lateral Meniscus −0.0065,
Fracture −0.0066, Lateral OA −0.0091. Its gains (Effusion, Synovitis, ACL) fall on labels where
FocalNet-T lost against MambaVision (Effusion −0.0045, ACL −0.0008), a candidate for the blend.

## 16. edgenext-base (rejected; above the MambaVision reference)

EdgeNeXt-B (ImageNet-21k), the recipe's settings, ten epochs, padding to 16. ×4 0.936033 against the
FocalNet incumbent's 0.938816: −0.0028 (80% interval −0.0046 to −0.0011), +0.0009 above the
MambaVision reference. ×1 0.929372 at epoch 10 (FocalNet 0.933241); level or ahead of FocalNet-T
from epoch 1 to 8 (+0.006 after epoch 2), behind at epochs 9 and 10 as DaViT-T was. Per label at ×4
against FocalNet-T, largest gain to largest loss: Effusion +0.0033, Baker's +0.0009, Medial OA
+0.0009, ACL −0.0001, Synovitis −0.0007, Medial Meniscus −0.0022, Contusion −0.0023, MCL −0.0038,
Lateral Meniscus −0.0047, PF OA −0.0048, Lateral OA −0.0081, Fracture −0.0118. About 4 min per
epoch, the cheapest family.

## 17. efficientvit-b2 (blocked by the preflight: out of memory)

Real-row BatchNorm padding as specified (checked: batch statistics and running-statistics update
from the real rows through native_batch_norm, padded rows normalised without gradient). The preflight
ran out of memory at its 52nd training batch under the 22.5 GiB cap (22.29 GiB allocated), so it never
trained and is not an invalid run; the idea had used its one repair and was blocked. Backbone alone at
256 images: 18.5 GB and 0.306 s; with stages 0 and 1 checkpointed 8.6 GB and 0.353 s. Requeued as
`efficientvit-b2-checkpointed`, whose recompute pass must not update the BatchNorm running statistics a
second time.

## 18. wtconvnext-tiny (blocked by a harness failure; relaunched)

The developer vendored the official Triton WTConv kernels and loaded the official checkpoint
strictly (356 tensors). The review asked for backward parity; the keeper measured it on the GPU
against the official layer (prior-work.md, below 6.4e-7 relative for every gradient with TF32 off).
The repaired candidate's training job died in 17 s before the preflight: Triton writes its kernel
cache to ~/.triton, read-only in the training sandbox (writable: the experiment and ~/.cache). The
idea had used its repair and was blocked. Keeper fix in the worktree (`supervisor/experiments.py`):
training jobs get TRITON_CACHE_DIR=~/.cache/triton. Requeued as `wtconvnext-tiny-relaunch`, which
copies the reviewed candidate's files unchanged.

## 19. efficientvit-b2-checkpointed (rejected; the best challenger at ×4)

Real-row BatchNorm padding, stages 0 and 1 checkpointed with the recompute pass leaving the running
statistics alone (checked in the candidate), the recipe's rate, anchors 0.001 and 0.00003, ten
epochs. Preflight 3,684 s and 12.2 GiB. ×4 0.937280 against the FocalNet incumbent's 0.938816:
−0.0015 (80% interval −0.0032 to +0.0002), +0.0021 above the MambaVision reference; the closest
challenger so far. ×1 0.931092 at epoch 10 (FocalNet 0.933241): +0.0125 after epoch 2, ahead or level
through epoch 7. Per label at ×4 against FocalNet-T, largest gain to largest loss: Medial OA +0.0061,
ACL +0.0039, Fracture +0.0037, Effusion +0.0018, PF OA +0.0008, Contusion +0.0002, Baker's −0.0007,
Medial Meniscus −0.0037, MCL −0.0053, Lateral Meniscus −0.0073, Lateral OA −0.0082, Synovitis −0.0096.
Gains on Medial OA, ACL, Fracture and Effusion; DaViT-T and EdgeNeXt-B also gained on Effusion.

## 20. wtconvnext-tiny-relaunch (stopped by the preflight twice; nine-epoch seed queued)

First preflight: 0.835 s per step, 4,936 s projected for ten epochs, 22.3 GiB. The developer's cost
repair rewrote the Triton kernels in blocks: 0.726 s per step (−13%), 4,183 s projected, 2.3% over
the 4,090 s bound, 22.5 GiB predicted peak; stopped before training both times, no GPU run wasted.
The idea used its repair. Queued `wtconvnext-tiny-9ep`: the repaired candidate unchanged at nine
epochs (about 3,770 s projected), epochs being a family setting.

## 21. wtconvnext-tiny-9ep (blocked on the recorded limits; requeued)

The developer twice refused to set nine epochs because the campaign's recorded limits still said ten
epochs and 4,420 steps (set by the keeper when ten became the default), and the reviewer blocked the
idea. In the range budget those fields are not enforced (the run limits take the epochs from the
config). Keeper fix in the worktree (`supervisor/context.py`): in a range budget the agents' context
omits epochs, optimizer steps, visits and triplets from the limits. The idea was requeued unchanged.

## 22. wtconvnext-tiny-9ep (setup crash after a passing preflight; repair)

Preflight passed (nine epochs, 0.718 s per step, 3,660 s projected, 22.5 GiB). The training then
failed in its setup, before epoch 1: train.py records the backbone's `pretrained_cfg`, which the custom
WTConvNeXt class did not define. Not an invalid run; the developer repairs it. The preflight now also
reads the backbone's pretrained config and hashes its state as train.py does, so such a setup error
stops the candidate in the preflight.

## 23. wtconvnext-tiny-9ep (rejected)

The repaired WTConvNeXt-T (official Triton WTConv, GPU-verified parity, `pretrained_cfg` added) at nine
epochs. Trained 3,375 s at a 21.9 GiB peak (preflight 3,660 s, 22.5 GiB). ×4 0.935647 against the
FocalNet incumbent's 0.938816: −0.0032 (80% interval −0.0047 to −0.0016), +0.0005 above the MambaVision
reference. ×1 0.928740 at epoch 9 (best 0.929164 at epoch 8): the best epoch 1 of the campaign (0.8805,
+0.0077 against FocalNet-T) and ahead or level through epoch 8. Per label at ×4 against FocalNet-T,
largest gain to largest loss: Effusion +0.0033, Contusion +0.0026, ACL +0.0004, Synovitis −0.0002,
PF OA −0.0011, Baker's −0.0013, Medial Meniscus −0.0023, MCL −0.0052, Medial OA −0.0061, Lateral
Meniscus −0.0086, Fracture −0.0092, Lateral OA −0.0104.

## Forced seeds: summary at 06:10 CEST

| Family | Epochs | ×4 | Δ to FocalNet-T | Notes |
|---|---:|---:|---:|---|
| FocalNet-T SRF | 12 | 0.938816 | incumbent | 1×1 convs as linear |
| FocalNet-T SRF | 10 | 0.938154 | −0.0007 | level |
| EfficientViT-B2 | 10 | 0.937280 | −0.0015 | stages 0–1 checkpointed, real-row BatchNorm |
| DaViT-T | 10 | 0.936669 | −0.0021 | |
| EdgeNeXt-B | 10 | 0.936033 | −0.0028 | cheapest (2,798 s) |
| WTConvNeXt-T | 9 | 0.935647 | −0.0032 | Triton WTConv |
| MambaVision-T (reference) | 12 | 0.935160 | −0.0037 | |
| DINOv3 ConvNeXt-T | 10 | 0.934199 | −0.0046 | |
| PVTv2-b2 | 10 | 0.933099 | −0.0057 | stage 0 checkpointed |
| Hiera-T, HRNet-W18, RDNet-T | | | | aborted, over budget, or blocked |

Every challenger led or matched FocalNet-T on ×1 early and lost ground in the last epochs; Effusion is
the label most of them gain on (DaViT, EdgeNeXt, EfficientViT, WTConvNeXt), FocalNet-T's weakest against
MambaVision-T.

## 24. focalnet-early-anchor-follows-rate (blocked by the preflight)

The researcher's first idea: FocalNet-T's early anchor following the rate schedule, at the incumbent's
twelve epochs. Preflight projected 4,309 s (0.637 s per step); the developer's repair added
torch.compile to the FocalNet blocks, which also failed the preflight, and the idea was blocked. No
training ran. Re-proposed at ten epochs as `focalnet-early-anchor-rate-ten-epoch-combination`. The
per-family epoch table in prior-work.md now tells the researcher which counts fit.

## 25. efficientvit-real-row-batch-renorm (rejected)

Bounded Batch Renormalization in EfficientViT-B2's real-row BatchNorm. First preflight 4,785 s
(0.887 s per step), stopped; the repaired candidate 0.612 s per step, 3,414 s projected, trained
3,076 s at 11.4 GiB. ×4 0.936045 against FocalNet-T's 0.938816: −0.0028 (80% interval −0.0046 to
−0.0008); −0.0012 against plain EfficientViT-B2 (0.937280). ×1 0.930695 (plain 0.931092); no dip at
epoch 5. Per label at ×4 against FocalNet-T: Medial OA +0.0046, ACL +0.0039, Contusion +0.0029,
Effusion +0.0025, Baker's −0.0009, PF OA −0.0029, Fracture −0.0034, Medial Meniscus −0.0035,
Synovitis −0.0071, MCL −0.0077, Lateral Meniscus −0.0102, Lateral OA −0.0115.

## 26. focalnet-early-anchor-rate-ten-epoch-combination (rejected, level)

FocalNet-T with the early anchor following the rate schedule as the late one does, at ten epochs.
Trained 3,308 s. ×4 0.938806 against the twelve-epoch incumbent's 0.938816: −0.00001 (80% interval
−0.0013 to +0.0012), and +0.0007 above the plain ten-epoch FocalNet-T (0.938154). ×1 0.931532 at epoch
10 (incumbent 0.933241, plain ten-epoch 0.932380); ahead of the incumbent from epoch 2 to 8. Level with
the incumbent at five sixths of its training time; the runtime keeps the incumbent on a tie.

## 27. dinov3-convnext-twelve-epoch-fit (aborted at epoch 8 on the user's order)

DINOv3 ConvNeXt-T at twelve epochs: ×1 0.8162, 0.8763, 0.8929, 0.9050, 0.9069, 0.9158, 0.9193, 0.9218
after epochs 1 to 8, 0.007 to 0.057 behind FocalNet-T throughout. Aborted at 09:21 CEST when the user
stopped the campaign.

## From 2026-10-16 09:20 CEST: EfficientViT-B2 only

The user stopped the campaign and asked for EfficientViT-B2 ideas only. The queued
`focalnet-stride-sixteen-export-layer-norm` was retired and `wtconvnext-normalized-dense-exports`
removed; `efficientvit-eleven-epoch-fit-from-batchnorm` stays. The researcher's sealed scope now names
EfficientViT-B2 only (backup state.json.bak-efficientvit-only).

## 28. efficientvit-read-layernorm (rejected)

LayerNorm2d on EfficientViT-B2's stride-16 and stride-32 reads (head rate, no anchor). Preflight
3,739 s and 12.3 GiB. ×4 0.936863 against FocalNet-T's 0.938816: −0.0020 (80% interval −0.0039 to
−0.0001); −0.0004 against plain EfficientViT-B2 (0.937280). ×1 0.929011 at epoch 10 (plain 0.931092):
no dip at epoch 5 (0.9207 against 0.9185), but behind plain from epoch 8. The read norms did not hold
the late epochs.

Research is switched off from 10:41 CEST at the user's request (state refill_below 0, backup
state.json.bak-no-research); the campaign stops after efficientvit-bf16-attention.

## 29. efficientvit-bf16-attention (parity with plain EfficientViT-B2)

LiteMLA's two attention products in bfloat16, the normalising division in float32 (code: a LiteMLA
subclass reusing timm's weights; activity requires the bf16 path to run). Preflight 3,585 s, 0.644 s
per step, 11.2 GiB. Trained 3,319 s at 10.4 GiB, against plain EfficientViT-B2's 3,406 s and 11.4 GiB.
×4 0.937130 against FocalNet-T's 0.938816 (80% interval −0.0035 to +0.0002); −0.00015 against plain
EfficientViT-B2 (0.937280). ×1 0.930151 (plain 0.931092), within ±0.004 of the plain run at every epoch
with training losses within 0.002. bf16 attention is free in accuracy and saves 2.6% of the time and
1 GiB.

From 11:43 CEST the queue holds the user's `efficientvit-stochastic-depth` and
`efficientvit-checkpoint-stage0`; research stays off.

## 30. efficientvit-stochastic-depth (rejected; level with the incumbent, best EfficientViT-B2)

Stochastic depth 0.1 on EfficientViT-B2's identity residual blocks (DropPath on the main branch, rates
rising linearly from 0, same masks in the checkpoint recompute). Preflight 3,894 s, 12.2 GiB; trained
3,633 s at 11.4 GiB. ×4 0.938641 against FocalNet-T's 0.938816: −0.00018 (80% interval −0.0019 to
+0.0016), and +0.0014 above plain EfficientViT-B2 (0.937280), the best EfficientViT-B2 run. ×1
0.929984 at epoch 10 (plain 0.931092), so the gain shows at ×4 only. Rejected because the rule needs a
strict gain at both ×4 and ×1. Per label at ×4 against FocalNet-T, largest gain to largest loss: Medial OA +0.0043,
Fracture +0.0030, Effusion +0.0028, PF OA +0.0025, Contusion +0.0019, ACL +0.0012, Synovitis +0.0004,
Baker's −0.0007, Medial Meniscus −0.0034, MCL −0.0035, Lateral Meniscus −0.0050, Lateral OA −0.0054.

## 31. efficientvit-checkpoint-stage0 (aborted after its preflight on the user's order)

Only stage 0 checkpointed, plain EfficientViT-B2 otherwise. Preflight 0.662 s per step (against 0.704 s
with both stages checkpointed and stochastic depth), 3,690 s projected, 17.0 GiB predicted. Aborted
at 12:57 CEST before its first epoch: the user asked for the stage-0-only checkpointing to carry
stochastic depth, queued as `efficientvit-stochastic-depth-stage0`.
At the user's request the queued `efficientvit-stochastic-depth-stage0` also carries the bf16 attention
of `efficientvit-bf16-attention` (its first build, without it, was superseded at 13:07 CEST).

## 32. efficientvit-stochastic-depth-stage0 (rejected)

Stochastic depth 0.1, only stage 0 checkpointed, bf16 attention products. Preflight 0.638 s per step;
trained 3,244 s at 14.8 GiB (the stochastic-depth run with stages 0 and 1 checkpointed and float32
attention: 3,633 s, 11.4 GiB; −11% time). ×4 0.937366 against FocalNet-T's 0.938816 (80% interval
−0.0032 to +0.0004) and −0.0013 against the stochastic-depth run (0.938641). ×1 0.929824 against
0.929984. Checkpointing depth leaves outputs and gradients unchanged by construction; the attention
precision changes the arithmetic, which alone cost −0.00015 at ×4 without stochastic depth (run 29).
The stochastic-depth run with float32 attention stays the best EfficientViT-B2.

## 33. efficientvit-stochastic-depth-02 (rejected; level with the incumbent, best EfficientViT-B2)

Stochastic depth 0.2, only stage 0's first block checkpointed, bf16 attention. Preflight 0.604 s per
step, 19.0 GiB measured, 20.5 GiB predicted (the keeper's estimate from saved activations: 20.2).
×4 0.938773 against FocalNet-T's 0.938816: −0.00004 (80% interval −0.0018 to +0.0018); +0.0001 above
stochastic depth 0.1 (0.938641) and +0.0015 above plain EfficientViT-B2. ×1 0.930888 (0.1: 0.929984).
The best EfficientViT-B2 and the closest challenger of the campaign; it goes to the five-fold training.
