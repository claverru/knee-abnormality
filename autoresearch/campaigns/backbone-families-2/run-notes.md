# Backbone families 2: run notes

Fold 2 of `all_5fold_scanner_v2`, ×4 macro AUC. Blend: equal logit average with the ResNet-50
`resnet50-silver-fold2` (×4 0.937820 alone); gain over the ResNet-50 alone. Deltas before
2026-10-18 10:00 are against FocalNet-T SRF `focalnet-silver-fold-2` (×4 0.938816, ×1 0.933241);
after it against NextViT-S (×4 0.938851, ×1 0.934578). Started 2026-10-17 13:55 CEST with Codex,
stopped by the user 2026-10-18 15:02 after 25 trainings (21 completed, four stopped early).

## Results

| Run | Pretraining | Epochs | ×4 alone | ×4 Δ | ×1 Δ | Blend gain | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `nextvit-small-ssld` | SSLD 6M | 10 | 0.938851 | +0.000034 | +0.001337 | **+0.003267** | provisional accept, incumbent |
| `tinyvit-21m-in22k` | IN-22k distillation | 10 | 0.938808 | −0.000043 | −0.002656 | +0.003223 | reject |
| FocalNet-T SRF (reference) | IN-1k | 12 | 0.938816 | | | +0.003179 | |
| `focalnet-tiny-lrf-twelve-epochs` | IN-1k | 12 | 0.937903 | −0.000948 | −0.002588 | +0.002933 | reject |
| `focalnet-tiny-lrf` | IN-1k | 10 | 0.937354 | −0.001463 | −0.001082 | +0.002710 | reject |
| `efficientnet-b3-noisy-student` | JFT-300M | 10 | 0.935737 | −0.003113 | −0.005371 | +0.002233 | reject |
| `flash-internimage-seven-epochs` | IN-1k | 7 | 0.935955 | −0.002862 | −0.005331 | +0.001833 | reject |
| `inceptionnext-tiny` | IN-1k | 10 | 0.936240 | −0.002611 | −0.003392 | +0.001829 | reject (keeper's offline reading; comparison cancelled by the stop) |
| `internimage-half-stage-middle-read` | IN-1k | 7 | 0.935852 | −0.002965 | −0.005734 | +0.001714 | reject |
| `flash-internimage-eight-epochs` | IN-1k | 8 | 0.935718 | −0.003099 | −0.005862 | +0.001711 | reject |
| `lingbot-native-normalized-middle-read` | LingBot (web, boundary SSL) | 10 | 0.933450 | −0.005366 | −0.006760 | +0.001229 | reject |
| `lingbot-stage-decay-08` | LingBot | 10 | 0.932707 | −0.006109 | −0.006119 | +0.000779 | reject |
| `lingbot-midpoint-normalized-middle-read` | LingBot | 10 | 0.931819 | −0.006997 | −0.006472 | +0.000348 | reject |
| `vssd-mesa-five-epochs` | IN-1k | 5 | 0.933081 | −0.005736 | −0.006123 | +0.000315 | reject |
| `radimagenet-inception-resnet-v2-rerun` | RadImageNet | 10 | 0.931412 | −0.007405 | −0.010313 | −0.000634 | reject |
| `lingbot-vit-small-stride16` | LingBot | 10 | 0.928395 | −0.010421 | −0.010156 | −0.000649 | reject |
| `transnext-low-resolution-direct-four-epochs` | IN-1k | 4 | 0.928007 | −0.010809 | −0.010475 | −0.001120 | reject |
| `radimagenet-densenet121` | RadImageNet | 10 | 0.928967 | −0.009849 | −0.010287 | −0.001425 | reject |
| `radimagenet-densenet121-normalized-middle` | RadImageNet | 10 | 0.928541 | −0.010275 | −0.010328 | −0.001630 | reject |
| `radimagenet-inception-v3` | RadImageNet | 10 | 0.927303 | −0.011513 | −0.015051 | −0.001935 | reject |
| `radimagenet-densenet121-stage-decay-06` | RadImageNet | 10 | 0.926036 | −0.012781 | −0.013524 | −0.002352 | reject |
| `medsam2-hiera-tiny` | MedSAM2 (CT, MRI, PET) | 10 | 0.917773 | −0.021043 | −0.021869 | −0.002900 | reject |

Invalid: `medsam2-stem-separated-clipping` and `medsam2-thirteen-epochs` (stopped by the user, not
going to catch up), `radimagenet-inception-resnet-v2-grids` (stopped at epoch 2 for a disk
compaction, rerun), `nextvit-small-twelve-epochs` (stopped by the user: no optimisation of families
that already worked). Blocked on the preflight (over the time bound at ten epochs before the bound
was raised): OverLoCK-XT (14,736 s), TransNeXt-Micro (11,038 s), VSSD-Tiny (6,828 s),
FlashInternImage-T (5,286 s), FlashInternImage-T nine epochs (4,330 to 4,414 s).

NextViT-S per label against FocalNet-T SRF (×4): PF OA +0.0075, Contusion +0.0050, Medial OA
+0.0036, Effusion +0.0030, ACL +0.0003, Medial Meniscus +0.0003, Baker's +0.0001, Synovitis
−0.0033, Lateral OA −0.0033, Lateral Meniscus −0.0033, MCL −0.0035, Fracture −0.0058. Three-way
reading (not a campaign result): ResNet-50 + FocalNet-T SRF + FocalNet-T LRF 0.941207 against
0.940999 for the pair.

## Lessons

- **Pretraining decides the blend, not the score alone.** Every family pretrained on RadImageNet
  (DenseNet-121, InceptionV3, InceptionResNetV2, in four settings) made the ResNet-50 blend worse,
  even InceptionResNetV2 at ×4 0.9314; the ResNet-50 is pretrained on the same images (guess: shared
  errors). SSLD (NextViT-S), ImageNet-22k distillation (TinyViT-21M), JFT noisy student
  (EfficientNet-B3) and ImageNet-1k FocalNets add +0.002 to +0.0033.
- **Optimising a family moved results by about 0.001; a new family by about 0.01.** LingBot reads
  and stage decay, FlashInternImage epochs and reads, DenseNet reads and stage decay and twelve-epoch
  FocalNet-T LRF stayed within ±0.002 of their parents. The user stopped optimisation on 2026-10-18.
- **Speed guesses from published throughput were wrong by 1.5 to 3.8 times** (OverLoCK-XT,
  TransNeXt-Micro, VSSD-Tiny, FlashInternImage-T): the training forms, pure-PyTorch paths and
  batches of about 256 images make memory and small kernels set the time. From the second batch of
  candidates every seed was timed on the GPU first (64 images forward and backward, bf16; ten epochs
  ≈ 10 × (138 + 1,954 × s), peak at 256 images ≈ 4.7 × the 64-image figure).
- **The time bound was 1.006 of the reference by mistake** (the first round's window); the user set
  up to 110% and then 120% on 2026-10-18. Raising it on a sealed campaign needs a stop, then
  `campaign.yaml`, the sealed `policy` and `budget`, `limits` in `state.json`, `limits.json` and
  `state["sealed_sha256"]`; editing `campaign.yaml` alone while running fails the seal check.
- **Integration defects of new families:** a wrapper module without `pretrained_cfg` stops the
  preflight; maps that are not 2:1 between stride 16 and 32 (Inception's 12 and 5) break the 2×2
  repeats in `model.py`, replaced in those candidates by a nearest-centre gather that equals the
  repeat for every 2:1 backbone. Seeds that spelled both out built without a repair.
- **Original RadImageNet weights** (Keras files converted by the keeper, checked against Keras to
  5e-6) load correctly but plateau at ×1 about 0.923 on every architecture; stage-decayed rates and a
  normalised DenseNet read did not move them. The weights' flat 0.1 to 0.3 RMS cannot be rescaled
  exactly under Keras's BatchNorm epsilon of 0.001.
- **The epoch-1 validation-loss spike of DenseNet-121** is a calibration shift of the main head (all
  logits up by about one for one epoch, ranking intact), not warm-up (warm-up is per step).
- **MedSAM2 Hiera-T** started 0.13 below at epoch 1 and closed to −0.021; medical segmentation
  pretraining did not carry to these labels.
- **Run-to-run spread:** the same InceptionResNetV2 build differed by 0.003 at ×1 at epoch 1 between
  two trainings (non-deterministic kernels).

## After the campaign

NextViT-S was merged on main as `configs/nextvit.yaml` (`318eba2`) and trained on folds 0, 1, 3 and
4 with one seed per fold (451509333, 1514023685, 350704958, 1277371158; fold 2 used 1110484336) as
`outputs/training/nextvit-silver-fold-{k}`: ×4 0.93819 (fold 0), 0.93127 (fold 1), 0.93469 (fold
3); fold 4 was training when this was written.
