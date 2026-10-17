# Crop views: prior work

Measured 2026-10-16 from saved ×4 predictions; no training. Blends are equal logit means. Values
×1e-4 macro AUC unless marked. "Pair" is the blend's macro minus the kept run's; "solo" is the run's
own macro minus the kept run's.

## The control

`outputs/training/resnet50-silver-fold2`: `configs/resnet50.yaml` with only `validation.folds`
overridden to `[2]`, trained 2026-10-14 from clean commit `9a93087`; ×4 0.937820, best epoch 9 of
10, training 2,617 s, peak 16.4 GB. Its input config equals `configs/resnet50.yaml` at `485e779`.
The commits since add the FocalNet-T and EfficientViT-B2 adapters and the autoresearch preflight;
the backbone-families campaign recorded that they leave the ResNet-50 path unchanged. Declared
exceptions: the machine paths and `preprocessing_identity.sources_sha256`.

Not the control: targets-and-inputs run 27 (×4 0.939634). It carries the removed mask decoder, and
every one-setting run of that campaign loses 2.5 to 4.8 when blended with it, against +3 to +9
with the control.

## Yardstick

Complement gain against the control of the 29 fold-2 ResNet-50 runs of the targets-and-inputs
campaign, each one setting away from the recipe outside the views (`campaign/complement.py`):
median **+5.86**, mean +5.87. Runs with a solo change within ±3: +3.5 to +7.4. Runs that are
better alone add more: average-pool shortcuts +13.3 (solo +18.1), GIN off +20.1 (solo +19.1).

## Fold-2 family runs

For the sixth-member reading: `resnet50-silver-fold2` 0.937820, `mamba-silver-fold-2` 0.935160,
`focalnet-silver-fold-2` 0.938816, `efficientvit-sd02-fold-2` 0.938773,
`convnext-silver-only-imagenet-fold-2` 0.936123 (all `outputs/training/`, `fold_2`); the five
blended 0.941358. Against the control, each other family adds +18.4 (MambaVision-T) to +31.8
(FocalNet-T) in a pair, about +26 to +32 beyond half its solo change, against +5 for a ResNet-50
one setting away.

## Precedents

ResNet-18 runs, one each, against the kept run of their campaign. Yardsticks: on fold 0 runs one
setting away add +1.6 to +2.0 and two hand-view runs that differ only by the label bias +4.8; on
fold 1 a retraining of the kept run adds +2.0. 133 crop and pose runs were ranked; the rows below
are the ones the seeds and the exclusions rest on.

### Whole-framing changes

| Run (campaign, fold) | Pair | Solo | Per label in the blend, gain to loss |
|---|---:|---:|---|
| view-size-menu-bias-guided (learnable-views, 0) | +16.5 | +3.5 | Fra +45, PFOA +28, Syn +23, LM +19, LOA +15, MCL +15, MM +15, ACL +14, Con +11, Bak +8, Eff +2, MOA +1 |
| two-stream-paired (mixed-field, 0; its 140 mm stream never trained, so 120 mm alone) | +16.3 | +8.3 | MCL +48, LM +45, Fra +22, LOA +21, MOA +19, Eff +19, PFOA +18, ACL +14, Bak +9, Con +3, Syn −9, MM −13 |
| two-views-every-type (learnable-views, 0) | +13.3 | −9.3 | Bak +51, LM +25, MCL +17, LOA +17, PFOA +15, Syn +12, Fra +9, MOA +7, ACL +6, Con +2, Eff +2, MM −2 |
| learnable-views runs started from the 120 mm centre crop | +8 to +16 | | |
| learnable-views runs started from the hand views | +4.1 to +7.3 | | |
| smart-crops runs on a 120 mm base plus a crop | +7 to +9 | about −20 | |
| rn18-field-120-control against view-frozen-control (0) | +5.5 | −19.6 | MCL +62, LM +31, Bak +18, PFOA +16, MM +6, Con +3, Eff 0, MOA −4, LOA −7, Fra −10, ACL −11, Syn −39 |
| rn18-field-120-control against rn18-view-label-bias (0) | +6.7 | −18.1 | MCL +39, LM +32, Bak +30, LOA +17, ACL +12, PFOA +8, MM +5, Con +3, Eff −2, MOA −4, Fra −27, Syn −34 |

Learned widths drifted to about 110 mm from any start; the size menu chose coronal 93 to 120,
coronal fluid 105, sagittal T1/PD 93 to 105 and sagittal fluid 120 mm
([learnable views](../../../docs/validation/results/2026-10-03-learnable-views-campaign.md)).

### Single-type widenings

| Run (campaign, fold) | Pair | Solo |
|---|---:|---:|
| coronal crops a ninth wider (anatomy-masks, 1) | +4.1 | −0.3 |
| sagittal fluid back to 120 mm (smart-crops, 0) | +1.9 | −6.9 |
| coronal fluid back to 120 mm (smart-crops, 0) | +1.9 | −4.2 |
| sagittal fluid 100 mm (smart-crops, 0) | +1.3 | −1.6 |
| sagittal fluid 112 mm (anatomy-recipe, 1) | −0.6 | −8.7 |
| sagittal crops a ninth wider (anatomy-masks, 1) | −0.9 | −8.0 |
| coronal fluid 108 mm (anatomy-recipe, 1) | −1.0 | −16.8 |
| sagittal fluid 106 mm context view (rn50-input-sampling, 1) | −2.0 | −25.0 |
| sagittal fluid 72 + 112 mm sharing the slices (manual, 1) | −4.5 | −27.4 |

### Pose settings

| Run (campaign, fold) | Pair | Solo | Per label |
|---|---:|---:|---|
| crop size per knee (anatomy-recipe, 1; adopted since as the per-knee scale) | +6.7 | +3.2 | MCL +49, Syn +21, MOA +14, MM +10, PFOA +10, Bak +9, Eff +7, ACL −4, Con −4, LM −7, LOA −10, Fra −15 |
| learned spot shift with spatial maps (routing-and-shifts, 1) | +5.3 | +1.4 | MCL +33, LM +25, MOA +7, MM +5, Bak +5, PFOA +4, Eff +3, Fra +1, Syn −3, LOA −4, ACL −5, Con −9 |
| learned per-view shifts (routing-and-shifts, 1) | −4.0 to +2.5 | | |
| crops on the notch or the menisci, a centre per contrast (view-poses, 1) | −5.7 to −0.2 | | |

### Pose estimation

| Run (campaign, fold) | Pair | Solo | Per label |
|---|---:|---:|---|
| soft bone shares for the joint plane (anatomy-masks, 1) | +4.2 | −0.4 | Fra +21, MCL +19, PFOA +12, MM +11, ACL +6, Eff +4, Syn +3, LM −2, MOA −3, Con −6, LOA −7, Bak −7 |
| first look at interior depth quantiles (anatomy-masks, 1) | +4.2 | 0.0 | Fra +16, MM +16, LOA +11, ACL +9, Syn +8, PFOA +8, Eff +4, Bak 0, MOA −1, MCL −3, LM −3, Con −16 |

### Auxiliary backbones

| Run (campaign, fold) | Pair | Solo | Note |
|---|---:|---:|---|
| two-stream-paired (mixed-field, 0) | +16.3 | +8.3 | ResNet-10t 140 mm stream behind a zero-started scalar gate; it never trained (readout 0.506), so the run is 120 mm alone |
| two-views-fluid-140 (smart-crops, 0) | +7.7 | −21.2 | ResNet-10t 140 mm fluid stream feeding the pool from the first step, no gate |
| two-views-112-fluid-140 (smart-crops, 0) | +6.2 | −25.3 | the same with the main view at 112 mm |
| crop-own-loss (smart-crops, 0) | +5.5 | −24.7 | a crop's tokens also pooled alone under their own loss |

Multiply-adds per image (timm, measured): ResNet-50 4.09 GMAC at 224 px; ResNet-10t 1.10 at 224 px
and 3.0 at 368 px. Per study the control spends about 143 ResNet-50 forward equivalents: 111 for
training (37 images, forward and backward) and 32 for the no-gradient first look. The
Refined-RadImageNet ResNet-10t (`resnet10t.c3_in1k`, `refined_radimagenet_resnet10t.pth`, sha256
`f5219dff…a079ab`) is in the torch hub cache; its registry entry was removed at `94868bf`.

## Coverage

Measured on 150 fold-1 knees ([view-pose studies](../../../docs/research/view-pose-studies.md#1-knee-size-and-per-knee-scale-back)):
share of series with the structure fully inside.

| View | Structure | Today | Per-knee scale | Wider for all |
|---|---|---:|---:|---:|
| Coronal fluid | collateral region | 34.8% | 17.4% | 93.5% |
| Coronal 84 mm | femoral condyles | 91.6% | 98.5% | 100% |
| Sagittal fluid | whole kneecap | 26.0% | 15.3% | 88.7% |
| Sagittal 84 mm | whole kneecap | 7.8% | 1.1% | 56.4% |

## What did not carry

None of the old ResNet-18 checkpoints adds to today's five-family blend as a sixth member (held-out
halves −0.0001 to +0.0001 on folds 0 and 1); the quality gap of 0.006 to 0.010 outweighs their
diversity. This campaign retrains the framings on the ResNet-50 instead.
