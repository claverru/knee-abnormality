# 2026-10-17 Crop views campaign

Earlier crop and pose framings retrained on the ResNet-50 recipe as ensemble companions, in the
[crop-views campaign](../../../autoresearch/campaigns/crop-views/INDEX.md) (2026-10-16 22:00 to
2026-10-17 10:30 CEST, Codex, 13 decided runs, stopped by the user), then four cross-checks across
folds and backbones. Control: `resnet50-silver-fold2` (fold 2 of `all_5fold_scanner_v2`, seed
1110484336, ×4 0.937820, 2,617 s, 16.4 GB). Nothing was merged into the recipe.

## Adoption by complement

The campaign ran on an uncommitted runtime change in the worktree `knee-abnormality-crop-views`
(since removed): a run was adopted when the ×4 macro AUC of its equal-logit blend with the control
gained more over the control than the incumbent's blend did. The control's own bar was +0.00059,
the median blend gain of 29 fold-2 targets-and-inputs runs with the control. Two further readings:

- **Beyond ½Δ:** blend gain minus half the run's own ×4 change, the diversity left after the
  quality change. Near copies score about +5 ×1e-4; other backbone families +26 to +32.
- **Sixth member:** the gain when the run joins the fold-2 five-family blend (ResNet-50,
  MambaVision-T, FocalNet-T, EfficientViT-B2, ConvNeXt-T; ×4 0.941358).

## Runs

×1e-4 against the control.

| Run | Own ×4 | Blend gain | Beyond ½Δ | Sixth member | Verdict |
|---|---:|---:|---:|---:|---|
| Coronal crops a ninth wider (93.3/103.7 mm) | +16.5 | +13.6 | +5.4 | +3.9 | adopted, final incumbent ×4 0.939467 |
| Coronal wider with per-knee scale off | +7.2 | +11.8 | +8.2 | +3.2 | rejected |
| Per-knee scale off | +7.7 | +9.8 | +5.9 | +3.1 | adopted |
| Posed and wide views split by depth (scale off) | −1.9 | +9.8 | +10.8 | +2.1 | adopted |
| ResNet-10t full-canvas locator | +4.1 | +8.7 | +6.7 | +2.1 | rejected; its pose matched the first look |
| Two views, wide and posed | −5.6 | +7.6 | +10.4 | +1.6 | adopted |
| Unposed 120 mm everywhere | −20.3 | +7.6 | +17.7 | +0.7 | rejected |
| Wide views, unposed axial, scale off | −13.6 | +7.3 | +14.1 | +0.9 | rejected |
| Soft bone shares for the joint plane | +4.3 | +5.3 | +3.1 | +1.7 | rejected |
| Wide fluid stream, 3 slices (ResNet-10t) | −7.7 | +2.1 | +6.0 | +1.9 | rejected |
| Axial views unposed | −12.0 | −0.6 | +5.5 | −0.7 | rejected |
| ImageNet ResNet-50 | −50.2 | −1.8 | +23.3 | −5.5 | rejected |
| ImageNet ResNet-50, unposed 120 mm | −59.5 | | | | rejected |

Two trainings of one recipe differ by about ±10 at ×4; small labels swing ±40 (Fracture has 60
positives).

- **The axial pose helps** about +12 at ×4: both runs that unposed the axial views lost it.
- **Crop diversity is small next to family diversity:** the best crop variant adds +3.9 as a sixth
  member; the ImageNet ResNet-50 has family-level diversity (+23.3 beyond ½Δ) but costs −50 alone.
- **Pretraining sets the label split between families:** the ImageNet ResNet-50's per-label
  pattern against the control correlates +0.73 with the other families' pattern.
- **Framing by pretraining 2×2:** unposing moved Lateral Meniscus (+85, z 2.9) and Synovitis
  (+79, z 2.7) in opposite directions on RadImageNet and ImageNet weights; the framing effects of
  the two pretrainings correlate only 0.34.
- **ImageNet weights need a rescale for this recipe:** `resnet50.a1_in1k` convolutions have 4 to
  27 times RadImageNet's weight RMS, so the recipe's rate and anchors left the backbone nearly
  frozen (epoch-1 ×1 0.7535). Rescaling each convolution and its BatchNorm statistics to
  RadImageNet's RMS preserves the function within 0.4% and gave epoch-1 ×1 0.834
  ([patch](../../../autoresearch/campaigns/crop-views/imagenet-conv-scale.patch)).

## Cross-checks

| Check | Δ ×4 | Δ ×1 | Notes |
|---|---:|---:|---|
| ResNet-50 coronal wider, fold 0 | −0.0003 | +0.0001 | level |
| ResNet-50 per-knee scale off, fold 0 | −0.0004 | +0.0003 | level |
| MambaVision-T coronal wider, 10 epochs, against the 12-epoch reference | −0.0007 | −0.0010 | Fra +80, MCL −130; blend +9.6, beyond ½Δ +12.9 |
| MambaVision-T unposed 120 mm, 10 epochs, against the 12-epoch reference | −0.0017 | −0.0034 | blend +15.7, beyond ½Δ +24.2 |

- **The two solo gains did not carry over:** both are level on fold 0 and the wider coronal crops
  are about level on MambaVision-T. Fracture rose in all three wider-coronal runs (+39, +13, +80).
- **The framing by pretraining interaction is not confirmed:** the 2×2 predicted Lateral Meniscus
  and Synovitis up when an ImageNet-pretrained model is unposed; MambaVision-T shows LM −19 and
  Syn −4 against its reference, and LM −3 and Syn +14 against the epoch-matched wider-coronal run.
  Fracture rose (+42) where the RadImageNet ResNet-50 lost −100. Unposing costs macro AUC on both
  backbones.

Runs are in `outputs/training/xcheck-*` and `outputs/autoresearch/crop-views/` (not tracked); the
seed sources are frozen per experiment there.
