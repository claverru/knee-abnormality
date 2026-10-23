# Scout backbone candidates

timm backbones that could replace the MobileNetV3-small scout, surveyed on 2026-09-16 against
the learned-zoom baseline (`configs/baseline.yaml` (removed) at `ea19772` plus working changes; timm
1.0.26, torch 2.10).

The scout was removed later that day with the
[crop-selection investigation](../../history/crop-selection-investigation.md), so this survey is
historical. It applies only if a scout design returns. The shared background lives elsewhere:

- **Costs:** the
  learned-zoom backbone-swap snapshot (`f076a09:docs/history/crop-selection-sources/learned-zoom-backbone-swaps.md.txt`)
  has the scout's cost counts, delta formula and code requirements.
- **Mapping:** [mapping pitfalls](../../modeling/backbones/backbone-swaps.md#mapping-pitfalls) is current.
- **Evidence:** [backbone MRI evidence](backbone-mri-evidence.md) covers which families have
  held up on MRI.

## Answer

- **Evidence:** no timm backbone of scout size has proven itself on MRI. EfficientNet is the
  only family with consistent MRI evidence, from other MRI tasks; on this dataset
  EfficientNetV2-S ranked 38th of 40. ImageNet top-1 is a weak guide.
- **Candidates worth a run** (project proposal):
  - `efficientnet_b0` (+37% compute): evidence-backed. Moderate confidence that it is a sound
    scout, low confidence that it beats the current one here.
  - `rexnet_100` (+38%): rests on the ReXNet-200 result on this dataset. Low-to-moderate
    confidence.
  - `repghostnet_100` (+3%, fewer parameters): cheap and speculative.
- **A larger scout was deferred.** The crop-selection notes
  ([options not pursued](../../history/crop-selection-investigation.md#options-not-pursued))
  attributed flat attention to the training signal, not capacity; this survey only prices the
  options.
- **Cheapest check:** one fold-0 run per candidate, comparing `val_draft_macro_auc`. It scores
  the coarse scout tokens alone, so it isolates the scout (0.633 for
  [crop224](../../validation/results/crops-and-views/2026-09-16-crop224-result.md)). Then time the winners on a Kaggle T4
  in FP16.

## Candidates

- **Deltas:** relative to the whole learned-zoom model, from the scout formula in the
  backbone-swap snapshot (`f076a09:docs/history/crop-selection-sources/learned-zoom-backbone-swaps.md.txt`).
- **Top-1:** from the timm table at each model's own evaluation size.
- **Adapt:** a [mapping category](../../modeling/backbones/backbone-swaps.md#mapping-pitfalls).
  - *native:* a mapping entry is enough.
  - *offset:* the mapping entry needs a non-zero origin.
  - *remap:* the family needs its own mapping function.

| Model | Top-1 | Scout params | GMACs at 300 px | Channels | Δ params | Δ compute, train | Δ compute, inference | Adapt | MRI evidence |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| *`mobilenetv3_small_100.lamb_in1k` (current)* | 67.6 | 0.93M | 0.104 | 624 | – | – | – | native | Weak |
| `repghostnet_080.in1k` | 72.2 | 0.34M | 0.125 | 192 | −4.4% | −1.2% | −1.2% | remap | Weak |
| `repghostnet_100.in1k` | 74.2 | 0.52M | 0.185 | 240 | −3.2% | +3.0% | +3.0% | remap | Weak |
| `repghostnet_130.in1k` | 76.4 | 0.86M | 0.298 | 312 | −0.9% | +10.7% | +10.8% | remap | Weak |
| `regnety_002.pycls_in1k` | 70.3 | 2.79M | 0.384 | 520 | +11.7% | +17.5% | +17.7% | remap | Some |
| `mobilenetv4_conv_small.e2400_r224_in1k` | 74.6 | 1.26M | 0.344 | 1056 | +2.8% | +18.1% | +18.3% | native | None |
| `mobilenetv3_large_100.ra4_e3600_r224_in1k` | 77.2 | 2.97M | 0.400 | 1072 | +13.8% | +21.8% | +22.0% | native | Weak |
| `convnext_zepto_rms_ols.ra4_e3600_r224_in1k` | 73.2 | 1.91M | 0.577 | 384 | +5.9% | +29.1% | +29.4% | offset | Family only |
| **`efficientnet_b0.ra4_e3600_r224_in1k`** | 79.4 | 3.60M | 0.688 | 432 | +16.7% | +36.6% | +37.0% | native | **Consistent** |
| **`tf_efficientnet_b0.ns_jft_in1k`** | 78.7 | 3.60M | 0.688 | 432 | +16.7% | +36.6% | +37.0% | remap | **Consistent**; in lumbar 11th's ensemble |
| `hgnetv2_b0.ssld_stage2_ft_in1k` | 78.6 | 1.85M | 0.592 | 1536 | +7.4% | +37.0% | +37.4% | offset | Family only |
| **`rexnet_100.nav_in1k`** | 77.9 | 3.28M | 0.717 | 313 | +14.5% | +37.8% | +38.2% | native | **Full-size version best on this dataset** |
| `regnety_004.tv2_in1k` | 75.6 | 3.90M | 0.759 | 648 | +19.0% | +42.5% | +42.9% | remap | Some |
| `convnext_atto.d2_in1k` | 77.0 | 3.37M | 0.927 | 480 | +15.4% | +52.3% | +52.9% | offset | Family only |

## Dropped

Scout GMACs at 300 px are given as multiples of the current scout.

- **Too expensive (about 1.0 GMAC or more, roughly +58% compute or more):** ConvNeXt femto (13×)
  and pico (22×), RegNetY-006/008 (11–15×), ReXNet-130 (12×), EfficientNet-B1 (10×) and
  MobileNetV4-medium (15×).
- **Transformers without MRI evidence:** ViT-Tiny/DeiT-Tiny (22×), TinyViT-5M (28×),
  XCiT-nano (10×), FastViT-T8 (10×), MobileViT-XS (15×), SHViT-S4 (11×) and MobileOne-S0 (19×).
- **Fixed input size:** LeViT, EfficientViT-M and EfficientFormer/V2 need `scout_size: 224`,
  which confounds the backbone change with scout resolution.
- **Beaten by a kept model at similar cost, or barely better than the current scout:**
  `repghostnet_058`, GhostNet V1–V3, EfficientViT-B, LCNet, TinyNet, MnasNet-100, FBNet,
  MobileNetV2 and EfficientNet-Lite0.
- **`mnasnet_small`:** has some evidence, but it trailed ConvNeXt-nano in lumbar 3rd place's
  comparison and scores below the current scout on ImageNet.

## Method and artifacts

- **Candidates:** every pretrained timm model with an ImageNet result and at most 20M
  parameters, from `results/results-imagenet.csv` on timm's main branch (downloaded
  2026-09-16). 385 were measured; 10 had no `features_only` support.
- **Features:** each model was built with `features_only` and `pretrained=False`, using the
  last stride-16 and stride-32 features. The indices come from `feature_info`, which matters
  for ViT-style feature getters.
- **Input size:** fixed-size models were retried with `img_size=300`, then
  `dynamic_img_size`, then at their native size.
- **Compute:** `torch.utils.flop_counter.FlopCounterMode` on one 3×300×300 input, halved for
  GMACs. timm's fused attention was disabled, because the counter skips the fused kernel.
- **Geometry:** from `_spatial_mapping` over module-order convolutions (see
  [mapping pitfalls](../../modeling/backbones/backbone-swaps.md#mapping-pitfalls)).
- **Artifacts:** [`outputs/research/scout-backbones-20260916`](../../../outputs/research/scout-backbones-20260916)
  contains `sweep.py`, `sweep.log`, `sweep.parquet` (SHA-256 prefix `63d07d8cca7a09d1`),
  `results-imagenet.csv` (`30972bf7c7ab451b`) and `budget.py`.
