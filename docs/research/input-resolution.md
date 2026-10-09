# Input resolution, field of view and slice density

Researched on 2026-09-15 from four sources: an acquisition-header audit of all train and
example-test series, image measurements on 300 series, this repository's past results, and
a literature and competition review. Artifacts and scripts are in
`outputs/analysis/i/input-resolution-research-20260915/` (`acquisition/`, `images/`). The
[crop-run evidence](#evidence-from-the-crop-runs) was added on 2026-09-16; none of it is a
controlled resolution test.

## Answer

**Since 2026-10-01 the ResNet recipes use 224 px / 0.536 mm / 120 mm**: every run with a
narrower, finer field beat the 140 mm reference, 160 mm added nothing in five designs, and
120 mm scored 0.947 public against 0.943
([result](../validation/results/2026-10-01-mixed-field-campaign.md)). The measurements below
were taken on the earlier input and explain why.

The 224 px / 0.625 mm / 140 mm input discards some real in-plane detail and clips
posterior soft tissue in a third of sagittal series. Finer than about 0.5 mm adds almost
nothing, because the stored ~0.31 mm images are interpolated. Depth sampling, with triplets
about 17 mm apart at 1×, is a larger gap than in-plane resolution.

The crop runs of 2026-09-15/16 suggest that ReXNet-200 needs both of these:

- most slices;
- a spacing of 0.625 mm or finer.

Resolution at a fixed field of view is still untested. The 0.5 mm, 280 px test is deferred to
the [end of the competition](../backlog/endgame.md#bigger-images).

## Acquisition (all 24,371 train series)

- **Stored images:** pixel median 0.31 mm (p5–p95 0.17–0.56), matrix median 512, FOV
  median 160 mm (p5 140, p95 182–190) in every plane.
- **Acquired resolution** (FOV ÷ acquisition matrix, usable for 86% of series): finer axis
  median 0.49 mm (p25–p75 0.42–0.56), coarser axis median 0.625 mm (0.55–0.70), median
  interpolation 2×. The spectra confirm interpolation, with sharp k-space cutoffs matching
  the header, in 95–100% of series.
  - GE has the coarsest phase axis (median 0.71 mm).
  - Philips is the most isotropic.
  - Fat-suppressed fluid-sensitive series are coarser than non-fat-suppressed ones.
- **Share of series whose full acquired band fits under each spacing's Nyquist:**
  0.3125 mm 99%, 0.417 mm 85%, 0.5 mm 59%, 0.625 mm 23%, 0.7 mm 12%.
- **Downsampling loss** (repo resampler, native → spacing → native, inside tissue):

  | Spacing (mm) | 0.3125 | 0.417 | 0.5 | 0.625 | 0.7 |
  |---|---|---|---|---|---|
  | SSIM | 0.97–0.99 | 0.94–0.96 | 0.91–0.95 | 0.84–0.90 | 0.79–0.87 |

  0.625 → 0.5 mm cuts the error by about 21%.
- **Through-plane:** thickness median 3.0 mm, spacing median 3.5 mm (p95 5.0), 30 slices,
  depth 106 mm. A triplet spans about 10 mm. About 3.6% of series are 3D (0.4–1 mm spacing).
- **Test examples** fall inside the train distributions.

## Field of view and joint position

- **Tissue sticking out more than 10 mm past an image-centred crop:**

  | Crop | 120 | 130 | 140 | 150 | 160 |
  |---|---|---|---|---|---|
  | Sagittal (almost all posterior) | 81% | 64% | 34% | 8% | 2% |
  | Coronal | 49% | 25% | 9% | 1% | 0% |
  | Axial | 59% | 29% | 16% | 10% | 6% |
- **Sagittal clipping is width, not placement (re-read 2026-09-21).** Skin-to-skin AP tissue is
  already centred: its midpoint sits a median 3.6 mm posterior of the image centre (p5–p95 −1.4
  to +11.2 mm, 96 sagittal series, middle slice). The width is median 146 mm and exceeds 140 mm in
  67 of them. None of the 32 series clipped more than 10 mm posteriorly fits in 140 mm, and their
  median anterior margin is 0.4 mm, so a tissue-centred crop would move the clipping to the
  patella rather than remove it. Only a wider field helps; 13 of the 32 fit in 150 mm. The tissue
  mask includes the posterior soft tissue, so no rate of cysts leaving the 140 mm window is
  measured. Source: `series_metrics_overhang.parquet` in the `images/` artifacts.
- **Baker's cyst clipping was a 112 mm crop-era loss.** Its per-label evidence is the centred-crop
  drop (−0.028) and the 128–130 mm losses. The three Baker's-positive fold-0 studies whose
  prediction fell most under the centred 112 mm crop (0.95 → 0.22, 0.91 → 0.25, 0.92 → 0.32 in
  `outputs/analysis/c/crop-placement-20260915/`) were reviewed on fluid-sensitive slices. In two of
  them (…93267826, …11059491) the cyst lies just outside 112 mm and inside 140 mm; the third is
  unclear.
- **Placement reopened, 2026-09-21 (user decision):** anatomy-anchored crop placement (joint line
  up–down, tissue in the other direction) at the fixed 140 mm / 0.625 mm window is in scope of the
  [sampling campaign](../../autoresearch/campaigns/sampling/contract.md). Spacing stays shared and
  unchanged; the field-of-view part of the decision below stands.
- **Decision, 2026-10-01 (user): 120 mm on every series,** which supersedes the decision below
  for the ResNet recipes. A wider field is closed at 224 px; placement stays image-centred and
  no per-plane or per-contrast field is used
  ([result](../validation/results/2026-10-01-mixed-field-campaign.md)).
- **Decision, 2026-09-21: no field-of-view change on the current recipe.** Neither a
  tissue-centred crop nor a wider field, and no per-plane spacing, is pursued. Tissue centring
  moves the window a few millimetres and never vertically, because the leg crosses the top and
  bottom edges. Widening costs resolution, and the last field change, narrowing to 140 mm, was a
  gain. Per-plane spacing would also give one structure a different pixel size in each plane, and
  per-plane pixel sizes would break shared batching. The targets that could plausibly reach past
  140 mm are anatomical guesses, not measurements: a Baker's cyst dissecting into the calf, a
  distended suprapatellar recess (effusion), a low MCL tibial attachment (about 5–7 cm below the
  joint line against the coronal window's median 53 mm) and fibular-head or metaphyseal
  fractures. If the field is ever revisited, as part of
  [bigger images](../backlog/endgame.md#bigger-images), choose its size from measured extent of
  those structures rather than from an AUC sweep.
- **Joint line:** a median 12.7 mm (sagittal) and 16.8 mm (coronal) below the native image
  centre, more than 25 mm off in 6% and 16%. The heuristic is cross-checked within 10 mm
  between planes in 96–98% of studies.
- **Narrower than 140 mm:** [what a 120 or 130 mm field cuts](#what-a-narrow-field-cuts).
- **Past crop results agree:**
  - 150 → 140 mm at 224 px (0.67 → 0.625 mm) gave +0.003 local and +0.007 public.
  - 130 and 128 mm crops tied overall but lost on MCL, Baker's, effusion and medial meniscus.
  - That 140-mm gain also raised resolution, so the two effects are confounded.
- Resolution at a fixed field of view has never been tested here.

### What a narrow field cuts

Measured on 2026-10-01 for the mixed-field campaign draft (a 160 mm view beside a narrower one,
both at 224 px), with [`check_narrow_field_fit`](../../scripts/adhoc/check_narrow_field_fit.py):

```bash
rtk proxy uv run --locked python -m scripts.adhoc.check_narrow_field_fit \
  --output-dir outputs/analysis/n/narrow-field-fit-20261001
```

It reuses the tissue edges and joint-line offsets of the 300-series sample above (292 with a usable
mask, middle slice, image-centred fields as the loader crops them) and draws 24 series per plane
with the fields overlaid. Artifacts: `clip_depth.csv`, `joint_margin.csv` and three contact sheets
in `outputs/analysis/n/narrow-field-fit-20261001/`. Shares are population-weighted; with 96–99 series
per plane a share near 30% carries about ±9 points.

**An image-centred 120 mm field (0.536 mm) holds the tibiofemoral joint and cuts the periphery.**

| Tissue cut, measured from the skin | 120 mm | 130 mm | 140 mm |
|---|---:|---:|---:|
| Sagittal anterior (patella side), over 10 mm | 38% | 17% | 8% |
| Sagittal anterior, over 20 mm | 8% | 1% | 0% |
| Sagittal posterior (popliteal fossa), over 20 mm | 34% | 7% | 1% |
| Coronal medial or lateral, over 10 mm | 30–31% | 13% | 5–6% |
| Coronal medial or lateral, over 20 mm | 5–6% | 0–1% | 0% |
| Axial anterior, over 10 mm | 11% | 4% | 3% |
| Axial posterior, over 20 mm | 11% | 7% | 5% |

| Tibia left below the joint line | 120 mm | 130 mm | 140 mm |
|---|---:|---:|---:|
| Median, sagittal / coronal | 47 / 45 mm | 52 / 50 mm | 57 / 55 mm |
| Under 40 mm, sagittal / coronal | 22% / 28% | 8% / 13% | 3% / 4% |
| Under 30 mm, sagittal / coronal | 3% / 4% | 1% / 3% | 0% / 2% |

- **Contact sheets (qualitative, 72 tiles at about 320 px each).** On every knee-centred tile the
  condyles, tibial plateau, menisci and cruciates lie inside the 120 mm window. Its anterior edge
  runs at or through the front of the patella on a minority of sagittal and axial tiles, with the
  patellar cartilage still inside, and it cuts the popliteal soft tissue on most. Five axial tiles
  are not knee-centred (100 mm and 300 mm stored fields) and say nothing about the field.
- **Reading the depths is a guess.** Which structure a given depth reaches (patellar cortex,
  collateral ligaments, a popliteal cyst) depends on subcutaneous thickness, which is not
  measured; no bone or ligament mask exists.
- **Image-centred placement suits a narrow field.** With the joint line 13–15 mm below the image
  centre, a 120 mm window keeps a median 73–75 mm above it, where the patella and suprapatellar
  recess are, and 45–47 mm below; centring on the joint line would take room from the patella.
- **Consequence for the campaign draft.** 120 mm is a detail view of the central joint (cruciates,
  menisci, tibiofemoral and patellar cartilage). The labels read at the periphery (Baker's cyst,
  MCL, fracture below the plateau) depend on the 160 mm view, so a design that shows a slice at
  120 mm only loses them on that slice. 130 mm halves most of the cuts for 8% coarser pixels.
- **Acquired resolution by field** (header audit, 23,173 series, counted 2026-10-01 from
  `acquisition/series_acquisition.parquet`): the share of series whose finer acquired axis is
  at or above the pixel size is 15% at 140 mm, 30% at 120 mm, 50% at 112 mm (0.5 mm exactly,
  the most common value, 13% of series), 70% at 100 mm and 91% at 84 mm. T1/PD series are
  acquired finer than fluid-sensitive ones: 36% against 61% resolved at 112 mm.
- **Outcome:** 120 mm was the narrow field of the
  [mixed-field campaign](../../autoresearch/campaigns/mixed-field/INDEX.md) and was adopted for
  every series on 2026-10-01; the periphery labels moved little (MCL and fracture flat, Baker's
  cyst −0.003 to −0.005, within noise), and the medial meniscus fell in every 120 mm run
  ([result](../validation/results/2026-10-01-mixed-field-campaign.md#what-the-runs-established)).

## Evidence from the crop runs

The [crop-selection investigation](../history/crop-selection-investigation.md#what-the-runs-established)
changed resolution, field of view, slice coverage and architecture together. It is therefore
circumstantial evidence, from fold 0 and a single seed.

- **0.5 mm crops helped central detail targets.**
  - *Per-label crops:* 224 px windows at 0.5 mm on 75% of slices scored 0.891703 (−0.0011
    against `00-baseline`).
  - *Gains:* ACL +0.0111, synovitis +0.0077 and medial OA +0.0057.
  - *Losses:* edge targets such as the lateral meniscus (−0.0102) and fracture (−0.0085), whose
    slices or regions the crops left out
    ([result](../validation/results/2026-09-16-per-label-crops-result.md#per-target-auc)).
- **0.875 mm on every slice cost detail targets.**
  - *Setup:* learned zoom's ReXNet-200 evidence was in effect at 0.875 mm on every slice.
  - *Result:* 0.869951 (−0.0228).
  - *Losses:* the lateral meniscus −0.1057, MCL −0.0671 and ACL −0.0223.
  - *Diffuse findings held:* effusion +0.0055 and contusion +0.0004
    ([result](../validation/results/2026-09-16-learned-zoom-result.md#per-target-auc)).
- **Coverage mattered as much as resolution.** Runs that cropped about a quarter of the slices
  at 0.5 mm scored 0.8796 (shared pooling) and 0.8451 (private pooling).

## Depth sampling

- **Gaps:** at 1× density each series gives 5–6 triplets with a median gap of 16.5–16.9 mm
  (4–5 slices), and 36–58% of candidate slices fall inside some triplet. 2× halves the gap
  (8–9 mm) and covers 71–88%.
- **Clinical rule:** radiologists require a meniscal tear to appear on two images, which
  argues against sparse slice sampling for menisci.

## External evidence

- **Clinical protocols:** 3–4 mm slices, FOV 130–180 mm, and in-plane 0.35–0.7 mm acquired.
  ACR accreditation requires a pixel area ≤0.6 mm². Zero-filled interpolation adds no
  information.
- **Knee deep learning:** region crops beat full-scale inputs. Tack 2021: medial anterior
  horn AUC 0.74 → 0.87 → 0.94 for full scale → crop → crop with box loss. No knee
  resolution ablation exists.
- **Kaggle winners:** 384–512 px whole images with 224–384 px region crops (lumbar spine,
  abdominal trauma). None quantified a resolution gain.
- **Public knee notebooks:** 336 px over 130–150 mm. Unverified working notes report 384 px
  / 140 mm best, and 224 px −0.003.
- **Public slice-density ablation (2026-09-18):** on one CoAtNet pipeline, 44 slices spread over
  2–98% of each series scored 0.917 against 0.926 for the same 44 over 6–94%, and restoring
  density at the wide span gave 0.928 at 64 slices and 0.932 at 80; a competitor above 0.94 puts
  the field-of-view sweet spot at 140–160 mm and calls 224–288 px sufficient
  ([public research](public-research.md#slice-density-and-the-board-2026-09-18-re-sweep)).
- **CNN scaling:** doubling resolution (4× compute) gives about +1.5–2.8 ImageNet points,
  flattening at high resolution. Small medical findings benefit most from resolution
  (chest X-ray nodules). Downsampling must be antialiased; the repository resampler is.

## Compute constraints

- **Pixel cost:** backbone compute and activations scale with pixels: 288² ≈ 1.65× and
  320² ≈ 2.04× of 224². ReXNet-200 measured 2.462 GMACs per triplet at 280 px against 1.526 at
  224 px, which is 1.61× for 1.56× the pixels
  ([budget](../modeling/backbone-swaps.md#current-budget)).
- **Training memory:** training already peaks at 20.5 GB of 24 GB at batch 8 (`00-baseline`:
  20.02 GB), so a 1.65× input needs batch 4 with gradient accumulation 2, or fewer triplets.
- **Kaggle:** inference runs on two T4s with 4 studies per GPU, in FP16 with fixed 64-image
  backbone chunks. The backbone forward is about a quarter of prediction time at 16 loader
  workers ([profiling](../inference/2026-09-15-kaggle-inference-profiling.md)).
