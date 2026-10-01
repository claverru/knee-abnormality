# 2026-10-01 mixed-field campaign and the 120 mm adoption

**A narrower, finer field helps; a wider one does not.** The `mixed-field` autoresearch campaign
tested feeding the ResNet-18 recipe a 160 mm view beside a narrower one at the fixed 224 px input.
160 mm added nothing in five designs; every run that used 120 or 112 mm came out ahead of the
reference. Adopted on 2026-10-01 (user decision): **120 mm on every series**
(`data.spacing_mm: 0.5357`, from 0.625 and 140 mm), 0.947 public (S90) against 0.943 (S86), and
**augmentation and mirror TTA on the middle slice only**, which is neutral on score and faster.

Fold 0 of `all_5fold_scanner_v2`, `rebuild_state_v2` labels, seed 1533697213, ×4 predictions at
τ 1, paired report-group bootstrap with 2,000 draws. The reference is `rn18-l2sp-no-confusion`
(×4 0.928062), the `configs/training.yaml` recipe of 2026-09-30, so none of these runs carries the
plane-contrast readouts adopted on 2026-10-01. Every result is one run at one seed. Campaign
record, seeds, contract and keeper repairs:
[mixed-field campaign](../../../autoresearch/campaigns/mixed-field/INDEX.md); artifacts in
`outputs/autoresearch/mixed-field/` (`experiments/` for the campaign, `manual/` for the two runs
after it stopped).

## Campaign runs

Deltas against the reference; the campaign's own verdicts were against its incumbent at the time.

| Run | Field policy | Δ ×4 macro | 80% interval | Training s |
|---|---|---:|---|---:|
| `field-160-single-logged` | 160 mm on every slice | −0.0034 | [−0.0051, −0.0019] | |
| `field-120-single` | 120 mm on every slice | +0.0011 | [−0.0004, +0.0026] | 1,159 |
| `field-jitter` | Training slices at 120, 140 or 160 mm at random, prediction at 140 mm | +0.0005 | [−0.0004, +0.0014] | |
| `mixed-field-140-160` | Slices alternate 140/160 mm by depth, field embedding | −0.0013 | [−0.0023, −0.0003] | |
| `mixed-field-120-160` | Slices alternate 120/160 mm by depth, field embedding | +0.0006 | [−0.0007, +0.0018] | |
| `mixed-field-readout-weights` | As above with per-field readout weights in the joint pool | −0.0001 | [−0.0016, +0.0013] | |
| `plane-weighted-fields` | 160 mm on half of the axial and a quarter of the sagittal slices, 120 mm elsewhere | +0.0011 | [−0.0003, +0.0024] | |
| `extent-ruled-field` | 140 mm, or 160 mm for a series whose tissue overhangs the 140 mm window | −0.0017 | [−0.0027, −0.0007] | |
| `field-backbones-routed` | 120/160 mm alternating, one ResNet-18 per field | −0.0012 | [−0.0027, +0.0003] | |
| `contrast-split-112-140` | T1/PD series at 112 mm (0.5 mm), fluid-sensitive series at 140 mm | **+0.0015** | [+0.0004, +0.0026] | 1,273 |
| `two-stream-paired` | Every slice at 120 mm (ResNet-18) and 140 mm (ResNet-10t); the second stream stayed off | +0.0023 | [+0.0007, +0.0039] | 1,736 |

`contrast-split-112-140` was the campaign's one provisional accept. The original 160 mm yardstick
blocked on a seed-wording conflict and two `two-stream-paired` attempts were stopped for exceeding
the time budget; none of the three produced a score.

## What the runs established

- **160 mm adds nothing at 224 px.** It lost alone and in every mix, including when given only to
  the series whose tissue overhangs the 140 mm window. In the mixes the 160 mm tokens alone scored
  below the narrow ones at ×1 (0.9035 against 0.9079 with 140 mm; 0.9024 against 0.9099 with
  120 mm). At a fixed image size 160 mm costs resolution (0.714 mm); 160 mm at the same pixel size
  is the [bigger-images](../../backlog/endgame.md#bigger-images) test.
- **120 mm beats 160 mm directly:** +0.0046 (95% [+0.0016, +0.0076]), menisci +0.0086, the three
  OA labels +0.0075 (`outputs/analysis/mixed-field-120-vs-160`). Per label 120 mm leads clearly on
  the lateral meniscus (+0.0161), ACL (+0.0134), medial OA and effusion; 160 mm leads outside
  noise only on synovitis (+0.0097).
- **120 mm alone is worth about +0.0011 to +0.0012,** measured twice (`field-120-single` and the
  manual run below). Its gains are on PF OA, ACL and the OA labels; synovitis, Baker's cyst and
  the medial meniscus fall. The medial meniscus fell in every run that used 120 mm (−0.002 to
  −0.007).
- **The contrast split protects some of those labels** (synovitis recovered) but is not
  distinguishable from 120 mm alone (−0.0003, 80% [−0.0017, +0.0011]) and needs a per-series field
  policy through the loader and model.
- **Two views per slice are untested.** `two-stream-paired` pooled
  `narrow + tanh(gate) × (joint − narrow)` with one scalar gate initialized at zero; the wide
  backbone only receives gradient through that gate and never trained. Its joint ×1 AUC equals its
  narrow-only AUC to five decimals at every epoch and the wide-only AUC is chance (0.506). The run
  is in effect 120 mm alone; its extra 0.001 over the other two 120 mm readings is within the
  usual run-to-run noise.
- **Slice selection barely depends on the field.** With the ranking radius following the field,
  98.5% of selected slices are unchanged at 120 mm and 98.4% at 160 mm (150 studies), always by
  swapping to the sibling series at the same depth.

## Middle-slice augmentation

With `model.slice_channels: repeated_middle` the model reads only the middle slice of each
three-slice input, but reflection, contrast mixing and augmentation ran on all three. A profile of
30 training steps (synchronous CUDA) put augmentation at 26% of step time, bilateral smoothing
alone at 14%. Selecting the middle slice before those steps was run on two recipes after the
campaign stopped, each from a frozen copy of its parent's source with that one change:

| Run | Parent | ×4 macro | Δ vs parent (80%) | Δ vs reference (80%) | Training s (parent) | Peak GPU |
|---|---|---:|---|---|---:|---:|
| `middle-slice-augmentation-20261001` | `contrast-split-112-140` (0.929570) | 0.929212 | −0.0004 [−0.0014, +0.0007] | +0.0012 [−0.0002, +0.0025] | 1,103 (1,273) | 3.7 GiB |
| `field-120-middle-slice-augmentation-20261001` | `field-120-single` (0.929197) | 0.929301 | +0.0001 [−0.0011, +0.0013] | +0.0012 [−0.0003, +0.0028] | 1,095 (1,159) | 3.7 GiB |

Neutral on score in both, with no label outside noise, and 107–110 s per epoch against 115–119 s.
Evaluation is unchanged: on the parent's checkpoint the plain and mirror-TTA logits are
bit-identical between old and new code (16 studies each).

## Public check

Same inference settings as S86 (×4, τ 0.9, batch 4, no TTA), each bundle exported from its run's
frozen source.

| Submission | Model | ×4 local | Public |
|---|---|---:|---:|
| S86 | reference, 140 mm | 0.928062 | 0.943 |
| S90 | 120 mm, middle-slice augmentation | 0.929301 | **0.947** |
| S91 | contrast split (T1/PD 112 mm, fluid 140 mm), middle-slice augmentation | 0.929212 | 0.945 |

Earlier ResNet-18 variants of the reference recipe scored 0.942–0.944 (S81, S82, S84, S86, S87).

## Adoption

User decision, 2026-10-01, on the public pair S90 against S86:

- **`data.spacing_mm: 0.5357142857142857`** (120 mm at 224 px) in `configs/training.yaml` and in
  `configs/refined-resnet50.yaml`, which is the same recipe by definition. It is untested on the
  ResNet-50; the ReXNet recipe keeps 0.625 mm.
- **Middle slice only** in `repeated_middle` mode: `encode_batch` selects it before reflection,
  contrast mixing and augmentation, and `reflect_batch` (mirror TTA) flips only that slice, which
  for sagittal series means no image work at all. Adjacent-slice models keep the old path. On the
  current control's checkpoint (`no-anatomical-bias`) plain and mirrored logits are bit-identical
  between the code before and after the change, and likewise for an identically seeded
  adjacent-slice model. `modeling/model.py` and `inference/reflection.py` are hashed preprocessing
  sources, so earlier checkpoints export from their archived sources.
- **Not adopted:** the contrast split, any use of 160 mm, per-field readout weights, one backbone
  per field and field jitter.
- **Control:** not retrained yet. Both evidence runs predate the plane-contrast readouts, so the
  recipe now in `configs/training.yaml` has never been trained
  ([next](../../ongoing/next.md)).

## Open

- **Two views per slice** needs a run in which both streams feed the pool from the start.
- **Medial meniscus at 120 mm:** it is read mainly from coronal series, where a 120 mm window cuts
  more than 10 mm from a side in 30% of series
  ([geometry](../../research/input-resolution.md#what-a-narrow-field-cuts)); cause not
  established.
- **Unused slices are still decoded and resampled.** The loader reads three slices per input and
  the device resamples all three; only the middle one is used.
- **Harness:** an activation threshold of "above zero" let a mechanism that never acted pass;
  `data.study_triplet_total` is a floor, not a cap, so lowering it to 20 changed little (about
  25.4 slices per study against 26.8); a rule that new projections start at zero blocks learning
  when a whole stream sits behind one gate.
