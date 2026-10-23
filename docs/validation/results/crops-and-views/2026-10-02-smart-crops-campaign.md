# 2026-10-02 smart-crops campaign and the 84/93 mm view adoption

**A finer view in place of the wide one gains; an added view or crop does not; where a window
sits barely matters.** The `smart-crops` autoresearch campaign gave the ResNet-18 recipe finer
pixels on the joint and tried to let the model choose where to look. Adopted on 2026-10-02 (user
decision): **coronal and sagittal T1/PD at 84 mm (0.375 mm) and coronal and sagittal fluid at
93.33 mm (5/12 mm), in place of the 120 mm view, placed by rule; axial stays at 120 mm**
(`data.views.policy: detail_84_93`). ×4 0.932880 against 0.930666 for the 120 mm control
(+0.0022, 80% [+0.0007, +0.0037]), and **0.950 public (S93) against 0.947 (S92)**, with the
same fold, seed and inference settings.

Fold 0 of `all_5fold_scanner_v2`, `rebuild_state_v2` labels, seed 1533697213, ×4 predictions,
paired report-group bootstrap with 2,000 draws. The reference is `rn18-field-120-control`
(×4 0.930666). Every result is one run at one seed; one run's training noise is about 0.0006 on
the macro and 0.0008 to 0.0028 on a label
([regression](#measurements-made-for-the-campaign)). Campaign record, seeds, contract and keeper
repairs: [smart-crops campaign](../../../../autoresearch/campaigns/smart-crops/INDEX.md). Per-label
effects of every change, what each kind of token reads alone and the details of each run:
[view effects by target](../../../../autoresearch/campaigns/smart-crops/view-effects-by-target.md).
Artifacts in `outputs/autoresearch/smart-crops/`.

## Campaign runs

Δ and interval are against the 120 mm control. "Against its reference" is against the campaign's
incumbent when the run was judged (`t1pd-detail-84` from the second run on,
`fluid-detail-93-replacement` after its adoption). All views are 224 px unless a crop is named.

### A view replaced by another

| Run | View policy | ×4 macro | Δ | 80% interval | Against its reference | Training s |
|---|---|---:|---:|---|---:|---:|
| `t1pd-detail-84` | Coronal and sagittal T1/PD at 84 mm, rule-placed | 0.931918 | +0.0013 | [+0.0001, +0.0025] | +0.0013 | 1,077 |
| `t1pd-detail-acquisition-matched` | T1/PD width 84 to 120 mm per series, by stored pixel | 0.931728 | +0.0011 | [−0.0001, +0.0022] | −0.0002 | 1,134 |
| `t1pd-detail-acquired-matrix` | T1/PD width per series, by acquired matrix | 0.931373 | +0.0007 | [−0.0002, +0.0016] | −0.0005 | 1,112 |
| `t1pd-foveated-native-view` | T1/PD at 120 mm with a fine 80 mm core | 0.931607 | +0.0009 | [−0.0000, +0.0019] | −0.0003 | 1,178 |
| `t1pd-detail-dense-final-grid` | T1/PD at 84 mm with a 14×14 token grid | 0.931743 | +0.0011 | [−0.0001, +0.0023] | −0.0002 | 1,199 |
| **`fluid-detail-93-replacement`** | **T1/PD 84 mm, coronal and sagittal fluid 93 mm** | **0.932880** | **+0.0022** | [+0.0007, +0.0037] | +0.0010 | 1,134 |
| `fluid-foveated-native-view` | Fluid at 120 mm with a fine 80 mm core | 0.931927 | +0.0013 | [−0.0005, +0.0031] | −0.0010 | 1,296 |
| `axial-130-detail-84-93` | Adopted policy with axial at 130 mm | 0.932848 | +0.0022 | [+0.0007, +0.0037] | −0.0000 | 1,175 |
| `axial-140-detail-84-93` | Adopted policy with axial at 140 mm | 0.932259 | +0.0016 | [−0.0000, +0.0031] | −0.0006 | 1,163 |
| `coronal-fluid-93-sagittal-fluid-120` | Adopted policy with sagittal fluid back at 120 mm | 0.932186 | +0.0015 | [+0.0001, +0.0028] | −0.0007 | 1,154 |
| `sagittal-fluid-93-coronal-fluid-120` | Adopted policy with coronal fluid back at 120 mm | 0.932461 | +0.0018 | [+0.0004, +0.0031] | −0.0004 | 1,183 |
| `sagittal-fluid-93-centred` | Adopted policy without the sagittal fluid forward shift | 0.932729 | +0.0021 | [+0.0006, +0.0035] | −0.0002 | 1,162 |
| `sagittal-fluid-100` | Adopted policy with sagittal fluid at 100 mm | 0.932719 | +0.0021 | [+0.0005, +0.0036] | −0.0002 | 1,221 |
| `coronal-t1pd-84-only` | Adopted policy with sagittal T1/PD back at 120 mm | 0.931699 | +0.0010 | [−0.0003, +0.0024] | −0.0012 | 1,173 |

### A view or crop added

| Run | What was added | ×4 macro | Δ | 80% interval | Against its reference | Training s |
|---|---|---:|---:|---|---:|---:|
| `crop-rule` | 60 mm T1/PD crop at 160 px beside the 120 mm view, rule-placed | 0.930904 | +0.0002 | [−0.0011, +0.0015] | −0.0010 | 1,342 |
| `crop-attention` | The same, placed by the model's attention | 0.930333 | −0.0003 | [−0.0018, +0.0011] | −0.0016 | 1,359 |
| `crop-own-loss` | The same, with a loss on the crop tokens alone | 0.930412 | −0.0003 | [−0.0016, +0.0010] | −0.0015 | 1,367 |
| `crop-compartments` | Two crops per coronal T1/PD slice, medial and lateral | 0.930825 | +0.0002 | [−0.0014, +0.0017] | −0.0011 | 1,410 |
| `crop-evidence-utility-placement` | Crop placed by evidence utility | 0.930338 | −0.0003 | [−0.0017, +0.0011] | −0.0016 | 1,373 |
| `crop-rule-overlap-replacement` | Crop with the wide tokens under it removed | 0.929959 | −0.0007 | [−0.0022, +0.0007] | −0.0020 | 1,361 |
| `crop-context-teacher-distillation` | Crop as a training teacher, 120 mm only at prediction | 0.930231 | −0.0004 | [−0.0017, +0.0009] | −0.0017 | 1,374 |
| `crop-full-context-ranking-locator` | Crop placed by a ranking-trained locator | 0.930737 | +0.0001 | [−0.0013, +0.0014] | −0.0012 | 1,562 |
| `two-views-fluid-140` | Fluid slices also at 140 mm through a ResNet-10t | 0.930763 | +0.0001 | [−0.0010, +0.0012] | −0.0012 | 1,685 |
| `two-views-112-fluid-140` | The same with every main view at 112 mm | 0.930353 | −0.0003 | [−0.0018, +0.0012] | −0.0016 | 1,680 |
| `sagittal-fluid-detail-rectangle` | 93×67 mm sagittal fluid window beside the 120 mm view | 0.931685 | +0.0010 | [−0.0005, +0.0024] | −0.0002 | 1,454 |
| `sagittal-fluid-correspondence-fusion` | The same, fused into the main tokens | 0.931919 | +0.0013 | [−0.0003, +0.0027] | +0.0000 | 1,477 |
| `sagittal-fluid-content-residual` | The same, fused as a learned residual | 0.932006 | +0.0013 | [−0.0001, +0.0028] | +0.0001 | 1,474 |
| `t1pd-shallow-detail-fusion` | 120 mm T1/PD kept, 84 mm fused in at stride 8 | 0.930077 | −0.0006 | [−0.0018, +0.0005] | −0.0018 | 1,303 |
| `sagittal-fluid-wide-for-bakers` | Adopted policy plus a 120 mm sagittal fluid view read only by Baker's cyst | 0.932573 | +0.0019 | [+0.0004, +0.0035] | −0.0003 | 1,424 |

`t1pd-detail-84` was the campaign's one accept by its rule (+0.001 with an 80% lower bound above
zero). `fluid-detail-93-replacement` missed the rule by a rounding error (+0.00096, lower bound
−0.00002) and was adopted by the user after S93. Seven first attempts stopped before producing a
score, most on a launch crash in the candidate's own code, and were repaired automatically. Two
T1/PD plane-split attempts were
blocked at review by the harness's dependency gate; one was requeued and run
(`coronal-t1pd-84-only`). Not run when the user stopped the campaign: its mirror
(`sagittal-t1pd-84-only`), T1/PD at 93 mm, compartment crops routed to the OA labels, the
sagittal windows without the downward shift, and separate normalisation statistics per pixel
size; these moved to
[untried ideas](../../../backlog/untried-ideas.md#views-and-fields); the campaign that
followed learned the views instead ([learnable views](2026-10-03-learnable-views-campaign.md)).

## What the runs established

- **Replacing a view by a finer one gains.** Five replacements of the T1/PD view beat the control
  by +0.0007 to +0.0013, and zooming the fluid series added +0.0010 on top. By the regression
  the three zooms together are worth +0.0013 (80% [+0.0000, +0.0026]); the measured pair says
  +0.0022.
- **Adding a view or crop ties.** Fifteen runs at 13 to 47% more training time: the ten crop and
  context-view runs are within −0.0007 to +0.0002 of the control, the three sagittal fluid
  windows within 0.0002 of `t1pd-detail-84`, the routed Baker's cyst view −0.0003 against the
  adopted policy, and the fused T1/PD view −0.0018. Tokens that enter every label's pool help
  the OA labels and synovitis and cost Baker's cyst, the menisci and contusion.
- **Where a window sits barely matters.** Rule, attention, per compartment, evidence utility and a
  ranking-trained locator are within 0.0006 of each other. The ranking run measured it: two
  candidate crop positions differed in loss by 0.003 and the locator's agreement with the better
  one was 49.9%.
- **The window's shape does not matter.** A plain uniform window ties or beats a fisheye window,
  a width matched to each series and a denser token grid.
- **The two fluid planes add up:** coronal zoom +0.0003, sagittal zoom +0.0005, both +0.0010.
- **The sagittal fluid zoom is the one real conflict.** It carries synovitis, PF OA, lateral OA,
  the lateral meniscus and medial OA, and costs Baker's cyst 0.0047 to 0.0090. Read alone at ×1,
  axial plus sagittal fluid at 120 mm scores Baker's cyst 0.949, the zoomed fluid tokens 0.878.
  No width between 93 and 120 mm serves both (100 mm: Baker's cyst −0.0020), removing the
  forward shift does not help, and a wider axial view does not either (axial tokens read it at
  0.935 to 0.936 at 120, 130 and 140 mm).
- **Routing a view to one label works mechanically and its payoff is unproven.** With a 120 mm
  sagittal fluid view read only by Baker's cyst, that label recovered about half its loss
  (+0.0023, 80% [−0.0014, +0.0067]) and the macro tied (−0.0003) at 26% more time.
- **The sagittal T1/PD zoom carries most of the T1/PD gain:** putting it back at 120 mm cost
  0.0012 (80% [−0.0020, −0.0003]), with the lateral meniscus −0.0061 and ACL −0.0027.
- **The sagittal forward shift earns its place:** without it on fluid, lateral OA and PF OA fall
  back to the control's level.
- **Lateral OA rose with a T1/PD crop in all six crop runs** (+0.0032 to +0.0054); medial OA did
  not move with T1/PD crops and rose only with finer fluid views.
- **Axial width does not matter between 120 and 140 mm** (−0.0000, −0.0006).
- **Two labels of the control look off:** synovitis rose in every run against it and the lateral
  meniscus fell in almost every one.

## Measurements made for the campaign

- **Attention and saliency probes** of a 140 mm model
  (`outputs/analysis/l/label-attention-20261001/`; `scripts/adhoc/dump_label_attention.py`,
  `analyze_label_attention.py`, `dump_input_saliency.py`, `analyze_input_saliency.py` and
  `analyze_attention_by_series_type.py`): where each label's attention sits, per plane and
  contrast.
- **Spacing review** (`outputs/analysis/v/view-spacing-20261002/report.md`): acquired resolution,
  stored field and tissue cut per series type on all 24,371 training series
  (`scripts/adhoc/measure_series_type_resolution.py`, `measure_view_window_cut.py` and
  `summarize_view_window_cut.py`). 84 mm and 93 mm
  keep the acquired detail of about nine in ten coronal and sagittal series; the next coarser
  candidate loses 10 to 17 points. A second, image-based review is in
  [plane and fluid spacing](../../../data-preparation/plane-fluid-spacing-review.md).
- **Slice supply** (`outputs/analysis/s/slice-budget-20261002/`, `scripts/adhoc/measure_slice_capacity.py`): every series type has a median 28
  to 32 slice positions per study against 5 to 6.5 given at ×1, so supply never limits the
  per-type budget.
- **Regression of width per type against AUC per label**
  (`outputs/analysis/v/view-regression-20261002/report.md`, 17 single-width runs): the adopted
  widths are the best combination among tested widths; every single-type change from them is
  predicted 0.0001 to 0.0005 lower and none is distinguishable from zero. Untested: anything
  finer than 84 mm on T1/PD, 93 mm on fluid or 120 mm on axial. The per-run label table that fed
  it, `scripts/adhoc/tabulate_view_run_labels.py`, was deleted; the last commit holding it is
  `d8def44:scripts/adhoc/tabulate_view_run_labels.py`.

## Adoption on main

- **Recipe:** `configs/resnet50.yaml` carries `data.views.policy: detail_84_93`; the code is the
  campaign run's frozen source for eight files (`config.py`, `data/dataset.py`,
  `data/preprocessing/preparation.py` and `spatial.py`, `inference/engine.py`,
  `modeling/model.py` and `spatial_tokens.py`, `training/lightning.py`).
- **Port check:** the S93 checkpoint exported from the ported code gives the same `model.ckpt`,
  `config.json` and preprocessing identity as the submitted bundle; the only bundle files that
  differ are two autoresearch harness files changed the same day.
- **Control:** `fluid-detail-93-replacement`
  (`outputs/autoresearch/smart-crops/experiments/fluid-detail-93-replacement-20261002T061326Z-c4e4d640/training`),
  ×4 0.932880, ×1 0.929261, 1,134 s, 0.950 public (S93). It was trained from the campaign's
  frozen source; a retrain from main is the user's to launch.
- **Known cost:** Baker's cyst, 0.9414 against 0.9454 for the 120 mm control.
