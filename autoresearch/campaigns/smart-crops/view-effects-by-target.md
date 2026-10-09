# Which view serves which target

Running notes of the smart-crops campaign, started 2026-10-02 and updated with every result:
the ledger of design choices, the effect of every view change per label, what each kind of token
reads alone, and the details of each run.

## How to read this

- **Views.** Every view is 224 px, so width = 224 x pixel size. "Zoom" means a narrower window
  at a finer pixel, placed by rule (coronal on the knee centre, sagittal 7 mm forward, both 10 mm
  down). The six series types and their share of a study's 26 slices: axial fluid 24%, axial
  T1/PD 5%, coronal T1/PD 15%, sagittal T1/PD 19%, coronal fluid 19%, sagittal fluid 19%.
- **Numbers.** x4 AUC on fold 0 (912 studies), candidate minus the named reference. Every row is
  one run; all runs share the fold and the seed.
- **Noise.** One run moves a single label by up to about 0.005, and identical changes have
  differed by up to 0.0067 on one label. MCL (48 positives), fracture (59) and lateral OA (95)
  are the noisiest. **Bold** marks a change of 0.003 or more; it is not a significance mark.
  Trust a pattern that repeats across runs, not one cell.
- **Two labels of the 120 mm control look off.** Synovitis rose in every run against it and the
  lateral meniscus fell in almost every one, so read those two against the incumbent, not the
  control.
- **Positives of 912:** ACL 132, MCL 48, medial meniscus 344, lateral meniscus 136, medial OA 190,
  lateral OA 95, PF OA 292, effusion 244, synovitis 113, Baker's cyst 113, contusion 92,
  fracture 59.
- **Regenerate the tables:** `uv run --locked python <scratchpad>/view_effects.py`.

## Where each design choice stands

The goal is one recipe that keeps every gain found so far. This is the running ledger; each row
changes only when a run or a measurement moves it.

| Choice | Current best | Evidence | Status |
|---|---|---|---|
| Coronal T1/PD width | 84 mm (0.375 mm) | with sagittal T1/PD: +0.0013; 92% of series fully resolved | open: never changed alone; the 120 mm and 93 mm runs were built or queued and not run |
| Sagittal T1/PD width | 84 mm (0.375 mm) | back at 120 mm: -0.0012 (80% [-0.0020, -0.0003]), lateral meniscus -0.0061, ACL -0.0027 | settled against 120 mm; 93 mm untested |
| Coronal fluid width | 93 mm (5/12 mm) | +0.0003 alone, +0.0004 with sagittal zoomed; ACL up in both | settled, small |
| Sagittal fluid width | 93 mm | zoom +0.0005 and +0.0007; Baker's cyst -0.0090 and -0.0047; 100 mm is no compromise (-0.0002, Baker's cyst -0.0020); a routed 120 mm view for Baker's cyst ties (-0.0003) | settled at 93 mm for now; the Baker's cyst cost stays open |
| Axial width | 120 mm | 130 mm -0.0000, 140 mm -0.0006; axial tokens read Baker's cyst the same at all three (0.935-0.936) | settled |
| Sagittal horizontal placement | 7 mm forward of the knee centre | centred: -0.0002; PF OA and lateral OA fall back to the control's level; Baker's cyst does not recover | settled: keep the shift |
| Sagittal vertical placement | 10 mm below the image centre | untested | open: `sagittal-no-vertical-shift` queued |
| Window shape | plain uniform window | fisheye ties on T1/PD and on fluid and costs fracture and Baker's cyst | settled |
| Width per series | one fixed width per type | two per-series variants: -0.0002, -0.0005 | settled |
| Token grid on zoomed views | 7x7 | 14x14: -0.0002 | settled |
| Placement method | fixed rule | five learned or attention placements within 0.0006 of the rule; swapping crop position changes the loss by 0.003 | settled |
| Extra crops or views in every label's pool | none | twelve runs, all ties; cost Baker's cyst, the menisci, contusion | settled: only routed by label from now on |
| Extra tokens routed to chosen labels | works mechanically; payoff unproven | `sagittal-fluid-wide-for-bakers`: routing exact, Baker's cyst +0.0023 (half the gap, interval includes zero), macro -0.0003 at 26% more time | open: one run cannot resolve a gain of this size; carry routing into the next campaign as a learned bias |
| Scale handling in the encoder | shared statistics | zoomed-view embedding stayed small (norm 0.10) | open: `spacing-specific-batch-statistics` queued |
| Slices per series type | 6 axial, 5 others, per contrast | supply is four to five times the budget in every type | open: need per label being measured on Kaggle |

## Summary per target

"Read on" is where the incumbent finds the label when each kind of token is scored alone
([tables](#what-each-kind-of-token-reads-on-its-own)).

| Target | Read on | Helped by | Hurt by |
|---|---|---|---|
| ACL | zoomed coronal/sagittal views (fluid 0.943, T1/PD 0.932 alone); axial is poor (0.772) | T1/PD zoom (+0.0020); fluid zoom (+0.0015), mostly coronal | nothing consistent |
| MCL | axial (0.879) and fluid (0.865) | rule-placed T1/PD crop (+0.0024 to +0.0033, three runs) | sagittal fluid zoomed or given a detail window (-0.0025 to -0.0062, four runs); main view at 112 mm (-0.0153) |
| Medial meniscus | zoomed fluid (0.945), then T1/PD (0.919); axial is poor (0.701) | T1/PD at 84 mm (+0.0020) | giving width back to T1/PD (-0.0017 to -0.0039, three runs); every added crop (-0.001 to -0.002) |
| Lateral meniscus | zoomed fluid (0.916), then T1/PD (0.881) | sagittal fluid zoom (+0.0024) | T1/PD zoom (-0.0020); added T1/PD crops (-0.002 to -0.003 in five of six runs); a second 140 mm fluid view (-0.0053) |
| Medial OA | zoomed fluid (0.933) | anything finer on fluid slices: zoom (+0.0019), fisheye (+0.0030), sagittal detail window (+0.0018 to +0.0032), 140 mm context (+0.0028) | T1/PD crops, slightly (about -0.001) |
| Lateral OA | zoomed fluid (0.885), T1/PD (0.859) | fluid zoom (+0.0036); any T1/PD crop beside the wide view (+0.0032 to +0.0054, six runs); sagittal fluid detail window | replacing the T1/PD view by the 84 mm one (-0.0017) |
| PF OA | sagittal fluid and axial (coronal fluid alone only 0.735) | sagittal fluid zoom (+0.0024 and +0.0036, two runs), fisheye (+0.0044), sagittal detail window (+0.0028 to +0.0035) | T1/PD zoom (-0.0017); wider axial, slightly |
| Effusion | fluid everywhere (axial 0.908, zoomed fluid 0.913) | T1/PD zoom (+0.0018); 140 mm fluid context (+0.0024) | sagittal fluid detail window beside the wide view (-0.0020 to -0.0035); wider axial (-0.0018, -0.0024) |
| Synovitis | spread over all types (0.83 to 0.87 alone) | fluid zoom (+0.0054, mostly sagittal); axial at 140 mm (+0.0035); 140 mm fluid context | nothing; it rose in almost every run |
| Baker's cyst | fluid slices at a wide field: axial (0.936), plus sagittal fluid at 120 mm (0.949 together); zoomed T1/PD is blind (0.55) | sagittal fluid kept at 120 mm; a 140 mm fluid context view (+0.0054) | sagittal fluid zoom (-0.0047 and -0.0090, two runs); fisheye (-0.0088); added crops and detail windows (-0.0049 to -0.0095 in eight of nine runs) |
| Contusion | zoomed fluid (0.957) | T1/PD zoom, slightly (+0.0016) | added T1/PD crops (-0.0014 to -0.0035, six runs); fused views (-0.0051) |
| Fracture | fluid (0.923) and axial (0.908) | plain 84 mm T1/PD (+0.0078); sagittal fluid zoom (+0.0032) | any non-uniform T1/PD view: fisheye (-0.0072), per-series width (-0.0066), fusion (-0.0088) |

## Summary per view choice

| Choice | Macro | Gains | Costs |
|---|---:|---|---|
| Coronal + sagittal T1/PD at 84 mm, in place of 120 | +0.0013 | fracture, synovitis, ACL, medial meniscus, Baker's | lateral meniscus, lateral OA, PF OA (about -0.002 each) |
| Sagittal fluid at 93 mm, in place of 120 | +0.0005 alone, +0.0007 with coronal already zoomed | synovitis, the three OA labels, lateral meniscus (both runs) | **Baker's cyst -0.0090 and -0.0047** |
| Coronal fluid at 93 mm, in place of 120 | +0.0003 alone, +0.0004 with sagittal already zoomed | ACL (+0.0018, +0.0031) | nothing that repeats in both runs |
| Axial at 130 or 140 mm, in place of 120 | -0.0000, -0.0006 | synovitis (+0.0015, +0.0035) | effusion, PF OA (small); nothing for Baker's cyst |
| Fisheye (120 mm with a fine core) | 0.0000 on fluid, -0.0003 on T1/PD | on fluid: the three OA labels, synovitis | Baker's cyst, fracture, MCL |
| T1/PD width matched to each series | -0.0002, -0.0005 | PF OA | medial meniscus, Baker's, fracture |
| Extra T1/PD crop beside the 120 mm view | -0.0003 to +0.0002 | lateral OA (every run), MCL (rule-placed) | Baker's cyst, lateral and medial meniscus, contusion |
| Extra sagittal fluid detail window beside 120 mm | -0.0002 to +0.0001 | the three OA labels, synovitis, lateral meniscus | Baker's cyst, MCL, effusion |
| Extra 140 mm view of fluid slices | +0.0001 | Baker's cyst, synovitis, medial OA | MCL, lateral meniscus; 42% more training time |

The conflicts to resolve:

- **Sagittal fluid.** Zoomed, it gives most of the fluid gain (lateral meniscus, PF OA, lateral
  OA, synovitis, fracture). Wide, it gives Baker's cyst and MCL. Both cannot be had from one view.
- **Coronal and sagittal T1/PD.** 84 mm wins overall, but lateral meniscus, lateral OA and PF OA
  each give up about 0.002. The crop runs show lateral OA wants a fine T1/PD look in addition to
  the wide one, not in place of it.
- **Added tokens.** Every added crop or window ties overall because its tokens enter every
  label's pool: the labels it helps (OA) are paid for by Baker's cyst, the menisci and contusion.
  Routing by label is untested; two seeds on it are queued.

## Effect of each change, per target

### A view replaced by another (same number of images, same cost)

| Change | vs | Macro | ACL | MCL | MM | LM | MOA | LOA | PFOA | Eff | Syn | Baker | Cont | Frac |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Coronal + sagittal T1/PD: 120 → 84 mm | `control` | +0.0013 | +0.0020 | -0.0008 | +0.0020 | -0.0020 | +0.0004 | -0.0017 | -0.0017 | +0.0018 | **+0.0036** | +0.0022 | +0.0016 | **+0.0078** |
| Coronal + sagittal fluid: 120 → 93 mm | `t1pd-detail-84` | +0.0010 | +0.0015 | -0.0015 | -0.0002 | +0.0030 | +0.0019 | **+0.0036** | +0.0029 | -0.0002 | **+0.0054** | **-0.0062** | -0.0006 | +0.0019 |
| Coronal fluid only: 120 → 93 mm | `t1pd-detail-84` | +0.0003 | +0.0018 | +0.0010 | -0.0010 | +0.0005 | +0.0007 | +0.0022 | +0.0004 | -0.0001 | +0.0018 | -0.0016 | -0.0014 | -0.0012 |
| Sagittal fluid only: 120 → 93 mm | `t1pd-detail-84` | +0.0005 | -0.0016 | -0.0004 | -0.0000 | +0.0020 | **+0.0037** | **+0.0034** | **+0.0036** | +0.0007 | **+0.0060** | **-0.0090** | +0.0003 | -0.0023 |
| Sagittal fluid: 120 → 93 mm, coronal fluid already at 93 | `coronal-fluid-93-sagittal-fluid-120` | +0.0007 | -0.0003 | -0.0025 | +0.0008 | +0.0024 | +0.0011 | +0.0015 | +0.0024 | -0.0001 | **+0.0036** | **-0.0047** | +0.0008 | **+0.0032** |
| Coronal fluid: 120 → 93 mm, sagittal fluid already at 93 | `sagittal-fluid-93-coronal-fluid-120` | +0.0004 | **+0.0031** | -0.0011 | -0.0002 | +0.0010 | -0.0019 | +0.0002 | -0.0007 | -0.0009 | -0.0006 | +0.0028 | -0.0009 | **+0.0042** |
| Sagittal fluid: 93 mm centred, no forward shift | incumbent | -0.0002 | +0.0001 | **+0.0048** | +0.0000 | **-0.0034** | +0.0016 | -0.0019 | -0.0008 | -0.0012 | +0.0007 | -0.0003 | -0.0001 | -0.0013 |
| Sagittal fluid: 93 → 100 mm | incumbent | -0.0002 | -0.0007 | +0.0016 | +0.0001 | -0.0020 | +0.0005 | +0.0019 | -0.0024 | +0.0014 | -0.0011 | -0.0020 | -0.0006 | +0.0014 |
| Sagittal T1/PD: 84 → 120 mm, coronal T1/PD kept at 84 | incumbent | -0.0012 | -0.0027 | +0.0021 | -0.0007 | **-0.0061** | -0.0009 | +0.0017 | -0.0007 | -0.0016 | -0.0013 | -0.0020 | -0.0009 | -0.0012 |
| Coronal T1/PD: 84 → 120 mm, sagittal T1/PD kept at 84 | incumbent | not run | | | | | | | | | | | | |
| Coronal + sagittal T1/PD: 84 → 93 mm | incumbent | not run | | | | | | | | | | | | |
| Axial: 120 → 130 mm | incumbent | -0.0000 | -0.0005 | **+0.0041** | +0.0002 | -0.0005 | -0.0002 | -0.0011 | -0.0006 | -0.0024 | +0.0015 | -0.0018 | +0.0009 | +0.0001 |
| Axial: 120 → 140 mm | incumbent | -0.0006 | -0.0002 | -0.0021 | +0.0007 | -0.0019 | -0.0005 | -0.0003 | -0.0012 | -0.0018 | **+0.0035** | -0.0005 | -0.0017 | -0.0014 |
| Coronal + sagittal fluid: 120 mm fisheye (fine 80 mm core) | `t1pd-detail-84` | +0.0000 | +0.0004 | -0.0028 | -0.0004 | +0.0022 | +0.0030 | **+0.0031** | **+0.0044** | +0.0010 | **+0.0042** | **-0.0088** | -0.0001 | **-0.0060** |
| Coronal + sagittal T1/PD: 120 mm fisheye in place of 84 mm | `t1pd-detail-84` | -0.0003 | +0.0008 | +0.0019 | -0.0004 | -0.0010 | +0.0006 | +0.0003 | +0.0009 | -0.0019 | +0.0029 | -0.0004 | -0.0002 | **-0.0072** |
| Coronal + sagittal T1/PD: 84 → 84–120 mm per series (stored pixel) | `t1pd-detail-84` | -0.0002 | -0.0000 | -0.0009 | -0.0017 | +0.0010 | +0.0008 | -0.0006 | **+0.0032** | -0.0015 | +0.0007 | -0.0029 | +0.0014 | -0.0016 |
| Coronal + sagittal T1/PD: 84 → 84–120 mm per series (acquired matrix) | `t1pd-detail-84` | -0.0005 | -0.0012 | **+0.0037** | **-0.0034** | +0.0009 | +0.0010 | -0.0005 | +0.0024 | -0.0013 | +0.0024 | **-0.0037** | -0.0003 | **-0.0066** |
| T1/PD 84 mm with a 14x14 token grid | `t1pd-detail-84` | -0.0002 | -0.0018 | +0.0001 | -0.0007 | +0.0010 | +0.0010 | +0.0010 | +0.0016 | -0.0013 | **+0.0031** | -0.0018 | -0.0016 | -0.0026 |

"incumbent" is `fluid-detail-93-replacement` (S93): T1/PD 84 mm, fluid 93 mm, axial 120 mm.

### A view or crop added (more images, more time)

| Change | vs | Macro | ACL | MCL | MM | LM | MOA | LOA | PFOA | Eff | Syn | Baker | Cont | Frac |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1/PD: 120 mm kept, 84 mm fused in early | `t1pd-detail-84` | -0.0018 | -0.0017 | +0.0028 | **-0.0039** | -0.0019 | +0.0001 | +0.0004 | +0.0014 | +0.0022 | -0.0029 | **-0.0046** | **-0.0051** | **-0.0088** |
| 60 mm T1/PD crop, rule-placed | `control` | +0.0002 | +0.0012 | **+0.0033** | -0.0019 | -0.0023 | -0.0011 | **+0.0043** | -0.0001 | +0.0023 | **+0.0045** | **-0.0058** | -0.0025 | +0.0010 |
| 60 mm T1/PD crop, attention-placed | `control` | -0.0003 | +0.0007 | +0.0012 | -0.0018 | -0.0022 | -0.0009 | **+0.0048** | +0.0003 | +0.0017 | **+0.0045** | **-0.0095** | -0.0016 | -0.0014 |
| Two compartment crops per coronal T1/PD slice | `control` | +0.0002 | -0.0000 | **-0.0041** | -0.0010 | +0.0003 | -0.0007 | **+0.0054** | +0.0011 | +0.0015 | +0.0030 | -0.0010 | **-0.0035** | +0.0011 |
| 60 mm T1/PD crop with its own loss | `control` | -0.0003 | **+0.0032** | **-0.0052** | -0.0009 | -0.0030 | +0.0023 | **+0.0034** | -0.0015 | -0.0016 | **+0.0043** | **-0.0049** | -0.0019 | +0.0027 |
| 60 mm T1/PD crop, evidence-placed | `control` | -0.0003 | -0.0008 | +0.0027 | -0.0022 | **-0.0032** | -0.0010 | **+0.0032** | -0.0009 | +0.0019 | **+0.0045** | **-0.0080** | -0.0022 | +0.0020 |
| 60 mm T1/PD crop, ranking locator | `control` | +0.0001 | **-0.0039** | +0.0024 | -0.0017 | -0.0022 | **-0.0030** | **+0.0033** | +0.0021 | +0.0029 | **+0.0073** | **-0.0086** | -0.0014 | **+0.0036** |
| Sagittal fluid detail window (93x67 mm) beside 120 mm | `t1pd-detail-84` | -0.0002 | -0.0008 | **-0.0049** | -0.0017 | +0.0018 | **+0.0032** | **+0.0055** | +0.0028 | **-0.0035** | **+0.0030** | **-0.0072** | -0.0001 | -0.0008 |
| The same, fused into the main tokens | `t1pd-detail-84` | +0.0000 | +0.0004 | -0.0027 | -0.0006 | +0.0014 | +0.0018 | **+0.0031** | **+0.0035** | -0.0026 | **+0.0055** | **-0.0072** | -0.0013 | -0.0013 |
| The same, fused as a residual | `t1pd-detail-84` | +0.0001 | +0.0003 | **-0.0062** | +0.0010 | +0.0018 | +0.0028 | +0.0010 | **+0.0034** | -0.0020 | **+0.0060** | **-0.0054** | -0.0003 | -0.0013 |
| 120 mm view of sagittal fluid slices read only by Baker's cyst, in place of the zoomed one | incumbent | -0.0003 | +0.0004 | -0.0015 | -0.0015 | -0.0016 | -0.0001 | -0.0013 | -0.0005 | +0.0005 | +0.0024 | +0.0023 | -0.0009 | -0.0019 |
| 140 mm view of fluid slices, all planes, second backbone | `control` | +0.0001 | +0.0000 | **-0.0049** | -0.0018 | **-0.0053** | +0.0028 | -0.0002 | -0.0004 | +0.0024 | **+0.0056** | **+0.0054** | -0.0000 | -0.0024 |
| The same with every main view at 112 mm | `control` | -0.0003 | -0.0001 | **-0.0153** | -0.0010 | -0.0010 | **+0.0034** | +0.0021 | **+0.0042** | +0.0003 | **+0.0059** | +0.0027 | -0.0006 | **-0.0043** |

### Absolute scores of the four reference recipes

| Run | ACL | MCL | MM | LM | MOA | LOA | PFOA | Eff | Syn | Baker | Cont | Frac | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `control` (all 120 mm) | 0.9525 | 0.9245 | 0.9536 | 0.9331 | 0.9388 | 0.9064 | 0.8844 | 0.9304 | 0.8927 | 0.9454 | 0.9592 | 0.9469 | 0.9307 |
| `t1pd-detail-84` | 0.9545 | 0.9237 | 0.9556 | 0.9311 | 0.9392 | 0.9047 | 0.8827 | 0.9322 | 0.8963 | 0.9476 | 0.9608 | 0.9547 | 0.9319 |
| `coronal-fluid-93-sagittal-fluid-120` | 0.9563 | 0.9247 | 0.9546 | 0.9316 | 0.9399 | 0.9068 | 0.8831 | 0.9322 | 0.8981 | 0.9460 | 0.9593 | 0.9535 | 0.9322 |
| `sagittal-fluid-93-coronal-fluid-120` | 0.9529 | 0.9233 | 0.9556 | 0.9330 | 0.9430 | 0.9081 | 0.8863 | 0.9330 | 0.9023 | 0.9386 | 0.9611 | 0.9524 | 0.9325 |
| `fluid-detail-93-replacement` (incumbent, S93) | 0.9560 | 0.9222 | 0.9554 | 0.9340 | 0.9411 | 0.9083 | 0.8856 | 0.9320 | 0.9017 | 0.9414 | 0.9602 | 0.9567 | 0.9329 |

Taking each label's best of these five rows would give a macro of 0.9341. That is an upper
bound from one run each, not a forecast.

The two fluid planes add up: zooming coronal alone is worth +0.0003, sagittal alone +0.0005,
both +0.0010 (against `t1pd-detail-84`).

## What each kind of token reads on its own

x1, last epoch. "Alone" pools only that kind of token of the trained model; "attention" is the
share of the label's attention that kind receives. A kind that scores low alone and still takes
attention is diluting the label.

### Incumbent: T1/PD 84 mm, fluid 93 mm, axial 120 mm

| Target | joint | axial alone | T1/PD 84 alone | coronal/sagittal fluid 93 alone | attention on T1/PD 84 | attention on fluid 93 |
|---|---:|---:|---:|---:|---:|---:|
| ACL | 0.955 | 0.772 | 0.932 | 0.943 | 0.372 | 0.497 |
| MCL | 0.910 | 0.879 | 0.780 | 0.865 | 0.229 | 0.452 |
| Medial Meniscus | 0.952 | 0.701 | 0.919 | 0.945 | 0.311 | 0.626 |
| Lateral Meniscus | 0.929 | 0.720 | 0.881 | 0.916 | 0.354 | 0.570 |
| Medial OA | 0.937 | 0.831 | 0.903 | 0.933 | 0.310 | 0.482 |
| Lateral OA | 0.908 | 0.798 | 0.859 | 0.885 | 0.354 | 0.418 |
| PF OA | 0.876 | 0.832 | 0.781 | 0.846 | 0.293 | 0.359 |
| Effusion | 0.928 | 0.908 | 0.780 | 0.913 | 0.206 | 0.312 |
| Synovitis | 0.905 | 0.869 | 0.825 | 0.853 | 0.262 | 0.311 |
| Baker's | 0.941 | 0.936 | 0.546 | 0.878 | 0.178 | 0.348 |
| Contusion | 0.959 | 0.904 | 0.800 | 0.957 | 0.277 | 0.481 |
| Fracture | 0.952 | 0.908 | 0.780 | 0.923 | 0.294 | 0.470 |

### `coronal-fluid-93-sagittal-fluid-120`: sagittal fluid back at 120 mm

| Target | joint | axial + sagittal fluid 120 alone | T1/PD 84 alone | coronal fluid 93 alone | attention on T1/PD 84 | attention on coronal fluid 93 |
|---|---:|---:|---:|---:|---:|---:|
| ACL | 0.954 | 0.906 | 0.930 | 0.892 | 0.385 | 0.217 |
| MCL | 0.911 | 0.877 | 0.784 | 0.855 | 0.253 | 0.277 |
| Medial Meniscus | 0.951 | 0.917 | 0.918 | 0.916 | 0.333 | 0.347 |
| Lateral Meniscus | 0.926 | 0.896 | 0.878 | 0.881 | 0.377 | 0.318 |
| Medial OA | 0.935 | 0.918 | 0.906 | 0.906 | 0.328 | 0.263 |
| Lateral OA | 0.905 | 0.876 | 0.857 | 0.852 | 0.368 | 0.172 |
| PF OA | 0.870 | 0.860 | 0.777 | 0.735 | 0.288 | 0.117 |
| Effusion | 0.926 | 0.916 | 0.787 | 0.866 | 0.213 | 0.122 |
| Synovitis | 0.900 | 0.882 | 0.814 | 0.838 | 0.262 | 0.127 |
| Baker's | 0.951 | 0.949 | 0.567 | 0.777 | 0.195 | 0.179 |
| Contusion | 0.955 | 0.945 | 0.803 | 0.913 | 0.286 | 0.251 |
| Fracture | 0.949 | 0.938 | 0.776 | 0.877 | 0.295 | 0.251 |

### `sagittal-fluid-93-coronal-fluid-120`: coronal fluid back at 120 mm

| Target | joint | axial + coronal fluid 120 alone | T1/PD 84 alone | sagittal fluid 93 alone | attention on T1/PD 84 | attention on sagittal fluid 93 |
|---|---:|---:|---:|---:|---:|---:|
| ACL | 0.950 | 0.874 | 0.924 | 0.910 | 0.403 | 0.332 |
| MCL | 0.911 | 0.910 | 0.745 | 0.732 | 0.226 | 0.184 |
| Medial Meniscus | 0.951 | 0.913 | 0.919 | 0.913 | 0.310 | 0.300 |
| Lateral Meniscus | 0.928 | 0.870 | 0.878 | 0.896 | 0.355 | 0.266 |
| Medial OA | 0.939 | 0.921 | 0.903 | 0.906 | 0.306 | 0.219 |
| Lateral OA | 0.913 | 0.884 | 0.862 | 0.854 | 0.355 | 0.251 |
| PF OA | 0.879 | 0.841 | 0.790 | 0.832 | 0.293 | 0.255 |
| Effusion | 0.930 | 0.912 | 0.786 | 0.897 | 0.218 | 0.194 |
| Synovitis | 0.904 | 0.879 | 0.833 | 0.817 | 0.268 | 0.167 |
| Baker's | 0.941 | 0.937 | 0.539 | 0.861 | 0.170 | 0.187 |
| Contusion | 0.960 | 0.929 | 0.804 | 0.918 | 0.276 | 0.232 |
| Fracture | 0.944 | 0.933 | 0.756 | 0.899 | 0.291 | 0.232 |

### `t1pd-detail-84`: both fluid planes at 120 mm

| Target | joint | axial + coronal/sagittal fluid 120 alone | T1/PD 84 alone | attention on T1/PD 84 |
|---|---:|---:|---:|---:|
| ACL | 0.950 | 0.932 | 0.921 | 0.421 |
| MCL | 0.912 | 0.909 | 0.765 | 0.246 |
| Medial Meniscus | 0.952 | 0.944 | 0.923 | 0.335 |
| Lateral Meniscus | 0.922 | 0.907 | 0.880 | 0.383 |
| Medial OA | 0.937 | 0.930 | 0.910 | 0.332 |
| Lateral OA | 0.903 | 0.895 | 0.853 | 0.377 |
| PF OA | 0.870 | 0.863 | 0.770 | 0.291 |
| Effusion | 0.930 | 0.920 | 0.785 | 0.229 |
| Synovitis | 0.898 | 0.880 | 0.816 | 0.265 |
| Baker's | 0.952 | 0.952 | 0.533 | 0.181 |
| Contusion | 0.957 | 0.956 | 0.802 | 0.282 |
| Fracture | 0.949 | 0.947 | 0.756 | 0.302 |

What these tables add:

- **The zoomed T1/PD tokens are specialists.** They read ACL, the menisci and medial OA well
  (0.90 to 0.93 alone) and the fluid and periphery labels poorly: Baker's cyst 0.55, MCL 0.78,
  PF OA 0.78, effusion 0.78, fracture 0.78, contusion 0.80. They still take 18 to 30% of those
  labels' attention.
- **Axial reads the periphery and little else.** Alone: Baker's cyst 0.936, effusion 0.908,
  fracture 0.908, contusion 0.904; ACL 0.772, menisci 0.70 to 0.72.
- **Zoomed fluid is the strongest single kind for nine labels**, and the weak one for Baker's
  cyst (0.878 against 0.936 for axial alone).
- **Coronal fluid alone is weak for PF OA (0.735) and Baker's cyst (0.777)**; those two live on
  sagittal fluid and axial.

## Slice budget: what each series type can supply

All 4,407 training studies, headers only (`scripts/adhoc/measure_slice_capacity.py`, outputs in
`outputs/analysis/s/slice-budget-20261002/`). "Positions" are distinct slice positions in the
sampler's pool for that type. The budget in the config is 6 slices per axial type and 5 per
coronal or sagittal type, with a study total of at least 26.

| Series type | Studies with it | Positions available, median (p10–p90) | Slice gap, median | Slices given at x1 | Share of positions used at x1 | Distance between picked slices at x1 | Studies capped at x4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Axial T1/PD | 19% | 32 (26–90) | 3.3 mm | 6.3 | 19% | 18 mm | 13% |
| Axial fluid | 100% | 30 (20–44) | 3.9 mm | 6.5 | 22% | 18 mm | 26% |
| Coronal T1/PD | 77% | 28 (16–38) | 3.6 mm | 5.1 | 20% | 19 mm | 20% |
| Coronal fluid | 96% | 28 (17–36) | 3.6 mm | 5.3 | 20% | 18 mm | 17% |
| Sagittal T1/PD | 97% | 28 (20–36) | 3.5 mm | 5.3 | 19% | 18 mm | 12% |
| Sagittal fluid | 94% | 28 (19–36) | 3.5 mm | 5.2 | 20% | 18 mm | 15% |

- **Supply never limits the budget at x1.** Every type has four to five times more slice
  positions than it is given, and no study is capped at x1 or x2. Slices can be moved freely
  between types.
- **At x1 the model sees one slice in five, about 18 mm apart, in every type.** At x4 it sees
  three in four, and 12 to 26% of studies run out of positions; x4 is close to the whole stack.
- **Studies differ in what they contain.** 13% have all six types and get 32 slices, 59% have
  five and get 26, 29% have four and are topped up to 26. Axial T1/PD is missing in 81% of
  studies and coronal T1/PD in 23%.
- **So the budget is a question of need alone:** which type's extra slice is worth most to
  which label. That has not been measured yet.

## Notes per run

Details that the tables do not show: what each run's own diagnostics said, and what it taught.
Newest first within each group. x4 unless marked x1.

### Views replaced

- **`coronal-t1pd-84-only`** (0.931699, -0.0012 against the incumbent, 80% [-0.0020, -0.0003];
  the last run, decided while the campaign stopped, so it has no reporter write-up). With
  sagittal T1/PD back at the image-centred 120 mm view the lateral meniscus fell 0.0061 (80%
  [-0.0092, -0.0030]) and ACL 0.0027 (80% [-0.0054, -0.0000]); Baker's cyst -0.0020, effusion
  -0.0016; MCL +0.0021, lateral OA +0.0017. The remaining zoomed T1/PD tokens (coronal only)
  read the lateral meniscus at 0.778 alone at x1, against 0.881 when both planes are zoomed.
  Lesson: most of the T1/PD gain sits in the sagittal plane, through the lateral meniscus and
  ACL; this is larger than the regression's tied estimate (+0.0005 for both planes together).
  The mirror run, coronal T1/PD back at 120 mm, was not run.
- **`sagittal-fluid-100`** (0.932719, -0.0002 against the incumbent, 80% [-0.0007, +0.0005]).
  Baker's cyst did not recover at the middle width: 0.9393, -0.0020 against the incumbent (80%
  [-0.0038, -0.0002]), against 0.9460 with sagittal fluid at 120 mm. The zoomed fluid tokens
  alone read it a little better (0.894 against 0.878 at x1), as in the centred run, without
  moving the joint score. PF OA -0.0024 and the lateral meniscus -0.0020, the labels that the
  finer sagittal pixel had helped; lateral OA +0.0019, MCL +0.0016, effusion +0.0014. Lesson:
  there is no compromise width for sagittal fluid between 93 and 120 mm; Baker's cyst needs the
  full wide view and the detail labels the fine one.
- **`sagittal-fluid-93-centred`** (0.932729, -0.0002 against the incumbent, 80% [-0.0009,
  +0.0006]). Baker's cyst -0.0003: no recovery. The zoomed fluid tokens alone read Baker's cyst
  better (0.906 against 0.878 at x1), so the 7 mm at the back does hold cyst evidence, but the
  joint score barely moved (0.942 against 0.941): 93 mm is too narrow whatever its position.
  Lateral OA -0.0019 (80% [-0.0042, +0.0002]) and PF OA -0.0008 fell back to the control's
  level (0.9064, 0.8847); medial OA +0.0016 (80% [+0.0001, +0.0032]). MCL +0.0048, lateral
  meniscus -0.0034. Lesson: the forward shift earns its place for the labels read at the front
  of the sagittal slice; keep it.
- **`sagittal-fluid-93-coronal-fluid-120`** (0.932461, -0.0004 against the incumbent). Sagittal
  zoom alone: +0.0005 over `t1pd-detail-84`, with synovitis +0.0060, medial OA +0.0037, PF OA
  +0.0036, lateral OA +0.0034, lateral meniscus +0.0020, and Baker's cyst -0.0090 (0.9386).
  Sagittal fluid 93 mm tokens alone at x1: Baker's cyst 0.861, MCL 0.732, effusion 0.897.
- **`coronal-fluid-93-sagittal-fluid-120`** (0.932186, -0.0007 against the incumbent). With
  sagittal fluid back at 120 mm Baker's cyst returns to 0.9460 (x1 0.9515, wide tokens alone
  0.949). Coronal fluid 93 mm tokens alone at x1: PF OA 0.735, Baker's cyst 0.777, so coronal
  fluid serves neither. Lesson: the conflict between the detail gain and Baker's cyst sits
  inside sagittal fluid.
- **`axial-140-detail-84-93`** (0.932259, -0.0006) and **`axial-130-detail-84-93`** (0.932848,
  -0.0000). Axial tokens alone read Baker's cyst 0.935 at both, as at 120 mm (0.936), so the
  field was never the limit on axial. Synovitis rose with width at x4 (+0.0015, +0.0035) but
  the axial tokens alone did not read it better (0.874, 0.866 against 0.869), and at x1 the
  joint gain was not ordered; treat as weak. 140 mm cost MCL -0.0021 and the lateral meniscus
  -0.0019.
- **`fluid-foveated-native-view`** (0.931927, -0.0010 against the incumbent, +0.0000 against
  `t1pd-detail-84`). The fine 80 mm core gave the OA labels what the 93 mm zoom gave (PF OA
  +0.0044, lateral OA +0.0031, medial OA +0.0030 against `t1pd-detail-84`). Baker's cyst fell
  0.0088 although the whole 120 mm field was in the image; the squeezed outer band took 2 to 3%
  of its attention. Fracture -0.0060. Lesson: a squeezed periphery is not a usable periphery,
  and the detail gain comes from pixel size in the centre.
- **`fluid-detail-93-replacement`** (0.932880, the incumbent, adopted by the user; S93, 0.950
  public against 0.947 for the control). At x1 the zoomed T1/PD tokens read Baker's cyst at
  0.55, MCL, PF OA, effusion and fracture at 0.78, and still take 18 to 29% of those labels'
  attention.
- **`t1pd-detail-dense-final-grid`** (-0.0002 against `t1pd-detail-84`): four times the tokens
  on the zoomed T1/PD views changed nothing.
- **`t1pd-foveated-native-view`** (-0.0003 against `t1pd-detail-84`): kept the 120 mm field and
  matched the 84 mm view, with fracture -0.0072. Its detail tokens alone scored 0.818 at x1
  against 0.804 to 0.813 for the uniform view.
- **`t1pd-detail-acquisition-matched`** (-0.0002) and **`t1pd-detail-acquired-matrix`**
  (-0.0005): widening the window on series acquired more coarsely did not help; the medial
  meniscus and fracture fell.
- **`t1pd-detail-84`** (0.931918, +0.0013 against the control): the first accepted step.

### Views or crops added

- **`sagittal-fluid-wide-for-bakers`** (0.932573, -0.0003 against the incumbent, 80% [-0.0010,
  +0.0004]; 1,424 s, 26% more than the incumbent). The first view routed to one label. The
  routing did what it was told: Baker's cyst put 20% of its attention on the wide sagittal
  fluid tokens, none on the zoomed sagittal fluid ones, and no other label touched the wide
  ones. Baker's cyst recovered about half of what the zoom had cost: x4 0.9437, +0.0023 against
  the incumbent (80% [-0.0014, +0.0067]), where sagittal fluid at 120 mm for every label gives
  0.9460; x1 0.9472 against 0.9408 and 0.9515. Alone at x1 the wide sagittal fluid tokens read
  it at 0.905, axial at 0.936, both together 0.947. The other labels were meant to hold and
  drifted down by noise-sized amounts (fracture -0.0019, lateral meniscus -0.0016, MCL and
  medial meniscus -0.0015, lateral OA -0.0013; synovitis +0.0024), which cancels the Baker's
  gain in the macro. Lesson: routing by label works mechanically, and its payoff here is about
  +0.0002 to +0.0005 macro, below what one run resolves (training noise 0.0008), at a quarter
  more time. Whether the small drops elsewhere are noise or a side effect of training the
  shared encoder on views that only one label uses is not known.
- **`t1pd-shallow-detail-fusion`** (-0.0018 against `t1pd-detail-84`, 80% [-0.0033, -0.0005]):
  keeping the 120 mm T1/PD view and merging the 84 mm one into it early lost what the
  replacement had gained (fracture -0.0088, contusion -0.0051, Baker's cyst -0.0046).
- **`sagittal-fluid-detail-rectangle`, `-correspondence-fusion`, `-content-residual`**
  (-0.0002, +0.0000, +0.0001 against `t1pd-detail-84`, at 35 to 37% more time): a sagittal fluid
  detail window beside the wide view raised the OA labels and synovitis and cost MCL and Baker's
  cyst, whichever way the tokens were merged.
- **`two-views-fluid-140`** (+0.0001 against the control, 1,630 s): the second stream was alive
  (11% of attention, 0.821 macro alone at x1). Its 140 mm fluid tokens alone read Baker's cyst
  at 0.942 and the run has the best Baker's cyst of the campaign (0.9508). It cost MCL -0.0049
  and the lateral meniscus -0.0053.
- **`two-views-112-fluid-140`** (-0.0003): narrowing every main view to 112 mm cost MCL 0.0153.
- **`crop-full-context-ranking-locator`** (+0.0001): the two candidate crops differed in loss by
  0.003, the scoring head agreed with the better one 49.9% of the time and its gradient was near
  zero, although it moved the crop 26 mm from the rule. Lesson: where a 60 mm crop sits barely
  changes the loss, so no placement method can be rewarded.
- **`crop-compartments`** (+0.0002): lateral OA +0.0054, medial OA -0.0007. At x1 lateral OA was
  0.9055 from the wide tokens alone (control 0.9020) and 0.908 jointly, with the crop tokens at
  0.849 alone and 15% of the attention: about half of the gain is a training effect on the
  shared encoder. For medial OA the crop tokens alone scored 0.897 against 0.935 for the wide
  ones.
- **`crop-own-loss`** (-0.0003): the only crop run in which both medial OA (+0.0023) and
  lateral OA (+0.0034) rose.
- **`crop-rule`, `crop-attention`, `crop-evidence-utility-placement`,
  `crop-rule-overlap-replacement`, `crop-context-teacher-distillation`**: all within 0.0007 of
  the control. Lateral OA rose in every one with a crop at prediction; Baker's cyst fell 0.005
  to 0.0095 although nothing was cut, with the wide tokens themselves reading it worse
  (0.941 against 0.950 for the control at x1).

### Measurements

- **Regression of width per type against AUC per label** (17 single-width runs, 2026-10-02;
  `outputs/analysis/v/view-regression-20261002/report.md`, refit with
  `uv run --locked python -m scripts.adhoc.fit_view_width_regression`).
  - The incumbent's widths are the best combination among tested widths, but every single-type
    change from it is predicted only 0.0001 to 0.0005 lower, none distinguishable from zero.
  - Training noise of one run is 0.0006 on the macro (SD, from five replicate runs of the older
    recipe) and 0.0008 to 0.0028 on a label; a new run against the measured incumbent carries
    0.0008 on the macro. Macro steps under about 0.001 cannot be told apart by one run.
  - The three zooms together are worth +0.0013 by the model (80% [+0.0000, +0.0026]) against
    +0.0022 measured. The T1/PD zoom is +0.0005 by the model against +0.0013 from its one pair:
    the other runs imply the 120 mm control came out about 0.0008 low.
  - The one macro effect clear of both kinds of noise: T1/PD wider than 120 mm costs (120 to
    140 mm: -0.0012, 95% [-0.0023, -0.0001]; lateral meniscus -0.0048, ACL -0.0039).
  - The one conflict with both sides clear of noise is sagittal fluid: zooming it gives
    synovitis +0.0034 and costs Baker's cyst -0.0040 (-0.0058 at x1).
  - Routing views by label is worth at most +0.0011 within the tested widths, half of it
    Baker's cyst; an upper bound, biased upward.
  - Not identifiable yet: coronal against sagittal T1/PD; T1/PD between 84 and 120 mm;
    anything finer than 84 mm (T1/PD), 93 mm (fluid) or 120 mm (axial); width, pixel size and
    placement apart.
- **Control's luck.** Against the 120 mm control synovitis rose in all 20 runs (mean +0.0054)
  and the lateral meniscus fell in 18 (mean -0.0019); these are also the two labels where the
  control had diverged most from its 140 mm sibling.
- **Baker's cyst noise.** Runs whose Baker's-relevant inputs are identical span 0.9395 to 0.9517
  at x1, so a single-run change under about 0.005 on this label is not evidence.
- **Scale embedding.** The trained zoomed-view embedding has norm 0.10 in the incumbent and 0.25
  in `t1pd-detail-84`, against 0.35 to 0.43 for the plane and contrast embeddings.
- **Positions.** Zoomed tokens carry their true position in the 120 mm frame of their series,
  through the same resampling maps that cut the window.
- **Spacing review** (`outputs/analysis/v/view-spacing-20261002/report.md`): 84 mm and 93 mm keep
  the acquired detail of about nine in ten coronal and sagittal series; the next coarser
  candidate loses 10 to 17 points.

## Still to come

The campaign closes after these five runs (user, 2026-10-02 12:15 CEST); a watcher stops it when
the last one is decided.

| Run | Cell it fills |
|---|---|
| `sagittal-fluid-100` | sagittal fluid at 100 mm |
| `sagittal-fluid-wide-for-bakers` | a 120 mm sagittal fluid view that only Baker's cyst reads; the first test of routing a view to one label |
| `coronal-t1pd-84-only` | T1/PD zoom on the coronal plane only (requeue of `coronal-t1pd-84-sagittal-t1pd-120`, blocked at review by the dependency gate, not by a result) |
| `sagittal-t1pd-84-only` | T1/PD zoom on the sagittal plane only (requeue of `sagittal-t1pd-84-coronal-t1pd-120`, which meets the same gate) |
| `t1pd-detail-93-both` | T1/PD at 93 mm in place of 84 (requeue of `t1pd-detail-93`) |

Removed from this campaign, to be rerun in the next one as configuration: compartment crops
routed to the OA labels (`oa-compartment-readers`), the sagittal windows without the 10 mm
downward shift (`sagittal-no-vertical-shift`), and, once the view table is chosen, separate
normalisation statistics per pixel size (`spacing-specific-batch-statistics`).

## The next campaign

Decided with the user on 2026-10-02: treat this as an optimisation problem over a flexible
per-type view configuration, in a new campaign of configuration-only runs. The plan is in
[view table backlog](../../../docs/backlog/big-bets.md#view-table-and-label-routing); the
[learnable-views campaign](../learnable-views/INDEX.md) ran first and learned the views by
gradient instead.

Caution recorded on 2026-10-02: the "best of five recipes" macro of 0.9341 only chooses among
recipes already tried. Axial has never been tried finer than 120 mm, coronal and sagittal fluid
never finer than 93 mm, T1/PD never finer than 84 mm, so the true ceiling is unknown.
