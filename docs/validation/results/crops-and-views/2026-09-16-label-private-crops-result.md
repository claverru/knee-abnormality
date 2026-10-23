# 2026-09-16 label-private crops result

**A clear regression.** Giving each label its own two crops, pooled only by that label, reached
**0.845131** fold-0 macro AUC at epoch 9, −0.0466 against [per-label crops](2026-09-16-per-label-crops-result.md)
(0.891703) and −0.0476 against `00-baseline` (0.892763). Every target fell. The run did one thing
the earlier crop runs never did: crop placement now differs between studies with and without a
finding.

## Setup

- **Change from per-label:** each label takes exactly 2 crops (24 ReXNet-200 crops per study, no
  deduplication), the second on another plane when one is available, otherwise the best cell at
  least 3 cells from the label's first crop. Each label pools only its own 2 crops' fine tokens,
  mixed with the shared coarse pool by a learned per-label gate `g·coarse + (1−g)·fine` (starting at
  0.5). A zero-initialised per-label FiLM scales and shifts ReXNet-200's `features.10` activations.
  Everything else matches per-label: 300-pixel canvases at 0.5 mm, 224-pixel crops, batch 8,
  10 epochs, seed 42, fold 0 of `all_5fold_v3`.
- **Run:** `outputs/training/label-private-crops`, commit `ea19772` plus the working tree; the best epoch was
  the last.
- **Cost:** about 234 s per epoch including validation (per-label and crop224: about 200 s), 45 s
  for the final validation prediction, 4,380 optimizer steps, peak 22.4 GiB CUDA.
- **Probes and placement:** `outputs/analysis/l/label-private-probes-20260916` (inference-only
  variants, BF16) and `outputs/analysis/c/crop-diversity-label-private-20260916`.

## Training curve

| Epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| label-private | 0.7010 | 0.7512 | 0.7700 | 0.7903 | 0.8119 | 0.8255 | 0.8354 | 0.8403 | 0.8428 | **0.8451** |
| draft (coarse pool) | 0.6908 | 0.7187 | 0.7426 | 0.7623 | 0.7808 | 0.7984 | 0.8064 | 0.8168 | 0.8195 | 0.8217 |
| per-label | 0.7778 | 0.8284 | 0.8493 | 0.8633 | 0.8736 | 0.8819 | 0.8887 | 0.8898 | **0.8917** | 0.8915 |
| train loss, label-private | 0.595 | 0.571 | 0.558 | 0.548 | 0.542 | 0.533 | 0.525 | 0.517 | 0.511 | 0.508 |
| train loss, per-label | 0.582 | 0.543 | 0.519 | 0.505 | 0.491 | 0.480 | 0.470 | 0.460 | 0.454 | 0.449 |

Training loss stayed well above per-label's, so the model underfits rather than overfits; it was
still improving at epoch 9. Gold-label macro AUC is 0.9079, report-derived 0.8434, soft-target BCE
0.48823 (per-label 0.44834). The draft, now the independently trained coarse pool, reached 0.822
(per-label's draft: 0.626).

## Per-target AUC

| Target | `00-baseline` | per-label | label-private | vs per-label |
|---|---:|---:|---:|---:|
| Lateral Meniscus | 0.8939 | 0.8837 | 0.7429 | **−0.1408** |
| Medial Meniscus | 0.9239 | 0.9248 | 0.8442 | −0.0806 |
| ACL | 0.9146 | 0.9257 | 0.8478 | −0.0779 |
| MCL | 0.8424 | 0.8359 | 0.7733 | −0.0626 |
| Contusion | 0.8899 | 0.8873 | 0.8480 | −0.0393 |
| PF OA | 0.8715 | 0.8687 | 0.8343 | −0.0344 |
| Baker's | 0.9150 | 0.9111 | 0.8831 | −0.0280 |
| Medial OA | 0.9045 | 0.9102 | 0.8835 | −0.0267 |
| Synovitis | 0.8911 | 0.8988 | 0.8765 | −0.0223 |
| Lateral OA | 0.8662 | 0.8660 | 0.8452 | −0.0208 |
| Fracture | 0.9301 | 0.9216 | 0.9026 | −0.0190 |
| Effusion | 0.8702 | 0.8667 | 0.8601 | −0.0065 |

The targets that need fine detail (menisci, ligaments) collapsed; diffuse targets the scout can see
at 0.5 mm (effusion, fracture's oedema, the OA grades) lost least.

## Learned mixing

- **Gates barely moved:** 0.505–0.537 across labels, so every label kept about half coarse and half
  fine evidence.
- **FiLM stayed small:** mean |scale| about 0.02; contusion moved most (maximum |scale| 0.118, mean
  |shift| 0.032). Label vectors are unrelated to each other (mean cosine −0.04).

## Inference probes

The best checkpoint, changing only how validation studies are pooled:

| Variant | Macro AUC |
|---|---:|
| Private, 2 crops per label (as trained) | 0.845131 |
| Private, 4 crops per label | 0.847108 |
| Private, 8 crops per label | 0.848778 |
| Shared pooling of all 24 crops | 0.848914 |
| Shared pooling, 8 crops per label | 0.849023 |
| Fine only (gate forced to 0) | 0.807178 |
| Coarse only (gate forced to 1) | 0.821698 |

More crops or shared pooling at inference recover at most 0.004. The deficit is in the fine
features the training produced, not in what inference lets each label see: each ReXNet-200 crop was
trained through one label's loss instead of twelve, and fine tokens alone score below the scout.

## Selection-signal probe

The same checkpoint with shared pooling, changing only the per-label value that selection ranks.
The classifier is linear, so each coarse token has an exact per-label evidence score
`s = W_t·x + b_t` (a class-activation map). Paired study bootstraps (1,000 draws) against
attention; artifacts in `outputs/analysis/l/label-private-selection-probe-20260916/` and
`outputs/previews/label-private-cam-20260916/`.

| Selection value | Macro AUC | vs attention [95% interval] | Map entropy | Chosen-canvas share | Edge-slice crops |
|---|---:|---|---:|---:|---:|
| Attention | 0.848914 | — | 0.96 | 0.035 | 6.4% |
| Softmax of `s` | 0.846620 | −0.0023 [−0.0042, −0.0002] | 0.85 | 0.109 | 14.2% |
| Softmax of attention score + `s` | 0.848311 | −0.0006 [−0.0026, +0.0013] | 0.76 | 0.148 | 12.1% |
| Attention × relu(`s`) | 0.847927 | −0.0010 [−0.0028, +0.0009] | 0.80 | 0.124 | 10.9% |
| Random | 0.844647 | −0.0043 [−0.0064, −0.0021] | — | — | 21.6% |

- **Selection is worth about 0.004 with shared pooling** (0.014 with private pooling) at this
  geometry.
- **Evidence maps are sharper but not better aimed.** They share only 3–7% of cells with
  attention, double the crops on the first or last sampled slice, and in the renders highlight
  bright superficial or fluid signal; fracture (−0.008 to −0.011) and the menisci lost most. They
  are no more finding-specific than attention (positive-minus-negative shift 3.0–3.7 mm against
  3.1 mm over all 904 studies).
- The checkpoint's fine features were trained on attention-selected crops, which biases the probe
  towards attention; still, no evidence variant won, so selection stays on attention.

## Placement

`scripts/adhoc/analyze_crop_diversity.py` (removed) on the same 48 validation studies (seed 20260916):

- 24 crops per study, 19.4 distinct (canvas, cell) pairs; every label used 2 planes in every study.
- Crops cover the selectable area: 34–36 of 36 cells per plane, modal cell share 0.07–0.10 (per-label
  0.14–0.22), offset std 10–15 mm, between-study std 6–9 mm (per-label 3–6 mm).
- **Placement now responds to findings.** Crop centres of positive minus negative studies:
  effusion −14.5 mm vertical, fracture −9.3, synovitis −8.2, lateral meniscus −5.4, PF OA −5.5,
  ACL −4.4, MCL −6.2 horizontal. Earlier runs never exceeded 3.6 mm.
- Plane choices are anatomical: Baker's cyst and fracture favour axial, the menisci and OA
  coronal and sagittal, effusion and synovitis axial and coronal.
- Label mean offsets (image x, y; negative y is up): effusion (−7.0, −8.8), synovitis (−5.2, −9.4)
  and PF OA (−2.5, −8.9) sit high; lateral meniscus (1.1, +14.2) and ACL (0.3, +9.0) sit low.

## Visual review

All canvases and crops of four studies were rendered with the best checkpoint in
`outputs/previews/label-private-crops-20260916/`, including the two fracture studies of the
per-label review; this is not a radiological read.

- **Detected (…66582736):** fracture 0.72 (per-label 0.92), with its crops on a coronal and a
  sagittal fluid slice pushed down onto the proximal-tibia oedema. Contusion 0.89, effusion 0.95,
  ACL 0.76. Several labels chose the same sagittal slice.
- **Missed (…61922832):** fracture 0.11, crops on an axial and a sagittal slice away from the
  lateral tibial foci; effusion fell to 0.39.
- Many boxes sit at the top of the selectable range on slices where the top of the crop holds
  little anatomy (MCL and medial meniscus on coronal slices of …66582736, effusion on a coronal
  slice of …61922832).

### Why some crops sit high

Each label's coarse attention was rendered behind its crops in
`outputs/previews/label-private-attention-20260916/`, and positions were counted over 1,200 crops
of 50 studies (the 48 above plus the two fracture studies):

- **Attention is nearly flat.** A label's attention on its chosen canvas has a normalised entropy of
  0.97–0.99, and the chosen canvas holds about 3% of the label's study attention, roughly the
  average canvas. Canvas and position are chosen on weak tilts.
- **A crop moves away from low attention more than towards a target.** The crop covers 14 of 19
  scout rows, so its 6 positions differ only in which 5 edge rows are dropped. Crops at the top
  position (19% of crops; bottom 17%) have 29% of the canvas attention in the top 5 rows and 13%
  in the bottom 5, where a flat map would give 26% each.
- **It is label-specific.** The top position takes 43% of effusion crops, 39% of synovitis and
  PF OA, 30% of contusion and 24% of fracture, but 3–6% of ACL, menisci and OA crops. On axial
  slices the top is the anterior knee (patella, suprapatellar recess), which suits the first three.
- **Single bright spots drive whole groups.** In …61922832, effusion, synovitis, contusion and MCL
  all moved to the top of the same coronal slice, following one bright spot at the upper edge.
- **Unfocused crops are mostly canvas choices.** Fracture on a superior axial slice and ACL on
  axial slices away from the notch come from flat attention across canvases and plane spread
  forcing a second plane.

## Reading

- **Private pooling starved the fine pathway.** Two crops per label, each trained by one label,
  gave ReXNet-200 too little supervision; the collapse is largest where detail matters, and the
  probes show it cannot be fixed at inference.
- **The trained coarse pathway is the useful part.** Once the scout's pool was trained on its own
  (gate) rather than only as a selection prior, the draft rose from 0.63 to 0.82 and placement
  became study-specific for the first time.
- **The value map is still weak.** Per-label attention is close to uniform, so placement and
  canvas choice follow faint tilts and single bright spots; sharper per-label scores matter more
  than placement rules.
- **FiLM and the gate did nothing measurable**; they add parameters and a split backbone forward.
- **Follow-up:** keep independent per-label selection and the trained coarse pathway, return to shared
  fine pooling, and drop FiLM; this became the [shared-crops run](2026-09-16-shared-crops-result.md).
