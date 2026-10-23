# Side-routed heads: first to third builds (2026-10-05)

**The first three builds sat 0.0005 to 0.0010 below the reference at ×4 (0.924049), with the five
routed targets gaining and ACL, Synovitis and, in the third run, MCL losing; the second run scored
0.952 public (S112) against 0.950.** The side loss trained the trunk and attention left the axial
fluid view; the third build detached the side head and moved position into a linear attention
prior, which learned anatomical places. Design: [side-routed heads](../../../../modeling/side-routed-heads.md).

## Reference and first run

The reference is the fixes-and-removals campaign incumbent `slab-rule-and-positive-std-requeue`
(`outputs/autoresearch/fixes-and-removals/experiments/slab-rule-and-positive-std-requeue-20261005T155303Z-59280bbc/training`,
fold 1 of `all_5fold_scanner_v2`, seed 1110484336, ×4 0.924049, ×1 0.916134). The worktree
`knee-abnormality-side-routed-heads` started from that run's code and `configs/resnet50.yaml`
carried its fold and seed.

```bash
uv run --locked just train configs/training.yaml --name side-routed-heads
```

One run of this fold moves the macro by about 0.0005 and a 60-positive label by about 0.01 from
a changed data stream alone
([run notes](../../../../../autoresearch/campaigns/fixes-and-removals/run-notes.md)), so the five
compartment targets were read together and the validation loss beside the macro.

The first build routed the shared heads by soft medial and front cuts with learned widths, a
per-study midline landmark, a front landmark centred on the fitted centre and the scanner's axes,
with no position code on the tokens.

## First run (2026-10-05)

`outputs/training/20261005-181012-side-routed-heads`, fold 1, seed 1110484336, one run.

| | Run | Reference | Δ |
|---|---:|---:|---:|
| Macro AUC ×4 | 0.923102 | 0.924049 | −0.0009 |
| Macro AUC ×1 (epoch 10) | 0.914604 | 0.916134 | −0.0015 |
| Soft-target BCE ×4 | 0.4025 | 0.4037 | −0.0012 |

| Target | Run | Reference | Δ | Positives | Routed |
|---|---:|---:|---:|---:|---|
| ACL | 0.9415 | 0.9458 | −0.0043 | 131 | no |
| MCL | 0.9177 | 0.9176 | +0.0001 | 47 | no |
| Medial Meniscus | 0.9460 | 0.9469 | −0.0009 | 339 | yes |
| Lateral Meniscus | 0.9054 | 0.9081 | −0.0027 | 135 | yes |
| Medial OA | 0.9241 | 0.9229 | +0.0012 | 189 | yes |
| Lateral OA | 0.9016 | 0.9035 | −0.0019 | 95 | yes |
| PF OA | 0.9002 | 0.8994 | +0.0008 | 290 | yes |
| Effusion | 0.9190 | 0.9182 | +0.0007 | 244 | no |
| Synovitis | 0.8783 | 0.8900 | −0.0118 | 114 | no |
| Baker's | 0.9630 | 0.9636 | −0.0007 | 114 | no |
| Contusion | 0.9531 | 0.9468 | +0.0063 | 93 | no |
| Fracture | 0.9275 | 0.9257 | +0.0018 | 60 | no |

- The five routed targets summed to −0.0035, which is −0.0003 of macro. Synovitis alone was
  −0.0010 of macro; a label of its size moves by about 0.01 on this fold from a changed data
  stream.
- At ×1 the run led at epoch 1 (+0.0037) and trailed by 0.001 to 0.004 afterwards. Lateral
  Meniscus led through epoch 7 (up to +0.019) and PF OA through epoch 4; neither lead held.

### What the new parts learned

| | Epoch 1 | Epoch 9 |
|---|---:|---:|
| Side head accuracy on certain validation studies | 100% | 100% |
| Side head agreement with coordinates-only sides | 99.5% | 100% |
| Front boundary correction, mean | 13.5 mm | 27.0 mm |
| Midline correction, spread over studies | 0.6 mm | 3.2 mm |
| Medial/lateral width | 4.8 mm | 4.4 mm (epoch 6) |
| Front width | 4.9 mm | 4.7 mm (epoch 6) |

### Visual inspection and rotation census

Eight validation studies rendered with `scripts/adhoc/render_routing_views.py`
(`outputs/analysis/s/side-routed-views-20261005/`), and the rotation of every training series
measured from headers with `scripts/adhoc/census_knee_rotation.py`
(`outputs/analysis/k/knee-rotation-census-20261005/`).

- **Front boundary:** on the patellofemoral joint in all eight, 15 to 39 mm ahead of the fitted
  centre over 60 studies (mean 27.2, SD 5.6).
- **Fitted midline:** on the intercondylar notch on every coronal slice.
- **Learned midline:** a constant, not a per-study correction. Over 60 studies it moved lateral
  in 82%, by 2.1 mm on average (SD 2.6), and that drift was the same in every rotation band
  (correlation with the knee's rotation 0.04). The two largest corrections (8 and 9 mm) left
  the notch for the lateral tibial spine.
- **Crops:** every coronal and sagittal crop held the joint, both condyles and the plateau. The
  joint line sat at 44% to 69% of the crop height, mostly 55% to 60%, about ±5 mm between
  studies.
- **Rotation:** the first build's routing used the scanner's left-right axis, and the knee is
  rotated against it. Over 4,237 studies with a side, taking the axial and coronal row
  directions: 76% rotated outward, mean 4.5°; over 5° in 49%, over 10° in 19%, over 15° in 6%,
  over 20° in 2%. At 30 mm behind the centre, 10° misplaces the boundary by 5.3 mm and 20° by
  10.9 mm. Axial and coronal agree (correlation 0.83, median difference 1.8°); sagittal series
  are angled on purpose and do not track them (0.43 to 0.50).

## Second build (2026-10-05)

Built after the first run, with the user: the knee frame (header rotation of the census), learned
offsets in place of the per-study midline landmark, and the position code read unsigned by the
shared classifiers and signed by the others.

Checked on four training studies on the GPU: the position code was zero at the start; the
rotation matched the census per study; with random position weights a side flip swapped the
medial and lateral logits of both shared channels exactly, left PF OA unchanged and moved the
own-channel logits; a mirrored test-time pass kept each token's position; the offsets, the
front scorer and the position output received gradients.

```bash
uv run --locked just train configs/training.yaml --name side-routed-knee-frame
```

### Second run

`outputs/training/20261005-185535-side-routed-knee-frame`, same fold, seed and reference, one run.

| | Second run | First run | Reference |
|---|---:|---:|---:|
| Macro AUC ×4 | 0.923534 | 0.923102 | 0.924049 |
| Macro AUC ×1 (epoch 10) | 0.914544 | 0.914604 | 0.916134 |
| Soft-target BCE ×4 | 0.4026 | 0.4025 | 0.4037 |

| Target | Second − reference | First − reference |
|---|---:|---:|
| ACL | −0.0060 | −0.0043 |
| MCL | +0.0005 | +0.0001 |
| Medial Meniscus | −0.0006 | −0.0009 |
| Lateral Meniscus | −0.0007 | −0.0027 |
| Medial OA | +0.0003 | +0.0012 |
| Lateral OA | −0.0002 | −0.0019 |
| PF OA | +0.0044 | +0.0008 |
| Effusion | +0.0015 | +0.0007 |
| Synovitis | −0.0098 | −0.0118 |
| Baker's | +0.0000 | −0.0007 |
| Contusion | +0.0037 | +0.0063 |
| Fracture | +0.0006 | +0.0018 |

- The five routed targets summed to +0.0032 (first run −0.0035). ACL and Synovitis together
  were −0.0158, which is −0.0013 of macro and more than the whole gap of −0.0005; the signed
  position code did not recover them.
- Learned at the end: lateral offset 2.2 mm, front offset 4.8 mm, up offset 1.3 mm; front
  boundary 22.4 mm on average (SD 4.7); widths 4.4 and 4.7 mm; side head 100%.
- **Kaggle: S112, 0.952 public** (×4, τ 0.9, no reflection TTA; bundle
  `outputs/submission/exports/side-routed-knee-frame-f1-x4-tau09`). The same fold and inference
  setting on the campaign recipe scored 0.950 with the OA detector off (S111) and 0.950 with it
  on (S108).
- The trained position code was about a tenth of the token's norm (1.7 and 1.3 against 15.6 on
  40 validation studies) and removing it changed the logits by 0.03 to 0.14, so this run barely
  used position.
- At ×1 the run led the reference through epoch 3 (+0.0041, −0.0002, +0.0007) and trailed by
  0.0012 to 0.0033 afterwards, as the first run did from epoch 2.

Not built: a learned crop shift from the same centre, with the zoom fixed; the joint line varies
by about ±5 mm between studies, under half a token.

### Investigation of the first two runs

A subagent compared both runs with the reference and with 12 same-fold campaign runs.

- **ACL and Synovitis:** both runs were below all 12 sibling runs on Synovitis and below 11 or
  12 on ACL. The reference is a high draw on both: against the sibling mean the second run was
  −0.0044 (ACL) and −0.0057 (Synovitis). PF OA was above all 12 (+0.0051 against their mean).
- **The two runs are one draw:** their per-study logit changes against the reference correlate
  0.89 (ACL) and 0.93 (Synovitis), against about 0.48 for sibling pairs.
- **Position codes are not the cause:** `position-embeddings-off` on the reference recipe gave
  Synovitis −0.0035 and ACL −0.0018, the sibling mean.
- **Attention left the axial fluid view** for every own-channel target, outside the range of
  13 campaign runs (training, last epoch): ACL 0.091 to 0.059, Synovitis 0.415 to 0.335,
  Effusion 0.487 to 0.425. Synovitis' coronal-fluid single-view AUC fell from 0.812 to 0.766.
- **Routing localises:** attention centres at +22 mm (Medial Meniscus), −21 (Lateral Meniscus),
  +19 (Medial OA, reference +11), −17 (Lateral OA, reference −11), 34 mm anterior (PF OA,
  reference 16).
- **OA alone and together (×4, second run minus reference):** against studies with OA only in
  another compartment, PF OA +0.0099 (reference 0.7377) and Medial OA +0.0047 (0.8288), Lateral
  OA −0.0002 (0.8134); isolated Medial OA against OA-free knees −0.0096 (53 positives).
- The anatomical attention bias was already off in the reference (`model.anatomical_bias:
  false`), so it was not removed by this line.
- No coding bug found in initialisation, parameter groups, the routing fallback, the view-bias
  indexing or the detached side probability.

## Third build (2026-10-05)

Built after the second run and the [investigation](#investigation-of-the-first-two-runs), as one
combined change:

| Change | Reason |
|---|---|
| Side head reads detached tokens, pooled with a left-right weight | The side loss trained the trunk: a probe for the side scores 87% on the reference's features and 100% on the second run's |
| Front scorer reads detached tokens | Same reason |
| Position acts on the attention through a linear prior per target | The second run's token code ended at a tenth of the token's norm |
| Tokens carry no position code | The classifiers stay blind to the compartment by construction |
| The up offset is gone | It only shifted the token code |
| Routed auxiliary loss on the meniscus targets | The reference's half-knee loss was the one trunk-level term removed |
| Temperature also divides the prior and the routing term | At τ 0.9, S112 sharpened the content score only |

Checked on four training studies on the GPU: the side loss had no gradient on the trunk; the
prior was zero at the start; with a random prior a side flip swapped the medial and lateral
logits of both shared channels exactly, left PF OA unchanged and moved the own-channel logits;
the prior, the detector, the offsets, the three widths, the front scorer and the side head
received gradients; a temperature of 0.5 set the routing temperature and changed the logits.
Checkpoints of the first two runs do not load in this code.

```bash
uv run --locked just train configs/training.yaml --name side-routed-attention-prior
```

### Third run

`outputs/training/20261005-194836-side-routed-attention-prior`, one run: **×4 0.923085**
(reference 0.924049, second run 0.923534, first run 0.923102); ×1 0.914630; BCE 0.4027. Not
submitted.

Each run minus the reference at ×4, beside the 12 same-fold campaign runs minus the reference:

| Target | Third | Second | First | Siblings, mean ± SD | Third below how many of 12 |
|---|---:|---:|---:|---:|---:|
| ACL | −0.0039 | −0.0060 | −0.0043 | −0.0016 ± 0.0014 | 11 |
| MCL | −0.0128 | +0.0005 | +0.0001 | +0.0005 ± 0.0029 | 12 |
| Medial Meniscus | −0.0011 | −0.0006 | −0.0009 | −0.0002 ± 0.0007 | 11 |
| Lateral Meniscus | +0.0009 | −0.0007 | −0.0027 | −0.0015 ± 0.0024 | 3 |
| Medial OA | −0.0011 | +0.0003 | +0.0012 | +0.0005 ± 0.0008 | 11 |
| Lateral OA | +0.0031 | −0.0002 | −0.0019 | −0.0024 ± 0.0034 | 0 |
| PF OA | +0.0062 | +0.0044 | +0.0008 | −0.0007 ± 0.0005 | 0 |
| Effusion | +0.0010 | +0.0015 | +0.0007 | +0.0007 ± 0.0008 | 3 |
| Synovitis | −0.0042 | −0.0098 | −0.0118 | −0.0041 ± 0.0022 | 6 |
| Baker's | −0.0031 | +0.0000 | −0.0007 | +0.0001 ± 0.0015 | 12 |
| Contusion | +0.0026 | +0.0037 | +0.0063 | +0.0025 ± 0.0013 | 7 |
| Fracture | +0.0008 | +0.0006 | +0.0018 | +0.0072 ± 0.0044 | 10 |

- **Synovitis was back at the sibling mean** and ACL moved half-way; the five routed targets
  summed to +0.0080 (second run +0.0032), with PF OA and Lateral OA above all 12 siblings.
- **MCL fell to −0.0128, below all 12 siblings**, and was the whole macro gap (−0.0011); Baker's
  was also below all 12. Both were level in the first two runs.
- **The side head was weaker on detached tokens:** 96.5% on certain validation studies (93.5% in
  training) and 94.2% agreement with coordinates-only sides, against 100% before; its width
  went from 5 to 3.2 mm. The final side probability still agreed with the coordinates in 100%
  of coordinates-only studies; the 6.2% of studies with no side routed by the head alone.
- **The front boundary sat at 12.4 mm** (SD 3.6), against 22.4 mm in the second run; the front
  offset was 2.8 mm and the lateral offset 2.9 mm. Widths 4.2 and 4.9 mm.
- **The position prior learned anatomical places.** Where each target's prior peaks, median
  over 60 validation studies, in mm (medial, ahead of the front boundary, up), with its spread
  over a study's tokens in logits:

  | Target | Peak | Spread |
  |---|---|---:|
  | ACL | −4, −20, −15 | 0.29 |
  | MCL | +39, −10, +3 | 0.25 |
  | Medial / Lateral Meniscus | 34 / 32 into the own side, −19, −17 | 0.38 |
  | Medial / Lateral OA | 21 into the own side, −18, −16 | 0.35 |
  | PF OA | −1, +45, +21 | 0.46 |
  | Effusion | +10, +4, +41 | 0.21 |
  | Synovitis | +9, +29, +26 | 0.20 |
  | Baker's | +26, −42, +8 | 0.34 |
  | Contusion, Fracture | no clear place | 0.15 |

  Removing the prior changed the logits by 0.02 to 0.19.
- The routed auxiliary loss fell from 0.549 to 0.437 over training, as the reference's
  half-knee loss did (0.559 to 0.436).
