# Side-routed heads: fourth and fifth builds (2026-10-05)

**Removing the routed auxiliary loss brought MCL back and gave the fourth run 0.953 public (S113);
the fifth build's octants (fitted centre plus one learned rotation, own-channel position code) were
the first run above the reference at ×4 (0.924476 against 0.924049, S114 0.952).** The public score
followed the routed targets' sum, not the local macro. Patient motion between series was measured
small enough that no per-series correction was made. Design:
[side-routed heads](../../../../modeling/side-routed-heads.md).

## Fourth build (2026-10-05)

The third build with two changes, from what the three runs showed:

| Change | Reason |
|---|---|
| The routed auxiliary loss and its detector are removed | Added on weak evidence; the shared routed classifier already is the per-half detector. MCL and Baker's fell below all 12 sibling runs only in the run that had it, and both sit on the medial side where it acts (a suspicion, not shown) |
| The side head is a per-token network instead of one linear layer on the pooled tokens | On detached tokens the linear head reached 96.5%, against 100% when it trained the trunk |

Kept from the third build: the detached side head and front scorer, the position prior on the
attention, position-free tokens, the knee frame, the two learned offsets and the temperature on
the prior and routing term. The same GPU check as the third build passed. Checkpoints of the
earlier runs do not load in this code.

```bash
uv run --locked just train configs/training.yaml --name side-routed-run4
```

### Fourth run

`outputs/training/20261005-202001-side-routed-run4`, fold 1, seed 1110484336, one run: **×4
0.923952** (reference 0.924049; the 12 same-fold campaign runs span 0.9234 to 0.9248); ×1
0.915445; BCE at ×1 0.4085 against 0.4092. **Kaggle: S113, 0.953 public** (×4, τ 0.9, no
reflection TTA; bundle `outputs/submission/exports/side-routed-run4-f1-x4-tau09`), against 0.952
for the second run (S112) and 0.950 for the campaign recipe on the same fold (S108, S111).

| Target | Fourth − reference | Siblings − reference, mean | `position-embeddings-off` − reference |
|---|---:|---:|---:|
| ACL | −0.0034 | −0.0016 | −0.0018 |
| MCL | −0.0031 | +0.0005 | −0.0018 |
| Medial Meniscus | −0.0002 | −0.0002 | |
| Lateral Meniscus | −0.0001 | −0.0015 | |
| Medial OA | +0.0030 | +0.0005 | |
| Lateral OA | +0.0022 | −0.0024 | |
| PF OA | +0.0082 | −0.0007 | |
| Effusion | +0.0012 | +0.0007 | +0.0021 |
| Synovitis | −0.0054 | −0.0041 | −0.0035 |
| Baker's | −0.0057 | +0.0001 | −0.0028 |
| Contusion | +0.0036 | +0.0025 | +0.0036 |
| Fracture | −0.0015 | +0.0072 | +0.0002 |

- **The five routed targets summed to +0.0131** (third run +0.0080, second +0.0033, first
  −0.0035); PF OA and Medial OA were above all 12 siblings.
- **MCL came back** from −0.0128 to −0.0031 with the auxiliary loss removed.
- **The side head was back at 99.8%** on certain validation studies (99.2% agreement with
  coordinates-only sides) as a per-token network on detached tokens.
- **The own-channel targets carried the rest.** Against the sibling mean they summed to −0.0196
  (−0.0016 of macro), most of it Fracture (−0.0087) and Baker's (−0.0058), both below all 12
  siblings. `position-embeddings-off`, the reference recipe without the token position codes,
  was the one sibling with the same profile: the lowest sibling on Baker's and, at +0.0002
  against +0.0043 to +0.0117 for the others, on Fracture. One run each.
- Learned: lateral offset 2.0 mm, front offset 2.9 mm, front boundary 11.5 mm, widths 4.5, 4.9
  and 4.3 mm.
- At ×1 the run led the reference through epoch 4 (+0.0070, +0.0022, +0.0027, +0.0022) and
  trailed by 0.0006 to 0.0013 afterwards.

## Fifth build (2026-10-05)

Designed with the user after the fourth run, as one combined change. It was the base and starting
incumbent of the side-routing campaign
([contract](../../../../../autoresearch/campaigns/side-routing/contract.md)).

| Change | Reason |
|---|---|
| The frame is the fitted centre plus one learned rotation; the lateral offset, the front offset and the per-study front scorer are removed | The user's octant picture; the offsets were small or unneeded, and a cut that cannot slide cannot be abused |
| Medial and Lateral OA take the back of their side only | The front cut then has targets competing across it |
| Own-channel classifiers read tokens with a position code at the reference's scale | Fracture and Baker's were below all 12 siblings in the fourth run, the profile of `position-embeddings-off` |

Kept from the fourth build: the detached per-token side head, the position prior on the
attention, position-free tokens for the shared classifiers, the header rotation and the
temperature on the prior and routing term.

**Principle.** The fitted centre and a learned rotation cut the knee into eight octants; routing
was a table from octants to targets. A cut was used for routing only where targets competed
across it: medial against lateral targets across the midline, PF OA against the two
tibiofemoral OA targets across the front cut. No target pair was split by height, so the height
cut routed nothing.

**Routing.** Only the five compartment targets were routed; the seven own-channel targets pooled
unrouted over all valid tokens.

| Target | Region | Routing weight |
|---|---|---|
| Medial Meniscus | medial | `r_medial` |
| Lateral Meniscus | lateral | `1 - r_medial` |
| Medial OA | medial and back | `r_medial * (1 - r_front)` |
| Lateral OA | lateral and back | `(1 - r_medial) * (1 - r_front)` |
| PF OA | front, both sides | `r_front` |

- `r_medial = sigmoid(m / width_m)` and `r_front = sigmoid(f / width_f)`, with `(m, f, h)` the
  token's coordinates in the knee frame. A cut had no band: every token belonged to one side,
  softly near the cut.
- The two widths set how soft each cut was: a token one width from a cut counted 73% for its
  side. They were learned scalars, started at 5 mm and were kept within 1 to 40 mm.
- A token without a fitted centre took 0.5 for every routing weight. If a study's routing
  weights for a target summed to less than one token, that target pooled unrouted.

**Knee frame.**

```text
(m, f, h) = R * (s * u, a, z)
```

- `(u, a, z)` were the token's left-right, anterior and head-foot offsets from the fitted centre
  in mm, after the rotation from the headers
  ([census](2026-10-05-first-to-third-builds.md#visual-inspection-and-rotation-census)); `s` the
  side sign, so `m` was medial-positive for every knee.
- `R` was one learned rotation about the fitted centre, common to all studies: three angles (nod
  about the medial axis, roll about the front axis, turn about the vertical axis), each
  `90° * tanh(parameter)`, starting at zero. The nod was meant to tilt the front cut so that the
  kneecap and the trochlea fall in front and the tibiofemoral joint line behind
  ([sketches](#octant-sketches)).
- There was no offset and nothing per study. A cut pinned to the centre can turn but cannot
  slide, so PF OA could not widen its region to the whole knee. The earlier builds' lateral
  offset learned 2 to 3 mm and their front boundary moved between 12 and 27 mm with PF OA
  improving throughout.

Checked on four training studies on the GPU: the frame started as the identity and the three
angles received gradients; with a random prior a side flip swapped the medial and lateral logits
of both shared channels exactly; the three back and front routing weights summed to one; the
token code had norm 11.2; no side-loss gradient reached the trunk. At the start the front cut
was vertical through the centre, so Medial and Lateral OA began on the back half of their side.
Checkpoints of the earlier runs do not load in this code.

```bash
uv run --locked just train configs/training.yaml --name side-routed-run5
```

### Fifth run

`outputs/training/20261005-205251-side-routed-run5`, one run: **×4 0.924476** (reference
0.924049, fourth run 0.923952); ×1 0.915823. **Kaggle: S114, 0.952 public** (×4, τ 0.9, no
reflection TTA; bundle `outputs/submission/exports/side-routed-run5-f1-x4-tau09`), against
0.953 for the fourth run (S113) and 0.952 for the second (S112). The public score followed the
routed targets' sum (+0.0033, +0.0131, +0.0090 for S112, S113, S114) and not the local macro
(0.9235, 0.9240, 0.9245); three submissions at a resolution of 0.001.

| Target | Fifth − reference | Fourth − reference | Siblings − reference, mean | Fifth below how many of 12 |
|---|---:|---:|---:|---:|
| ACL | −0.0041 | −0.0034 | −0.0016 | 11 |
| MCL | +0.0012 | −0.0031 | +0.0005 | 4 |
| Medial Meniscus | −0.0012 | −0.0002 | −0.0002 | 11 |
| Lateral Meniscus | +0.0015 | −0.0001 | −0.0015 | 3 |
| Medial OA | +0.0013 | +0.0030 | +0.0005 | 3 |
| Lateral OA | +0.0003 | +0.0022 | −0.0024 | 3 |
| PF OA | +0.0071 | +0.0082 | −0.0007 | 0 |
| Effusion | +0.0031 | +0.0012 | +0.0007 | 0 |
| Synovitis | −0.0027 | −0.0054 | −0.0041 | 2 |
| Baker's | −0.0035 | −0.0057 | +0.0001 | 12 |
| Contusion | +0.0021 | +0.0036 | +0.0025 | 8 |
| Fracture | +0.0000 | −0.0015 | +0.0072 | 11 |

- **First run of the line above the reference** (+0.0004), with the own-channel targets summing
  to −0.0039 against −0.0143 in the fourth run: MCL, Synovitis, Effusion, Baker's and Fracture
  all moved up with the token position code. Baker's was still below all 12 siblings, Fracture
  and ACL below 11.
- **The routed five summed to +0.0090** (fourth run +0.0131). Medial OA and Lateral OA, now on
  the back of their side only, gave back 0.0017 and 0.0019; PF OA stayed above all siblings.
- **The frame barely turned:** nod −5.1° (the front cut leaned 5° towards front-and-up, against
  the 25° to 45° of the sketches), roll −2.4°, turn −1.0°. The front width grew from 5.0 to
  5.7 mm; the medial width ended at 4.5 mm.
- The side head reached 99.8% on certain validation studies.
- At ×1 the run led the reference through epoch 5 and ended 0.0003 behind.

### Octant sketches

`outputs/analysis/o/octant-sketches-20261005/` (loader only): the cuts drawn on midline sagittal,
coronal and axial slices of two studies, and `tilt-sheet/sheet.png` with the front cut at 0°,
25° and 45° on ten validation studies. The fitted centre sits at the notch, about 15 mm above
the joint line on the coronal slice. With the cut through the centre, a nod of 25° to 45°
leaves the kneecap, the trochlea and the front of the femur in front and the joint line, the
tibial plateau and the back of the condyles behind in all ten. The angles are illustrations,
not learned values.

### Patient motion between series

Measured by a subagent on the full training set, 2026-10-05
(`outputs/analysis/i/inter-series-motion-20261005/`; scripts in
`scripts/adhoc/inter_series_motion/`). Every pair of same-plane series was resampled by header
geometry onto a 1 mm grid and aligned by a normalised-gradient-field correlation over ±20 mm;
12,083 comparable pairs in 4,398 studies (pairs angled more than 3° apart excluded).

| | Median | p90 | p95 | p99 | Over 3 mm | Over 5 mm |
|---|---:|---:|---:|---:|---:|---:|
| Same-plane shift, all pairs | 0.41 mm | 1.15 | 1.48 | 2.71 | 0.84% | 0.29% |
| Worst pair per study | 0.66 mm | | 1.87 | 3.77 | 1.61% of studies | 0.61% (27 studies) |
| Crossing planes, 200 studies, along the intersection | 0.36 mm | 1.33 | 1.82 | 3.20 | 1.1% | 0.21% |

- Validation: a synthetic shift was recovered with a median error of 0.08 mm; a triangle
  closure over same-plane and crossing estimates left a median residual of 0.14 mm (p99 1.11);
  adjacent slices of one series gave a median of 0.36 mm.
- The nine largest cases viewed were real repositioning, some beyond the 20 mm search limit.
  Shift grows with the gap in series number (median 0.36 mm for neighbours, 1.08 mm seven
  apart); repeated same-contrast series carry the heavier tail.
- **Decision: no per-series correction.** Tokens are 12 to 17 mm apart and the cuts were soft
  over about 4.5 mm; 99% of pairs move under 2.7 mm. Not measured: through-plane shift and
  rotation.
