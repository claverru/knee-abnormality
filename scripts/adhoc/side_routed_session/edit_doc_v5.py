from pathlib import Path

p = Path("docs/ongoing/side-routed-heads.md")
s = p.read_text()


def between(start, end, new):
    global s
    a = s.index(start)
    b = s.index(end, a)
    s = s[:a] + new + s[b:]


def sub(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)


between(
    "0.924049, [second run](#second-run)",
    "One shared meniscus classifier",
    "0.924049, [second run](#second-run) at 0.923534 and 0.952 public (S112), [fourth run](#fourth-run)\n"
    "at 0.923952 (S113 submitted); the design below is the [fifth build](#fifth-build-2026-10-05),\n"
    "not yet run.** ",
)
between(
    "**The classifiers read what a token shows;",
    "- **Shared classifiers.**",
    """**The fitted centre and a learned rotation cut the knee into eight octants; routing is a table
from octants to targets.** A shared classifier reads position-free tokens, so it cannot tell
which compartment a token is in. A cut is used for routing only where targets compete across
it, which holds it in place: medial against lateral targets across the midline, PF OA against
the two tibiofemoral OA targets across the front cut. No target pair is split by height, so the
height cut routes nothing.

""",
)
between(
    "### Routing\n",
    "### Position prior on the attention\n",
    """### Routing

For a compartment target `k` of shared channel `h`, with `r_k` the token's routing weight in
[0, 1]:

```text
attention_k = softmax over the study's valid tokens of (score_h + prior_k + log r_k)
logit_k     = sum over tokens of attention_k * evidence_h + bias_k
```

| Target | Region | Routing weight |
|---|---|---|
| Medial Meniscus | medial | `r_medial` |
| Lateral Meniscus | lateral | `1 - r_medial` |
| Medial OA | medial and back | `r_medial * (1 - r_front)` |
| Lateral OA | lateral and back | `(1 - r_medial) * (1 - r_front)` |
| PF OA | front, both sides | `r_front` |

- `r_medial = sigmoid(m / width_m)` and `r_front = sigmoid(f / width_f)`, with `(m, f, h)` the
  token's coordinates in the knee frame below. A cut has no band: every token belongs to one
  side, softly near the cut.
- The two widths set how soft each cut is: a token one width from a cut counts 73% for its
  side. They are learned scalars, start at 5 mm and are kept within 1 to 40 mm.
- A token without a fitted centre takes 0.5 for every routing weight. If a study's routing
  weights for a target sum to less than one token, that target pools unrouted.

### Knee frame

```text
(m, f, h) = R * (s * u, a, z)
```

- `(u, a, z)` are the token's left-right, anterior and head-foot offsets from the fitted centre
  in mm, after the rotation from the headers: the scanner's left-right axis is rotated about
  the head-foot axis to the mean row direction of the study's axial and coronal series
  ([census](#visual-inspection-and-rotation-census)); a study without one keeps the scanner's
  axes.
- `s` is the side sign, so `m` is medial-positive for every knee.
- **`R` is one learned rotation** about the fitted centre, common to all studies: a rotation
  vector of three angles (nod about the medial axis, roll about the front axis, turn about the
  vertical axis), each `90° * tanh(parameter)`, starting at zero. The nod tilts the front cut so
  that the kneecap and the trochlea fall in front and the tibiofemoral joint line behind
  ([sketches](#octant-sketches)).
- **There is no offset and nothing per study.** A cut pinned to the centre can turn but cannot
  slide, so PF OA cannot widen its region to the whole knee. The earlier builds' lateral offset
  learned 2 to 3 mm and their front boundary moved between 12 and 27 mm with PF OA improving
  throughout.

""",
)
sub(
    "position           = (s * u + o_lat, a - d_a, z)                          in mm",
    "position           = (m, f, h), the knee-frame coordinates                in mm",
)
between(
    "- The 1D slice position and the 2D image position on the tokens;",
    "Kept: the centre fit",
    """- The 1D slice position and the 2D image position on the tokens; the position prior on the
  attention and the token position code of the own-channel classifiers replace them.

### Token position code

A token read by an own-channel classifier carries a fixed sinusoidal code of its knee-frame
position (42 wavelengths from 8 to 512 mm per axis, norm 11.2, the scale of the reference's
slice-position code), added before the normalisation; the position is detached there. A token
read by a shared classifier carries none. A token without a fitted centre has a code of zero.
The report-attribute heads read the own-channel version.

""",
)
sub(
    "The two learned offsets; the mean, spread and range of the front boundary over studies and\nthe share of its landmark's weight per series type; the mean absolute rotation of the knee\nframe; the three widths;",
    "The three frame angles; the mean absolute header rotation of the knee frame; the three widths;",
) if "The two learned offsets; the mean, spread and range of the front boundary over studies and\nthe share of its landmark's weight per series type; the mean absolute rotation of the knee\nframe; the three widths;" in s else None
sub(
    "### Investigation of the first two runs",
    """## Fifth build (2026-10-05)

Designed with the user after the fourth run, as one combined change:

| Change | Reason |
|---|---|
| The frame is the fitted centre plus one learned rotation; the lateral offset, the front offset and the per-study front scorer are removed | The user's octant picture; the offsets were small or unneeded, and a cut that cannot slide cannot be abused |
| Medial and Lateral OA take the back of their side only | The front cut then has targets competing across it |
| Own-channel classifiers read tokens with a position code at the reference's scale | Fracture and Baker's were below all 12 siblings in the fourth run, the profile of `position-embeddings-off` |

Kept from the fourth build: the detached per-token side head, the position prior on the
attention, position-free tokens for the shared classifiers, the header rotation and the
temperature on the prior and routing term.

Checked on four training studies on the GPU: the frame starts as the identity and the three
angles receive gradients; with a random prior a side flip swaps the medial and lateral logits
of both shared channels exactly; the three back and front routing weights sum to one; the
token code has norm 11.2; no side-loss gradient on the trunk.

At the start the front cut is vertical through the centre, so Medial and Lateral OA begin on
the back half of their side until the nod is learned.

```bash
rtk proxy uv run --locked just train configs/training.yaml --name side-routed-run5
```

Same fold, seed and reference. Checkpoints of the earlier runs no longer load in this code.

### Octant sketches

`outputs/analysis/octant-sketches-20261005/` (loader only): the cuts drawn on midline sagittal,
coronal and axial slices of two studies, and `tilt-sheet/sheet.png` with the front cut at 0°,
25° and 45° on ten validation studies. The fitted centre sits at the notch, about 15 mm above
the joint line on the coronal slice. With the cut through the centre, a nod of 25° to 45°
leaves the kneecap, the trochlea and the front of the femur in front and the joint line, the
tibial plateau and the back of the condyles behind in all ten. The angles are illustrations,
not learned values.

### Investigation of the first two runs""",
)
sub(
    "| No position at all | The first run |",
    "| No position at all | The first run |\n| A learned lateral offset, a learned front offset, a per-study front scorer; front tokens also counting for Medial and Lateral OA; no token position for the own-channel classifiers | The first four runs |",
)
p.write_text(s)
