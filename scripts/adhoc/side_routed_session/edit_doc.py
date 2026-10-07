from pathlib import Path

p = Path("docs/ongoing/side-routed-heads.md")
s = p.read_text()


def between(start, end, new):
    global s
    a = s.index(start)
    b = s.index(end)
    s = s[:a] + new + s[b:]


between(
    "**Status (2026-10-05)",
    "One shared meniscus classifier",
    "**Status (2026-10-05): built in worktree `knee-abnormality-side-routed-heads`, branch\n"
    "`side-routed-heads`, uncommitted; [first run](#first-run-2026-10-05) at 0.923102 against\n"
    "0.924049; the design below is the [second build](#second-build-2026-10-05), not yet run.** ",
)
between(
    "**Geometry enters the model in one place",
    "- **Shared classifiers.**",
    """**A shared classifier never learns which compartment a token is in; geometry assigns its
output to the targets.** Every token carries its position in the knee's own frame, and the two
kinds of classifier read it differently: a shared one sees the distance from the midline without
its sign, so a medial and a lateral token at the same depth and height look the same; a
classifier with a target of its own sees the full position, medial positive for every knee.

""",
)
between(
    "- `u` is the token's left-right offset",
    "### Knee side",
    """- Token coordinates are in the knee's frame (below): `u` left-right, `a` anterior, both in mm
  from the fitted centre; `s` is the side sign, `o_lat` the learned lateral offset and `d_a` the
  study's front boundary.
- `r_medial = sigmoid((s * u + o_lat) / width_m)`, `r_lateral = 1 - r_medial`,
  `r_front = sigmoid((a - d_a) / width_a)`. The split has no band: every token belongs to one
  half, softly near the line.
- The two widths are learned scalars, start at 5 mm and are kept within 1 to 40 mm.
- PF OA uses `r_front` alone; the front tokens also count for their side in Medial and Lateral
  OA, as in the removed detector.
- A token without a fitted centre takes 0.5 for every routing weight. If a study's routing
  weights for a target sum to less than one token, that target pools unrouted.

### Knee frame

- **Rotation from the headers.** The scanner's left-right axis is rotated about the head-foot
  axis to the mean row direction of the study's axial and coronal series; a study without one
  keeps the scanner's axes. Sagittal series are angled on purpose and are not used
  ([census](#visual-inspection-and-rotation-census)).
- **Origin.** The fitted centre plus three learned offsets, lateral, front and up, common to
  all studies. Each is `40 mm * tanh(parameter)` and starts at zero. Nothing is set by hand.
- **Front boundary per study.** One scorer rates every axial and sagittal token for "this is
  the front boundary" and the boundary is the weighted average of those tokens' anterior
  coordinates:

  ```text
  weight_i = softmax over the study's eligible tokens of (score_i - (a_i - o_front)^2 / (2 * 15^2))
  d_a      = sum of weight_i * a_i
  ```

  The scorer starts at zero and reads tokens without the position code. A study with no
  eligible token takes `o_front`.
- **The midline has no per-study part.** The first run's per-study midline landmark learned one
  constant, about 2 mm lateral, unrelated to the study; the lateral offset replaces it.

### Position code

```text
position = (s * u + o_lat, a - d_a, z - o_up)          in mm
code     = Linear(64, 256)(GELU(Linear(42, 64)(sin and cos at 7 wavelengths, 8 to 512 mm)))
```

- A classifier with its own target reads tokens with `code(position)`; a shared classifier
  reads the same tokens with `code(|s * u + o_lat|, a - d_a, z - o_up)`. The token is
  normalised after the code is added, so each token exists in two versions.
- The output layer starts at zero: the model starts without a position code.
- A token without a fitted centre takes one learned vector, also zero at the start.
- The offsets and the front boundary move every token's position, so all twelve targets train
  them, not only the routed five.
- For a study without a certain side, `s` is `1 - 2 * p_left`.
- The report-attribute heads read the signed version.

""",
)
between(
    "- **Every position code:**",
    "Kept: the centre fit",
    """- The 1D slice position and the 2D image position; the knee-frame position code replaces
  them.

""",
)
between(
    "The mean, spread and range of the learned centre corrections",
    "## Reference and first run",
    """The three learned offsets; the mean, spread and range of the front boundary over studies and
the share of its landmark's weight per series type; the mean absolute rotation of the knee
frame; the two widths; the side head's accuracy on certain studies, in training and validation;
for coordinates-only studies, how often the final probability sides with the coordinates; the
share of studies in each side state; the mean routing weight per compartment; the attention
mass of each target per view.

""",
)
between(
    "## Next\n",
    "## Variants considered",
    """## Second build (2026-10-05)

Built after the first run, with the user: the knee frame, the learned offsets in place of the
per-study midline landmark, and the position code read unsigned by the shared classifiers and
signed by the others. The first run had no position code, a per-study midline landmark, a
front landmark centred on the fitted centre, and the scanner's axes.

Checked on four training studies on the GPU: the position code is zero at the start; the
rotation matches the census per study; with random position weights a side flip swaps the
medial and lateral logits of both shared channels exactly, leaves PF OA unchanged and moves the
own-channel logits; a mirrored test-time pass keeps each token's position; the offsets, the
front scorer and the position output receive gradients.

```bash
rtk proxy uv run --locked just train configs/training.yaml --name side-routed-knee-frame
```

Same fold, seed and reference as the first run.

Not built: a learned crop shift from the same centre, with the zoom fixed. The joint line
varies by about ±5 mm between studies, under half a token, so it runs after this one.

""",
)
s = s.replace(
    "| A side-agnostic 3D position on the tokens (distance from the learned midline, front, up) | Built, then removed: location is only needed to route |",
    "| No position code at all | The first run |",
)
s = s.replace(
    "| One centre correction common to all studies; the fitted centre with nothing learned | Not run |",
    "| A per-study midline landmark | The first run: it learned a constant |\n| The fitted centre with nothing learned | Not run |",
)
p.write_text(s)
