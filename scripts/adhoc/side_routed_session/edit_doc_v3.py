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
    "0.924049; the design below is the [second build]",
    "One shared meniscus classifier",
    "0.924049, [second run](#second-run) at 0.923534 and **0.952 public (S112)**; the design below\n"
    "is the [third build](#third-build-2026-10-05), not yet run.** ",
)
between(
    "**A shared classifier never learns which compartment",
    "- **Shared classifiers.**",
    """**The classifiers read what a token shows; geometry decides where to look and which target the
evidence counts for.** No classifier receives a position: tokens are position-free, so a shared
classifier cannot tell which compartment a token is in. The token's position in the knee's own
frame acts on the attention only, through the routing and through a learned prior per target.

""",
)
sub(
    """- **Origin.** The fitted centre plus three learned offsets, lateral, front and up, common to
  all studies.""",
    """- **Origin.** The fitted centre plus two learned offsets, lateral and front, common to all
  studies.""",
)
sub(
    "  The scorer starts at zero and reads tokens without the position code.",
    "  The scorer starts at zero and reads detached tokens, so it does not shape the backbone.",
)
between(
    "### Position code\n",
    "### Knee side\n",
    """### Position prior on the attention

```text
position           = (s * u + o_lat, a - d_a, z)                          in mm
score of target k  = content score of its channel + view bias
                     + log r_k                       (routed targets)
                     + prior_k(position)
prior              = Linear(42, 10)(sin and cos of the position at 7 wavelengths, 8 to 512 mm)
```

| Targets | Prior map | Medial coordinate it reads |
|---|---|---|
| The seven own-channel targets | one each | signed, medial positive |
| Medial and Lateral Meniscus | one shared | depth into the target's own compartment |
| Medial and Lateral OA | one shared | depth into the target's own compartment |
| PF OA | one | absolute |

- The map is linear, starts at zero and adds straight to the attention score, with no hidden
  layer or normalisation between the position and the score.
- Medial and lateral targets share one map read from their own side, so the prior carries no
  compartment either.
- A token without a fitted centre has a prior of zero.
- At inference the temperature divides the content score, the view bias, the prior and the
  routing term together.

""",
)
sub(
    """The side is used by the routing and by the sign of the position that the own-target classifiers
read, and nowhere else.""",
    """The side is used by the routing and by the sign of the medial coordinate in the position prior,
and nowhere else.""",
)
between(
    "- **Side head:**",
    "- **Reflection:**",
    """- **Side head:** one linear layer on the study's axial and coronal tokens with a fitted centre,
  averaged with the weight `tanh(u / width)`, giving a logit for "left knee". Its input is
  detached, so the side loss trains this layer and its width and nothing else. The width is
  learned like the routing widths. Binary cross-entropy on the certain studies, weight 0.1.
  A study without such tokens has no head logit.
""",
)
sub(
    "- **Reflection:** a mirrored study looks like the other knee, so the head's target flips when",
    "- **Reflection:** a mirrored study looks like the other knee, so the head reads `-u` and its\n  target flips when",
)
between(
    "- The 1D slice position and the 2D image position;",
    "Kept: the centre fit",
    """- The 1D slice position and the 2D image position on the tokens; the position prior on the
  attention replaces them.

### Auxiliary loss

One linear detector scores every token. For each of the two meniscus targets its scores are
pooled over the study as a log-mean-exp weighted by that target's routing weight, and trained
with the class-balanced binary cross-entropy, weight 0.075. It is the half-knee loss of the
reference with the routing weights in place of the fixed halves; it is used in training only.

""",
)
sub(
    "The three learned offsets; the mean, spread and range of the front boundary",
    "The two learned offsets; the routed auxiliary loss; the mean, spread and range of the front boundary",
)
sub("frame; the two widths;", "frame; the three widths;")
sub(
    "## Variants considered",
    """## Third build (2026-10-05)

Built after the second run and an investigation of both runs
([findings](#investigation-of-the-first-two-runs)), as one combined change:

| Change | Reason |
|---|---|
| Side head reads detached tokens, pooled with a left-right weight | The side loss trained the trunk: a probe for the side scores 87% on the reference's features and 100% on the second run's |
| Front scorer reads detached tokens | Same reason |
| Position acts on the attention through a linear prior per target | The second run's token code ended at a tenth of the token's norm |
| Tokens carry no position code | The classifiers stay blind to the compartment by construction |
| The up offset is gone | It only shifted the token code |
| Routed auxiliary loss on the meniscus targets | The reference's half-knee loss was the one trunk-level term removed |
| Temperature also divides the prior and the routing term | At τ 0.9, S112 sharpened the content score only |

Checked on four training studies on the GPU: the side loss has no gradient on the trunk; the
prior is zero at the start; with a random prior a side flip swaps the medial and lateral logits
of both shared channels exactly, leaves PF OA unchanged and moves the own-channel logits; the
prior, the detector, the offsets, the three widths, the front scorer and the side head receive
gradients; a temperature of 0.5 sets the routing temperature and changes the logits.

```bash
rtk proxy uv run --locked just train configs/training.yaml --name side-routed-attention-prior
```

Same fold, seed and reference. Checkpoints of the first two runs no longer load in this code.

### Investigation of the first two runs

A subagent compared both runs with the reference and with 12 same-fold campaign runs.

- **ACL and Synovitis:** both runs are below all 12 sibling runs on Synovitis and below 11 or
  12 on ACL. The reference is a high draw on both: against the sibling mean the second run is
  −0.0044 (ACL) and −0.0057 (Synovitis). PF OA is above all 12 (+0.0051 against their mean).
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

## Variants considered""",
)
sub(
    "| No position code at all | The first run |",
    "| No position at all | The first run |\n| A position code on the tokens, unsigned for the shared classifiers | The second run: the code stayed at a tenth of the token's norm |\n| The side head on the mean of undetached tokens | The first two runs: it trained the trunk |",
)
p.write_text(s)
