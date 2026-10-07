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


sub(
    "is the [third build](#third-build-2026-10-05), whose [run](#third-run) reached 0.923085.** ",
    "is the [fourth build](#fourth-build-2026-10-05), not yet run; the [third run](#third-run)\nreached 0.923085.** ",
)
between(
    "- **Side head:**",
    "- **Reflection:**",
    """- **Side head:** a small network (256 to 128, tanh, to 1) scores each of the study's axial and
  coronal tokens with a fitted centre; the logit for "left knee" is the mean of those scores
  weighted by `tanh(u / width)`. Its input is detached, so the side loss trains this network and
  its width and nothing else. The width is learned like the routing widths. Binary
  cross-entropy on the certain studies, weight 0.1. A study without such tokens has no head
  logit.
""",
)
between("### Auxiliary loss\n", "Kept: the centre fit", "")
sub(
    "The two learned offsets; the routed auxiliary loss; the mean, spread and range",
    "The two learned offsets; the mean, spread and range",
)
sub(
    "### Investigation of the first two runs",
    """## Fourth build (2026-10-05)

The third build with two changes, from what the three runs showed:

| Change | Reason |
|---|---|
| The routed auxiliary loss and its detector are removed | Added on weak evidence; the shared routed classifier already is the per-half detector. MCL and Baker's fell below all 12 sibling runs only in the run that had it, and both sit on the medial side where it acts (a suspicion, not shown) |
| The side head is a per-token network instead of one linear layer on the pooled tokens | On detached tokens the linear head reached 96.5%, against 100% when it trained the trunk |

Kept from the third build: the detached side head and front scorer, the position prior on the
attention, position-free tokens, the knee frame, the two learned offsets and the temperature
on the prior and routing term.

The same GPU check as the third build passes: no side-loss gradient on the trunk, a zero prior
at the start, the exact medial/lateral swap under a side flip, gradients on every new part.

```bash
rtk proxy uv run --locked just train configs/training.yaml --name side-routed-run4
```

Same fold, seed and reference. Checkpoints of the earlier runs no longer load in this code.

### Investigation of the first two runs""",
)
sub(
    "| The side head on the mean of undetached tokens | The first two runs: it trained the trunk |",
    "| The side head on the mean of undetached tokens | The first two runs: it trained the trunk |\n| A routed half-knee auxiliary loss on the meniscus targets, weight 0.075; a linear side head on pooled detached tokens | The third run |",
)
p.write_text(s)
