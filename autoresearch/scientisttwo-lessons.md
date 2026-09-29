# Lessons from ScientistTwo

Notes from [ScientistTwo (arXiv 2609.19644)](https://arxiv.org/html/2609.19644), a
multi-agent system that goes from a research problem to a paper: limitation extraction,
seed ideas ranked by novelty, subset-first screening, evolution rounds over execution
traces, automated ablations, a simulated peer-review and rebuttal loop, and an integrity
audit. It reports that gains shrink over successive evolution rounds and that the best
ideas tend to appear early. It does not publish its round and budget settings.

## Already covered here

| ScientistTwo | Autoresearch |
|---|---|
| Limitation extractor feeding the idea generator | Researcher brief: start from observed or suspected weaknesses |
| Execution traces of every hypothesis, reused by the evolver | [Family ledger](records.md#family-verdicts) and history index |
| Subset critic verdict Bad / Good / Engineer | Family verdict close / extend / retry |
| Prune an idea after `N_eng` failed engineering attempts | Code-enforced closure after two consecutive rejected follow-ups |
| Score verification and specification compliance | Code-computed verdict from saved predictions; source and config gates |
| Method-code alignment | Orchestrator integration review of the real diff |

## Measured against our campaign records

[`analyze_campaign_screening.py`](../scripts/adhoc/analyze_campaign_screening.py), run on
2026-09-28 over every decided experiment with epoch logs: 157 runs in 13 campaigns,
5 `provisional_accept`, 126 `reject`, 26 `inconclusive`, all on `all_5fold_scanner_v1` with
`combined_v1` labels. The evaluation moved to `all_5fold_scanner_v2` and `rebuild_state_v2`
the same day, so the absolute thresholds below are untested on the new labels. Runs have 10 epochs; per-epoch
deltas use the ×1 epoch validation of candidate and pinned reference, the final delta the
scored macro AUC.

```bash
rtk proxy uv run --locked python -m scripts.adhoc.analyze_campaign_screening
```

### Subset-first screening does not transfer

Spearman correlation between the epoch-e delta and the final delta (152 runs with
|final| < 0.01):

| Epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Spearman | 0.13 | 0.25 | 0.06 | 0.49 | 0.17 | 0.59 | 0.68 | 0.60 | 0.83 | 0.85 |

A shortened run cannot rank candidates: the ordering only settles in the last epochs.
An abort rule on clearly losing runs is the only usable form:

| Abort when | Aborted | Finished > 0 | Adopted | Epochs saved |
|---|---|---|---|---|
| epoch 4 Δ < −0.005 | 9 | 0 | 0 | 2.9% |
| epoch 5 Δ < −0.003 | 26 | 0 | 0 | 6.6% |
| epoch 5 Δ < −0.005 | 18 | 0 | 0 | 4.6% |
| epoch 6 Δ < −0.003 | 25 | 2 | 0 | 4.8% |

Weak evidence: only five winners, and bilateral smoothing, an adopted run, was 0.010
behind at epoch 0 and 0.005 behind at epoch 2, so earlier cut-offs would have killed it.
At about 5–7% of GPU time, the abort rule is not worth the supervisor complexity and the
lost measurements.

### Wins come early in a campaign

| Origin | Runs | Adopted |
|---|---|---|
| User seed ideas | 68 | 4 (campaign positions 3, 3, 8, 9) |
| Researcher ideas | 89 | 1 (2nd researcher idea of its campaign) |

No campaign adopted anything after its ninth run (65 later runs), and no researcher idea
past the second in its campaign was adopted (about 70 runs). The user adopted learned plane
fusion by hand (+0.0009, below the rule) at position 4 of plane-specialization, also early. This agrees with the paper's
diminishing returns. Confounders: each adoption strengthens the incumbent, later researcher
ideas are mostly follow-ups on rejects, and the user stopped campaigns at different lengths.
Strength: moderate that returns diminish, weak for any exact cut-off (one researcher win).

Stopping a campaign after eight consecutive researcher ideas without an adoption would have
saved 30 of the 157 runs (19%) with no adoption lost; counting seed runs as well would
have stopped near-misses one run before token dropout. The threshold of eight is a guess.

## Proposals

Superseded by the [literature review](literature-mechanisms.md), which replays more rules on
the same records: the streak uses N = 6 (40 of 157 runs saved, no adoption or run with
Δ ≥ 0.001 lost) instead of N = 8 and rotates to a new topic instead of stopping, and the reserved new-family slot is rejected because
follow-ups beat new families (median Δ −0.0010 vs −0.0029, Mann-Whitney p = 0.0008). The
activation evidence gate stays, now checked after epoch 0.

## Not transferable

- Peer review, rebuttal experiments and manuscript drafting: we have no paper to write, and
  the fixed macro-AUC rule is the only acceptance signal.
- Choosing among several Good ideas on the full benchmark: adoption here is sequential
  against one incumbent.
- Post-adoption ablation loops: adoptions here are single mechanisms and results use one run
  per change, so no confirmation or ablation runs follow an adoption.
- Novelty scoring against the literature: campaigns are bounded to one topic, and proposals
  already name their closest precedents.
