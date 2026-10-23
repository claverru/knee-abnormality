# Prior work

Where the campaign comes from, the measured precedents and the closed directions.

- **Why sites:** the organizers list 19 contributing institutions and, separately, 3 additional
  sites (Mongolia, Australia, Italy). No Italian or Mongolian report exists in training, and the
  training reports hold 9 languages where RSNA describes a dozen
  ([sites and test composition](../../../docs/metadata/sites-and-test-composition.md)).
- **Measured cost:** two runs of the recipe without the five held-out sites, on Kaggle and
  locally, put the unseen-site cost at −0.0092 and −0.0099 macro AUC. The 23% smaller training
  set cost −0.0009 and −0.0016 on seen sites
  ([sites-out diagnostic](../../../docs/validation/results/validation-and-ensembles/2026-09-29-sites-out-diagnostic.md)).
- **Held-out cohort:**
  - Bulgarian (Sofia, Philips Ingenia 3T, 220 studies);
  - German (Würzburg, Siemens Avanto fit, 262);
  - Greek (Heraklion, Siemens SonataVision and Prisma, 321);
  - Dutch (AZ Delta, Siemens Vida and Sola, 152);
  - French (Marrakech, Siemens Aera, 81).

  Four Siemens sites and one Philips, all European or Mediterranean; no GE and no minor vendor. A
  mechanism that helps only these sites may not transfer.
- **Reporting conventions:** part of the measured cost may be the held-out sites' reporting
  conventions, which the test's image-derived labels do not share.
- **Scanner signal in the features:** the scanner confusion head's detached classifier reached
  0.314 accuracy on the 8 scanner groups (majority 0.263) from the pooled tokens, and 0.324 before
  normalization ([bias-free training](../../../docs/modeling/site-bias.md)). A competitor reports features that
  predict the manufacturer at 82.8% (chance 37.9%).

## Noise yardstick

A second run of the control's exact recipe (Kaggle, same seed, different hardware) moved,
against the control:
- **Held-out ×4 macro AUC:** −0.0012.
- **Seen-site ×4:** +0.0007.
- **Held-out sites:** Bulgarian +0.0083, German +0.0009, Greek −0.0011, Dutch +0.0012,
  French −0.0078.
- **Held-out per label:** from −0.0183 (MCL) to +0.0075 (ACL).

A fresh-seed rerun of the adopted ArcFace incumbent (identical source and config,
`site-incumbent-reseed-20260930T082402Z`, seed 3843979352) moved, against that run:
**held-out −0.0022** and **seen-site −0.0033**. Seed noise is therefore as large as the
adoption threshold and every effect seen so far, and an adopted run's scores carry selection
bias: all 14 candidates built on the adopted run lost seen-site AUC (median −0.0022). The
rerun replaced the adopted run as the incumbent reference on 2026-09-30 (user decision).

Leave-one-out validation at the fresh seed (2026-09-30):
- **L2-SP is real:** removing it cost −0.0014 held-out [80% −0.0022, −0.0005], seen +0.0002,
  reproducing its adoption gain (+0.0015) at a different seed.
- **The scanner confusion head does nothing measurable:** removing it gave held-out −0.0007
  [−0.0020, +0.0006] and seen +0.0009 (−0.0001 and −0.0012 at the old seed). It was removed from
  the incumbent as a simplification by user decision.
- **Per-label ArcFace does nothing measurable on unseen sites:** removing it gave held-out
  +0.0002 [−0.0014, +0.0019] and seen +0.0021, so its +0.0018 adoption gain did not reproduce.
  It was removed as a simplification by user decision; the incumbent is L2-SP alone.

Public leaderboard check (2026-09-30, ×4, attention temperature 0.9, seed 1533697213, noise about
±0.003): full-data recipe S82 0.943; sites-out control S83 0.941; + L2-SP S84 0.943; + L2-SP and
per-label ArcFace S85 0.940. The site cost on the public test is small (about the data loss), L2-SP
moves the same way as on held-out sites, and ArcFace costs on top of L2-SP, consistent with the
[mechanism audit](mechanism-audit.md).

Post-audit runs (2026-09-30, on L2-SP alone, seed 3843979352):
- **Gradient-reversal scanner adversary** (weight 1.0): collapsed; the reversed gradient reached
  1.8× the disease gradient, the adversary became anti-predictive (0.049), held-out −0.065.
  Unbounded reversed cross-entropy is closed.
- **Bounded unlearning** (`site-scanner-unlearning-kl`, KL toward the scanner prior, weight 1.0):
  worked; the scanner head stayed at 0.22 against the 0.208 majority for all 10 epochs instead of
  rising to 0.33, at a cost of 0.05–0.18% of the loss. Held-out −0.0003 [−0.0016, +0.0011], seen
  −0.0030. Actively unlearning scanner identity does not help unseen sites and costs seen sites.
- **Frozen backbone BatchNorm**: held-out −0.0086, seen −0.0106. BatchNorm has to adapt to the MRI
  input normalization; only the conv weights benefit from anchoring.
- **LP-FT on L2-SP alone**: held-out +0.0005 [−0.0007, +0.0018], seen −0.0012; its earlier +0.0016
  did not hold.

Activation first: most rejected ideas never did what they were meant to (see the audit). Every
new idea must name a quantitative activation criterion that its own logs can confirm, such as an
adversary's accuracy falling toward the majority rate or a penalty being a meaningful share of
the loss.

Final list (2026-09-30), after which the user stopped the campaign:
- **LP-FT** (on L2-SP + ArcFace): held-out +0.0016 [+0.00003, +0.0032], seen −0.0013 (guard
  missed by 0.0003). The one open lead; untested on L2-SP alone.
- **Mixture-contrast augmentation** (on L2-SP alone): held-out −0.0015, seen +0.0004.
- **Unlearning instead of removal** (`site-unlearning-only`: site-cluster confusion, no L2-SP,
  no ArcFace, no GIN, contrast mixing or feature-statistics augmentation): held-out −0.0027
  [−0.0048, −0.0007], seen −0.0017. Unlearning alone is worse on unseen sites than the removal
  family.

Each decided run's `held_out_sites` (in `recent_results`) gives its per-site AUC against its
reference. Treat a site or label move within these ranges as noise; only the pooled held-out
number with its interval decides.

## Precedent table

Measured runs outside this campaign, fold 3 of `all_5fold_scanner_v1` against
`half-stack-control`. They report ×4 macro AUC on seen sites; none was measured on unseen sites.

| Mechanism | Setting | Δ ×4 | Public | Source |
|---|---|---|---|---|
| Gradient reversal | scanner-group head, weight 0.1 | −0.0060 | – | [auxiliary-signals campaign](../../../docs/validation/results/campaigns-and-queues/2026-09-27-auxiliary-signals-campaign.md) |
| Gradient reversal | head at full strength | −0.0042 | – | same |
| Scanner confusion (bounded KL, detached head) | 8 scanner groups, weight 0.1 | −0.0008 | 0.948 against 0.947 (adopted) | same |
| Stronger confusion head | two-layer head, doubled rate | −0.0033 | 0.943 | same |
| Confusion over scanner models | 31 models | −0.0013 | 0.944 | same |
| Confusion before normalization | backbone features | invalid (hand −0.0016) | – | same |
| Per-site logit bias | 8 × 12, training only | +0.0004 | 0.947 | same |
| Per-model logit bias | 31 × 12, training only | −0.0028 | – | same |
| Post-hoc per-site centering | output logits | full −0.0075 to −0.0111; half −0.0016 to −0.0031 | – | [bias-free training](../../../docs/modeling/site-bias.md) |
| DSU feature statistics | stride 8, generic | adopted on ReXNet; in the recipe | – | [augmentation campaign](../../../docs/validation/results/campaigns-and-queues/2026-09-26-augmentation-campaign.md) |

## Closed directions

- **Gradient reversal:** it maximizes an unbounded cross-entropy and bent features to fool the
  head.
- **Stronger or finer scanner removal:** its public cost followed the disease cost on the major
  vendors.
- **Output-level site offsets:** added or removed, they hurt on gold.
- **Conditioning on vendor or site:** useless for an unseen site.
