# Site-generalization campaign

**Ran 2026-09-29 to 2026-09-30, stopped by the user.** It searched for training-time mechanisms
that close the unseen-site gap (about 0.01 macro AUC) on the sites-out split, judged by the
held-out-site evaluation. Outcome: L2-SP weight anchoring was the only mechanism that reproducibly
helped held-out sites; it was adopted on main on 2026-09-30 together with removing the inert
scanner confusion head ([result](../../../docs/validation/results/validation-and-ensembles/2026-09-30-rn18-l2sp-no-confusion.md)).
The [mechanism audit](mechanism-audit.md) holds the conclusions.

| Read when | Document |
|---|---|
| Checking the objective, recipe, held-out-site evaluation, budget, fixed and permitted mechanisms, training-site labels and evidence requirements | [Contract](contract.md) |
| Checking why sites, the measured unseen-site cost, the held-out cohort's make-up and the scanner-removal precedents | [Prior work](prior-work.md) |
| Checking the sealed recipe, reference, evaluation, budget, scope and role instructions | [Campaign definition](campaign.yaml) |
| Reading seeds 1–5 in run order (ArcFace prototypes first by user choice, then by expected potential: L2-SP anchoring, cross-site Fourier amplitude swap, cross-site MixStyle, class-conditional CORAL) | [Seeds 1](seed-ideas.yaml) |
| Reading seeds 6–8 (augmentation consistency, site-cluster confusion, GroupDRO over sites) | [Seeds 2](seed-ideas-2.yaml) |
| Reading the ArcFace retake on each label's attention-pooled feature, forced on the fly after the first ArcFace run's diagnosis | [ArcFace retake](arcface-retake-ideas.yaml) |
| Reading the forced ablation that removes the scanner confusion head, to measure it on unseen sites | [Confusion ablation](confusion-ablation-ideas.yaml) |
| Reading the scanner-subspace projection seeds (angular-margin scanner-identity embedding whose subspace is fully erased from the pooling tokens), on top of the ArcFace incumbent and substituting it | [Scanner subspace](scanner-subspace-ideas.yaml) |
| Reading the closed-form LEACE token-erasure retake, forced by Claude under the user's delegation after the learned scanner subspace erased dominant directions instead of scanner identity | [LEACE erasure](leace-erasure-ideas.yaml) |
| Reading the same-seed retake of channel-selective acquisition normalization against the fresh-seed incumbent, forced by Claude after the user-ordered reference replacement | [Acquisition retake](acquisition-reseed-ideas.yaml) |
| Reading the two simplification ablations at the fresh seed (per-label ArcFace without L2-SP; no scanner confusion head), forced on the user's instruction | [Simplification](simplification-ideas.yaml) |
| Reading the leave-one-out removal of per-label ArcFace at the fresh seed, forced on the user's instruction | [ArcFace removal](arcface-removal-ideas.yaml) |
| Reading LP-FT (probe epoch with the pretrained backbone frozen, then full fine-tuning), the second of the two final runs the user limited the campaign to on 2026-09-30 | [LP-FT](lp-ft-ideas.yaml) |
| Reading the re-added mixture-contrast augmentation, the third run of the final list (ports the reviewed build whose training was cancelled) | [Mixture contrast](mixture-contrast-final-ideas.yaml) |
| Reading recipe B, site-cluster unlearning instead of the removal family (no L2-SP, no ArcFace, no scanner-appearance augmentations), the fourth run of the final list | [Unlearning only](unlearning-only-ideas.yaml) |
| Checking whether each run's mechanism actually did what it was meant to do, the clashes between removing and learning signal, and the final mechanism ranking | [Mechanism audit](mechanism-audit.md) |
| Reading the post-audit ideas forced after the public-leaderboard check (a real gradient-reversal unlearning test, frozen backbone BatchNorm, LP-FT on L2-SP alone) | [Post-audit ideas](post-audit-ideas.yaml) |
| Reading the bounded unlearning retake (its own scanner head with a KL term toward the scanner prior at weight 1.0, under site_generalization) forced after the gradient-reversal adversary collapsed; an earlier version that changed a trainer field was blocked by the configuration policy | [Bounded unlearning](bounded-unlearning-ideas.yaml) |
| Checking results | [Findings](findings.md) |

## Setup

- **Branch and worktree:** `sites-out` in `../knee-abnormality-sites-out`. The campaign ran from
  there, because its evaluation code and the sites-out splits are not on main.
- **Reference:** `sites-out-control` (seen-site ×4 0.929085, held-out-site ×4 0.899579).
- **Budget:** range 0.5–1.5 of the reference's 916 s training time.
- **No topic rotation:** `rotate_after: 1000` (user, 2026-09-29), so the researcher keeps refilling
  this topic until the user stops the campaign.
