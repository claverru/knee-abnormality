# Mechanism activation audit

Audit of 2026-09-30, after the campaign stopped: for every decided run, whether the mechanism
actually did during training what it was meant to do, judged from its own logged diagnostics,
code diff and checkpoints. Held-out Δ is the held-out-site ×4 macro AUC delta against the
run's parent; seed noise between two runs of one recipe is about 0.002–0.003. Analysis scripts
are in the session scratchpad and are not kept.

Signal classes: **learns** (adds supervision), **removes** (augmentation, erasure, anchoring,
normalization, compactness or invariance pressure), **unlearns** (adversarial: detect, then
suppress), **reweights**.

## Per-experiment verdicts

| Run | Class | Did it work? | Evidence | Held-out Δ |
|---|---|---|---|---:|
| Base scanner-confusion head (control and incumbents) | unlearns | **No, inert** | Head reaches 0.33 on 8 groups vs 0.21 majority and keeps rising after the ramp saturates; its KL term is about 0.1% of the training loss | n/a |
| site-confusion-retarget | unlearns | **No, inert** | Site head +0.09 above majority and rising; KL at most 0.004 | +0.0016 |
| site-unlearning-only | unlearns | **No, inert** | Site head learns slightly faster without GIN, contrast mixing and feature statistics, never suppressed | −0.0027 |
| site-no-scanner-confusion (both seeds) | ablation | Yes | Head absent, nothing else changed | −0.0001 / −0.0007 |
| site-l2-sp-anchor | removes (weights) | **Yes** | Backbone drift −42 to −60% in stem and layer1, −9 to −43% later; BatchNorm statistics not anchored | +0.0015 |
| site-arcface-without-l2-sp | ablation | Yes | Anchor buffers gone, weight decay 0.01 restored, drift back to control levels | −0.0014 |
| site-lp-ft | removes (schedule) | **Yes** | Epoch 0 frozen (drift 0, BatchNorm frozen), backbone learning-rate ramp in epoch 1, final drift 5–14% below parent | +0.0016 |
| site-pretrained-feature-direction-anchor | removes (features) | No | Penalty mostly measured the BatchNorm-statistics mismatch; weight drift rose against the parent | −0.0019 |
| site-pretrained-canonical-feature-anchor | removes (features) | No, too weak | Correct construction, penalty 0.014% of the loss, drift unchanged | −0.0011 |
| site-arcface-prototypes | removes (compactness) | No | Loss fell but features moved away from their prototypes; cross-site spread flat | −0.0000 |
| site-arcface-per-label-pooled (both seeds) | removes (compactness) | **No** | Loss won by relative separation, not compaction; cross-site spread about doubled; raised backbone drift against L2-SP (stage 4 0.131 vs 0.112) | +0.0018 / reseed |
| site-arcface-balanced-site-states | reweights | Partial | Site contributions evened out; dispersion only slightly lower than the incumbent's | −0.0004 |
| site-arcface-antipodal-prototypes | removes (compactness) | Partial | Constraint exact, the only variant moving features toward prototypes, but weakly (cosine 0.23) | −0.0001 |
| site-l2-sp-without-arcface | ablation | Yes | ArcFace fully absent, augmentation activity identical epoch by epoch | +0.0002 |
| site-scanner-subspace-projection | removes (erasure) | No | Removed about 63% of the token norm while scanner identity stayed decodable (probe 0.187 vs 0.129 majority) | −0.0005 |
| site-scanner-subspace-instead-of-arcface | removes (erasure) | No | Same: about 64% of the token norm removed, identity still decodable | −0.0007 |
| site-scanner-leace-token-erasure | removes (erasure) | **Yes, costly** | Full-rank eraser, 23–29% of the token norm removed, downstream scanner head 0.32 → 0.24; eraser overlaps the disease-head rows (cosine 0.35–0.45) | −0.0089 |
| site-conditional-coral | removes (invariance) | No, too weak | Penalty about 0.03% of the loss; alignment loss rose | +0.0001 |
| site-group-dro | reweights | Yes, over-concentrated | Effective groups 12 → 3; one cluster took 53% of the weight | −0.0069 |
| site-dro-fresh-bounded-risk | reweights | Yes | Ratios at their bounds, worst groups doubled | −0.0016 |
| site-dro-entropy-corrected-risk | reweights | Yes, mild | Ratios 0.84–1.23 at the end, near population shares | −0.0007 |
| site-fourier-amplitude-swap | removes (appearance) | Yes | Cross-site swap at 50%, strongest perturbation (0.13 per slice) | +0.0007 |
| site-cross-mixstyle | removes (feature style) | Yes | Cross-site mixing at 50% in stem and layer1 | −0.0010 |
| site-augmix-consistency | removes (invariance) | Partial | Consistency term about 0.08% of the loss and not falling | +0.0003 |
| site-fourier-matched-shape-exchange | removes (appearance) | Partial | Fired at 50%, but clipping broke the spectral constraint | +0.0000 |
| site-fourier-feasible-residual-exchange | removes (appearance) | Yes, weaker | No clipping, but the change shrank to 0.031 per triplet | −0.0016 |
| site-acquisition-half-channel-normalization | removes (normalization) | Yes (delivery) | Normalization changed the outputs by 0.047 at every epoch; invariance not measured | +0.0010 |
| site-acquisition-channel-selective-normalization (both seeds) | removes, learned restore | Partial | Coefficients stayed at 0.96–0.98: effectively fixed normalization | +0.0015 / −0.0001 |
| site-acquisition-image-dependent-restoration | removes, learned restore | No, inert | Gates stayed at their 0.05 initialization | +0.0005 |
| site-acquisition-mixture-contrast-final | removes (appearance) | Partial | Fired at 30%, but about 13× weaker than the GIN it replaced, so in effect it removed GIN | −0.0015 |
| Base appearance augmentations (control) | removes (appearance) | Yes | GIN, contrast mixing, feature statistics, resolution and smoothing fire at their configured rates; intensity and noise are not logged | n/a |

## Findings across runs

- **Unlearning was never tested.** Every confusion head in the campaign detected its target
  weakly and was never suppressed, because the adversarial term was about 0.1% of the loss.
  The appearance augmentations are not the main cause: without them the site head learned only
  slightly faster. Removal versus unlearning therefore remains open; site-unlearning-only
  measured removal switched off, not unlearning switched on.
- **Removal mechanisms clash with learning when they act on the features the disease head
  reads.** LEACE removed scanner information together with disease signal (cosine 0.35–0.45 with
  the disease-head rows); the learned subspaces removed about two thirds of the token norm
  without removing scanner identity; per-label ArcFace and the feature-direction anchor pushed
  the backbone away from the pretrained weights L2-SP anchors.
- **Removal that acts before the learned features helps.** L2-SP (weights) reproduced; removing
  GIN (mixture-contrast-final, in effect) cost −0.0015; recipe B, with GIN, contrast mixing,
  feature statistics and L2-SP removed, cost −0.0027.
- **L2-SP protects the conv weights, not the inference features.** BatchNorm statistics are not
  anchored, and the features computed with each run's own statistics are as far from the
  pretrained ones in every run.

## Premise tests (2026-09-30)

- **The unseen-site cost is between-site label rates, not image features.** On the 224 fold-0
  held-out studies scored by a model that saw their sites and one that did not, the per-site AUCs
  average 0.929 in both; same-site pairs carry +0.0005 of the 0.0115 pooled cost, cross-site pairs
  +0.0110 [+0.004, +0.018]. The model that saw a site learned its label rates (Effusion 0.42 at the
  German and Greek sites vs 0.26 in training; Bulgarian Synovitis 0.01); Effusion and Synovitis carry
  about 60% of the cost and replicate in the Kaggle-trained twin. Label-free per-site, per-vendor or
  per-scanner score normalization lowers AUC for both models (about −0.020 in the realistic mixed
  case); borrowing the seen model's per-site score level recovers 60–70% of the cost.
- **Scanner and site identity stay fully decodable in every model, and the amount does not track
  unseen-site AUC.** Across 17 checkpoints, probes decode scanner group at 0.92–0.98 balanced
  accuracy from backbone stages, 0.87–0.91 from the token mean and 0.77–0.84 from the pooled
  features (majority 0.22); the same holds for 24 scanner models and 12 site clusters, and a frozen
  backbone still gives 0.96 at layer 2, so it arrives with the images. No debiasing run reduced it;
  gradient reversal raised it. Correlations with held-out AUC: 4 of 81 tests significant, as chance
  would give.
- **Consequence:** image-feature debiasing targets something that is not the cause. Features
  already rank as well within an unseen site; what is missing there is the site's label base rates
  (case mix and probably reporting conventions), which invariance cannot supply. Analysis scripts
  and outputs: `outputs/analysis/s/site-probes-20260930/`.

## Manual round after the campaign (2026-09-30)

Run by Claude outside the harness, same comparison tool and cohorts, seed 3843979352, runs under
`outputs/training/site-generalization-manual/` (95% paired bootstrap intervals).

- **L2-SP on independent held-out sites** (split `all_5fold_scanner_v2_sites_out_b`: Spanish,
  Turkish and Croatian sites held out, 1,632 studies): held-out +0.0009 [−0.00002, +0.0020], seen
  −0.0010 [−0.0023, +0.0003].
- **L2-SP × conditional unlearning** on the original split, one code copy, only the two switches
  differing. Conditional unlearning: a label-only scanner head q(s|y), a joint head r(s|z,y) on
  detached features, and KL(q || r on live features) at weight 1.0, so the features are pushed to
  carry no scanner information beyond the labels. Against the no-L2-SP, no-unlearning control:

  | Cell | Held-out Δ | Seen Δ |
  |---|---|---|
  | L2-SP | +0.0023 [+0.0012, +0.0036] | +0.0003 [−0.0009, +0.0015] |
  | Conditional unlearning | +0.0014 [−0.0008, +0.0036] | −0.0027 [−0.0051, −0.0003] |
  | Both | +0.0032 [+0.0010, +0.0056] | −0.0028 [−0.0052, −0.0003] |

  The effects are roughly additive (no clash). The unlearning gap (joint minus label-only head
  accuracy) grew to about 0.04 with or without L2-SP, so L2-SP is not what limits it.
- **Test-time label-shift correction** (EM prior re-estimation per site, scanner or vendor group,
  offline): no variant helps on the 1,036 held-out studies; the unseen model's scores carry no
  signal about a new site's label rates, while an oracle using true rates would recover +0.012.

Conclusion: L2-SP helps unseen sites at no seen-site cost (four consistent measurements:
+0.0015, −0.0014 when removed, +0.0009 on split B, +0.0023 here). Conditional unlearning trades
about −0.003 on seen sites for an uncertain +0.001 on unseen sites, with or without L2-SP.

## Mechanism ranking

1. **L2-SP weight anchoring**: worked, reproduced (+0.0015 adopted, −0.0014 removed). Keep.
2. **Base appearance augmentations (GIN in particular)**: worked; removing GIN cost −0.0015.
   Keep.
3. **LP-FT**: worked mechanically and reduced drift further; +0.0016 in one run, seen-site
   −0.0013. Open lead, no clash with L2-SP.
4. **Fixed acquisition normalization**: delivered; +0.0010 once, and its closest retake gave
   −0.0001. Weak.
5. **Worked, no benefit**: cross-site MixStyle, Fourier swaps, bounded and entropy DRO, AugMix.
   Drop.
6. **Never actually active, so untested**: all unlearning (confusion heads), CORAL, canonical
   feature anchor, learned restorations. Of these, only unlearning is worth a real test, with an
   adversarial term strong enough to suppress the head.
7. **Worked and harmful**: LEACE erasure, group DRO (over-concentrated), learned scanner
   subspaces (collateral removal).
8. **Did not work and clashed with L2-SP**: per-label ArcFace, feature-direction anchor.
