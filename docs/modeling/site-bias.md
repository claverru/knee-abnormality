# Bias-free training

**Status, 2026-09-27:** the scanner confusion head is the incumbent of the
[auxiliary-signals campaign](../../autoresearch/campaigns/auxiliary-signals/INDEX.md), adopted by the user
on public evidence (0.948 against 0.947 for its paired control; decision
`user-adopt-20260926T191359Z-da4f7994`). The per-site bias on top of it, the strongest candidate
by the site-signature scorecard, scored 0.947 public (S69): level with the control and 0.001 below
the incumbent, all within one rounding step. No bias-removal mechanism has yet produced a gain the
leaderboard can resolve. The scanner-model confusion target, the purest site signature, scored
0.944 (S70) and the stronger head 0.943 (S71): stronger site removal costs more on the test than it
removes, and the public ranking follows each variant's disease cost on the major vendors. Post-hoc
centering per scanner was measured and rejected.

## Premise

The hidden-test labels are assigned from the images under standardized criteria, independently of
the reports (host reply, discussion 733826;
[criteria](../../docs/metadata/official-label-criteria.md#labels-versus-report-text-discussion-733826-host-reply-2026-08-08)).
The training labels are mostly extracted from reports. Anything that shifts a report's labels
without changing the image is a bias: the local folds reward a model that learns it, and the
leaderboard does not. Local validation therefore understates any mechanism that removes such a
bias, and the public board is the judge.

## Evidence for site bias

- **One site reports more findings.** The four minor vendors are one English-reporting site whose
  labels run ×1.1–×3.6 higher on ten of twelve targets (Synovitis ×3.56). Its reports are longer
  (median 1,317 characters against 968) and its Synovitis labels match the literal word in 179 of
  181 reports: reporting completeness, not a sicker population
  ([vendor associations](../../docs/metadata/vendor-label-associations.md)).
- **Site offsets hurt on gold.** Adding header site information to out-of-fold logits (offset
  stacker, folds 0–2, 2,666 studies) gained locally but lost on the 35 gold studies: protocol
  signature +0.0009 [95% +0.0005, +0.0014], gold −0.006; site (vendor, model, field strength,
  software, coil) +0.0005, gold −0.006. Gold n is small; the sign is consistent across variants.
- **The local split rewards it.** `all_5fold_scanner_v1` is stratified, not grouped, by scanner,
  so validation sites and their conventions are all seen in training.

## What the headers provide

Vendor and model survive de-identification in train and test; field strength is missing in 238
training studies and one of three sample test studies. Age, dates, institution and private tags
are stripped, and UIDs are regenerated at random: no leak and no patient overlap. The 8 scanner
groups of the split protocol (Siemens, Philips, GE × 1.5T, 3T, unknown; other vendors pooled as
minor) and 51 scanner models are available per study from
`outputs/analysis/sharing-disagreement-20260914/study_headers.parquet`.

## Runs

All fold 3, seed 2617320043. Local Δ is ×4 macro AUC against the run's campaign reference: the
first three against `half-stack-control` (0.905205), the rest against the adopted confusion head
(0.904442). Head metrics are epoch-mean training accuracy and loss against the majority rate and
the frequency entropy of the head's target (0.263 and 1.851 for the 8 groups; 0.125 and 3.160 for
the 31 models).

| Run | Mechanism | Local Δ ×4 [80%] | Head or mechanism at the end | Public |
|---|---|---|---|---|
| `scanner-group-adversarial-head` | Gradient reversal on the pooled tokens, weight 0.1 | −0.00604 [−0.0086, −0.0035] | 0.263, loss 1.90 | not submitted |
| `scanner-adversarial-decoupled-weight` | Reversal, head loss at full strength | −0.00419 [−0.0065, −0.0020] | 0.259, loss 1.93 | not submitted |
| `scanner-confusion-detached-head` | Detached head + bounded KL confusion | −0.00076 [−0.0027, +0.0011] | 0.314, loss 1.724 | **0.948 (S67), adopted** |
| `scanner-confusion-stronger-head` | Two-layer head, doubled learning rate, warm-up | −0.00330 [−0.0053, −0.0014] | 0.310, KL 0.05 (2–5× the parent's) | **0.943 (S71)** |
| `scanner-confusion-backbone-features` | Head before the embeddings and `mil_norm` | invalid; hand comparison −0.0016 [95% −0.0045, +0.0012] | 0.324 already in epoch 0 | not submitted |
| `scanner-model-confusion-target` | Confusion over 31 scanner models | −0.00134 [−0.0035, +0.0007] | 0.172 vs 0.125, loss 2.895 vs 3.160 | **0.944 (S70)** |
| `site-bias-product-of-experts` | 8 × 12 training-only logit bias, dropped at inference | **+0.00040 [−0.0011, +0.0019]** | largest biases Synovitis 0.133, Effusion 0.121 | 0.947 (S69) |
| `model-bias-product-of-experts` | 31 × 12 training-only logit bias per scanner model | −0.00279 [−0.0046, −0.0010] | signature ρ −0.26, major-vendor Δ −0.0027 | not submitted |
| `half-stack-control` | Reference | — | — | 0.947 (S68) |

- **Reversal failed twice.** Both heads stayed above the frequency entropy all run: reversal
  maximizes an unbounded cross-entropy, so the backbone bent features to make the head
  confidently wrong. Scaling the head's loss does not slow its learning under AdamW, which
  normalizes each parameter's update, so the retry changed nothing.
- **The bounded confusion loss works.** The head trains on detached tokens with plain
  cross-entropy; the backbone gets `0.1 × ramp × KL(frequencies ‖ head prediction)`, minimized
  when the head can do no better than the frequencies. Disease AUC tracked the control.
- **The scanner is faint in the pooled tokens and clear before normalization.** The pooled-token
  head reached 0.314 after ten epochs; the head before the embeddings and `mil_norm` reached 0.324
  in its first.
- **Stronger pressure removes more of the site and more of everything else.** The stronger head
  erased scanner information (its accuracy fell mid-run) and lost broadly: Fracture −0.016, PF OA
  −0.011, Lateral OA −0.008.
- **The public gain is one rounding step.** S67 against S68 is +0.001; the fold itself moved the
  same recipe from 0.943 (fold 2) to 0.947.

## Site-signature scorecard

Per-label AUC hides the effect: the minor site is 50 of 872 fold-3 validation studies, so moving
its predictions barely changes a label's AUC. The incumbent's largest per-label losses (Medial OA
−0.0086, Synovitis −0.0063) fit the site-prior story only partly, since ACL, the second most
inflated label, gained. The direct test is how predictions shift **by site**.

For each variant against `half-stack-control` (same fold and seed; scripts in the 2026-09-27
session scratchpad):

- **Signature ρ:** Spearman correlation across the 12 labels between the label's minor-site
  inflation ratio (log) and the minor-site logit shift (mean shift of minor-vendor studies minus
  the others). Negative means the shift tracks the inflation, which is what removing a site prior
  does.
- **Synovitis and Effusion shifts:** the most inflated label should drop; Effusion, the one label
  less common at that site (×0.93), should rise.
- **Major-vendor local Δ:** macro AUC on the 822 major-vendor studies only, where the known
  inflated site can neither reward nor punish the change: disease signal kept. 80% report-group
  bootstrap, 1,000 draws.

| Variant | Signature ρ | Synovitis | Effusion | Local Δ, all | Local Δ, major vendors [80%] | Gold Δ (n=11) |
|---|---|---|---|---|---|---|
| Reversal (seed 6) | +0.00 | −0.31 | −0.17 | −0.0060 | −0.0052 [−0.0078, −0.0024] | +0.0002 |
| Reversal retry | −0.18 | −0.30 | −0.20 | −0.0042 | −0.0028 [−0.0051, −0.0005] | −0.0067 |
| Confusion (incumbent, S67) | −0.36 | −0.27 | +0.07 | −0.0008 | +0.0002 [−0.0020, +0.0023] | −0.0118 |
| Stronger head | −0.57 | −0.37 | +0.19 | −0.0041 | −0.0030 [−0.0057, −0.0004] | +0.0023 |
| Backbone features | −0.37 | −0.25 | +0.17 | −0.0024 | −0.0017 [−0.0038, +0.0001] | +0.0030 |
| Scanner-model target | **−0.59** | −0.28 | +0.12 | −0.0021 | −0.0013 [−0.0034, +0.0005] | −0.0056 |
| **Per-site bias (S69)** | −0.52 | **−0.35** | **+0.24** | **−0.0004** | **+0.0005** [−0.0017, +0.0024] | −0.0002 |
| Per-model bias | −0.26 | −0.30 | +0.25 | −0.0036 | −0.0027 [−0.0049, −0.0006] | +0.0056 |

- **Reversal removed no site prior.** It pushed every label down for the minor site, Effusion
  included (ρ ≈ 0): damage, not debiasing.
- **Every confusion variant removes a site prior** (ρ −0.36 to −0.59, Effusion reversed). The
  stronger ones pay with disease signal on the major vendors (−0.0013 to −0.0030).
- **The per-site bias is the best inference candidate on both axes:** one of the strongest
  signatures, the largest Synovitis drop and Effusion reversal, and no disease cost. It sits on top
  of the confusion head, so it combines both mechanisms.
- **The incumbent is the safe second** (no disease cost, weaker signature); **the scanner-model
  target** has the purest signature but costs about 0.0013 of disease signal, so it is the next
  public check after the per-site bias.
- **Finer bias resolution hurts.** The 31-model bias weakened the signature (ρ −0.26) and lost
  disease signal on the major vendors; rare models likely absorb real disease differences.
- **Limits:** the signature rests on 50 minor-site studies and 12 labels; the major-vendor subset
  still carries any conventions among the big vendors' sites; 11 gold studies are noise.
- **The public board did not confirm the ranking.** S69, the top scorecard candidate, scored 0.947
  against 0.948 for the incumbent (S67) and 0.947 for the control (S68). A prediction shift that
  tracks the site's label inflation is not, at this resolution, a measurable leaderboard gain.
- **Disease cost on the major vendors predicts the public loss.** The scanner-model target, with
  the purest signature (ρ −0.59) and −0.0013 on the major vendors, scored 0.944 (S70): −0.004
  against the incumbent, harsher than its local −0.0013. Removing site information at model
  resolution destroys image signal the test labels need. The stronger head (−0.0030 on the major
  vendors) scored 0.943 (S71), the lowest of the line: public order S67 0.948 > S68, S69 0.947 >
  S70 0.944 > S71 0.943 matches the major-vendor Δ order +0.0002, +0.0005 > −0.0013 > −0.0030.

## Post-hoc centering, rejected

Shifting each label's logit by its scanner group's mean over the evaluated set
(`logit − mean_group + mean_all`, label-free, so it could run on the test set) was measured on
the saved out-of-fold predictions of folds 0–3:

| Centering | Local Δ macro, per fold | Gold Δ (46 studies, pooled) |
|---|---|---|
| Full | −0.0075 to −0.0111 | −0.0096 |
| Half | −0.0016 to −0.0031 | −0.0003 |

Gold agrees with local: the per-site differences in the model's outputs are mostly real case mix.
Adding site offsets (stacker, gold −0.006) and removing them both hurt on gold, so the useful
changes happen inside training, not through per-site output levels.

## Ideas not run

- **Completeness bias.** A 5 × 12 training-only logit bias per completeness quintile, a
  label-noise model for findings a report leaves unmentioned. Blocked twice at review: first
  because report length is outside the campaign's permitted data, then, rebuilt on the v3
  not-mentioned answer mass (`answer-completeness-bias`), because its logit offset changes the
  fixed disease loss, although the per-site bias with the same exception was approved. Running it
  needs the contract to authorize training-only logit offsets explicitly.
- **Token harmonization** (`scanner-token-harmonization`). ComBat-style per-group location and
  scale adjustment of the tokens before `mil_norm`, applied at inference where the scanner group is
  known. Blocked because the campaign's validation predictions come from the exported model's
  prediction path, which reads images only; it needs prediction and inference loader changes
  outside the campaign's editable files.
- **Any idea that edits `modeling/model.py`** ends invalid in this campaign, because that file is
  in the hashed preprocessing identity and the sealed policy allows no identity differences.
  Allowing `preprocessing_identity` needs a user-approved amendment of the seal.

## Which variables to remove

| Variable | Why it biases labels | Tool | Status |
|---|---|---|---|
| Site (scanner group, model) | Reporting convention per site | Confusion on features; per-site logit bias | Confusion adopted; per-site bias S69; model target measured |
| Report completeness | Silence becomes a negative label | Training-only logit bias | Blocked by the contract |
| Report language | Reaches the image only through site | — | Dropped: duplicates site |
| Patient sex, study indication | Real case mix | — | Keep |
| Field strength, vendor contrast | Physics that shapes the image | Conditioning input | Researcher idea `acquisition-physics-feature-modulation` |

A variable the image cannot reveal can still bias training as label noise: a training-only bias
term absorbs it without needing the image to see it. Confusion only helps for variables the
features can carry.

## Judging

Local validation keeps the report-derived conventions, so a bias-removing mechanism is expected to
lose a little locally. Judge a candidate by the site-signature scorecard (the shift tracks label
inflation, Effusion moves the other way, no loss on the major vendors) and then by a public
submission paired with its reference at the same fold and seed. Public differences of 0.001 are
one rounding step, so a line is established by repeated agreement, not one pair.

## Open questions

- S69 did not beat S67 (0.947 against 0.948), S70 and S71 lost (0.944, 0.943): is the incumbent's light
  confusion itself a gain, or one rounding step? Answering it needs a second fold or seed.
- Does removing the scanner before normalization help if its disease cost can be reduced?
- Does the public gain survive on another fold?
