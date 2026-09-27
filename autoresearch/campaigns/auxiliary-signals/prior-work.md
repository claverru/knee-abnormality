# Prior work

Where the seeds come from, the measured precedents and the closed directions.

- **Selection review, 2026-09-26:** the user asked for a campaign on the best auxiliary heads
  and inference-available inputs from existing data, since a new JevK5 extraction (about 4.5
  GPU hours) is not affordable. Seeds kept only where evidence exists: the attribute
  coefficient, loss balancing by finding area, a reliability filter on questions, removal of
  report-completeness and hedged options (the user asked for fine-grained option and question
  work; the rest of that space is the researcher's), and a scanner-group adversarial head,
  which the user kept although the local rule is expected to understate it. Lower-confidence
  ideas were left to the researcher: a plain scanner-group head
  (the user's idea, dropped as a seed because the header stacker shows the backbone already
  absorbs site), heads kept at inference feeding the disease readout, heads sharing the disease
  pooling, an acquisition-physics head and acquisition-conditioned inputs.
- **Heads design and targets:** [report-attribute heads](../../../docs/modeling/report-attribute-heads.md)
  (pooling, loss, gates, reliability weights, coefficient reasoning) and the
  [extraction](../../../docs/ongoing/report-attribute-extraction.md) (question set, audits, v3
  analysis).
- **Seed noise:** re-seeding the adopted smoothing recipe moved ×4 macro AUC by 0.0049
  ([augmentation result](../../../docs/validation/results/2026-09-26-augmentation-campaign.md#seed-noise));
  re-seeding the knee-geometry recipe moved it by −0.0063, with Contusion −0.0205, PF OA −0.0154,
  Fracture −0.0137 and MCL −0.0129. Treat single-label moves of that size in those labels as
  noise.

## Precedent table

Measured runs outside autoresearch; deltas are ×4 macro AUC against the control of their day.

| Mechanism | Setting | Run | Δ | Source |
|---|---|---|---|---|
| Attribute heads | v3 targets, weight 0.1, fold 1 | `report-attribute-heads` | +0.00075 [95% −0.0015, +0.0031]; public 0.938 vs 0.938 | [result](../../../docs/validation/results/2026-09-25-report-attribute-heads.md) |
| Attribute heads | target set 3.1 (osteophytes, Hoffa, side dropped; same-meaning options merged; extent and PF surface gated; reliability weights) | `report-attribute-heads-v31` | +0.00105 [95% −0.0012, +0.0035]; public 0.940 vs 0.938; +0.0003 vs v3 | same |
| Header stacker | protocol signature, post-hoc offset on out-of-fold logits, folds 0–2 | scratch analysis 2026-09-26 | +0.0009 [95% +0.0005, +0.0014]; gold (n=35) −0.006 | [site priors](#site-priors) |
| Header stacker | site (vendor, model, field strength, software, coil) | same | +0.0005 [95% +0.0001, +0.0009]; gold −0.006 | same |
| Header stacker | patient sex | same | +0.00003; gold −0.0015 | same |
| Header stacker | protocol choices (series count, oblique, 3D, contrast) | same | ≤ 0; gold −0.019 | same |
| Header stacker | label co-occurrence on all 12 logits | same | −0.0029 | same |

Per-label pattern of the two head runs (one seed, one fold, exploratory): MCL (+0.014) and
Contusion gained in both; Fracture and Baker's lost in both (Baker's −0.0027, then −0.0064). The
attribute loss fell 1.006 → 0.806 on 3.1 with a floor near 0.55; the worst-fitted questions were
PF and medial cartilage depth, effusion size and the medial meniscus questions.

## v3 answers: what the targets can and cannot carry

- **Reliability per question** (mean one-minus-TV between option orders over 4,407 studies, 3.1
  table): lowest medial cartilage extent 0.735, Baker's size 0.758, lateral and PF cartilage
  extent 0.776, medial cartilage depth 0.787, PF surface 0.799; highest loose bodies 0.973,
  medial extrusion 0.972, patellar instability 0.968, extensor tendons 0.966, PCL 0.955.
- **Known weak targets:** Baker's size and Hoffa invent findings; the patellar-facet leak into
  femorotibial cartilage remains; osteochondral counts bone lesions; unexplained edema is called a
  contusion. Order disagreement does not flag these systematic errors.
- **Dead options:** bone-on-bone, context "other", synovitis "possible", ACL graft.
- **Position bias:** first option up to +0.27, last up to +0.26; osteophytes mostly noise.
- **Low grades over-called:** the answers read small effusions, small Baker's cysts and
  low-grade cartilage loss that the official criteria grade negative; gold partly confirms.
- **Question groups** (question-set `group` field, 35 questions in 3.1): Study 1, ACL 1, MCL 2,
  Medial Meniscus 4, Lateral Meniscus 3, Medial OA 2, Lateral OA 2, PF OA 3, OA 1, Contusion 2,
  Fracture 1, Effusion 1, Synovitis 1, Baker's 2, Extra 9.

## Site priors

Scratch analysis of 2026-09-26 (one full header per series for all 24,386 series; offset stacker
on out-of-fold logits of `knee-geometry-merged-2`, `report-attribute-heads-v31` and
`report-attributes-control`, trained on two folds and scored on the third, 2,666 studies):

- **Available at test time:** vendor, model, field strength, software, coil, patient sex, series
  descriptions, TR/TE. Stripped: age, dates, institution, private tags. 238 training studies and
  one of three sample test studies lack TR, sex and field strength. UIDs are regenerated at
  random; no leak or patient overlap.
- **The backbone already absorbs site:** protocol and site predict Synovitis at 0.73 univariate
  AUC but add at most +0.0009 over the model's logits.
- **Gold disagrees:** every header variant scored negative on the gold studies (n=35, not
  significant). The four minor vendors are one English-reporting site whose labels are elevated
  on ten of twelve targets, with evidence for reporting style
  ([vendor associations](../../../docs/metadata/vendor-label-associations.md)); the host states
  that test labels come from images, not reports
  ([criteria](../../../docs/metadata/official-label-criteria.md#labels-versus-report-text-discussion-733826-host-reply-2026-08-08)).
  Local folds reward site priors because the split is stratified, not grouped, by scanner.

## Closed directions

- Knee side as a target: reflection augmentation contradicts it.
- Osteophyte and Hoffa questions: dropped in 3.1 as position noise and invented findings.
- Per-bone marrow questions: copied one bone's finding to the others in v1 and v2.
- Training-label masking on report silence: rejected
  ([criteria](../../../docs/metadata/official-label-criteria.md#labels-versus-report-text-discussion-733826-host-reply-2026-08-08)).
- Gold-study loss weighting and per-label losses or routing.
