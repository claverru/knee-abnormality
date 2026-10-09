# Knee-geometry prior work

Evidence map for the [knee-geometry campaign](contract.md); not a backlog. Measured facts are
marked with their source; hypotheses are marked as such.

## How the model uses side today

**Anatomical attention**, adopted 2026-09-21, +0.0023 macro AUC
([record](../../anatomical-attention.md)):

- Every token gets a signed medial position
  `medial_mm = knee_side_sign × (patient_x − knee_center_x_mm)` (`modeling/model.py`,
  `encode_batch`). `knee_side_sign` comes from the audited resolver (`laterality.py`): −1 left,
  +1 right, 0 unknown.
- The four compartment labels add `± strength × tanh(medial_mm / 20)` to their attention scores,
  + for medial and − for lateral. One learned strength per family; in `k05-clean` they ended near
  1.47 (meniscus) and 1.12 (OA).
- Reflection augmentation mirrors the knee and keeps `medial_mm` consistent. Unknown side or
  invalid geometry gives 0, so no bias. Token `geometry_valid` requires physical sampling, so a
  study with a rejected centre gets no bias.

## Study centre

- **Fit.** `estimate_reference` (`geometry.py`) solves for the point where every series' slab
  is centred; `normal_projection` is `I − nnᵀ`, so each series constrains only its own image
  plane and sagittal series contribute nothing to the centre's x. It rejects 177 of 4,407
  training studies (30 of 905 on fold 0), all for slabs placed elsewhere (patellar axials, thigh
  slabs, the second knee of bilateral studies)
  ([cohort fit](../../../docs/data-preparation/accurate-sampling-centers.md#cohort-wide-center-fit-and-plane-extents)).
- **Anatomical accuracy** (measured 2026-09-23, visual marks on 30 fold-0 studies, about ±3 mm
  marking precision, marks not kept): against the intercondylar notch and tibial plateau the
  centre is accurate medial-lateral (1.1 ± 1.3 mm, max 3), a constant offset anterior-posterior
  (plateau middle 10.6 ± 4.7 mm anterior) and off superior-inferior (joint line 12.8 ± 7.8 mm
  below, range 3–28 mm; 20 accepted studies). Robust refits of 10 rejected studies behave the
  same (M–L 0 ± 1, S–I 16.9 ± 9.1).
- **Robust centre.** Refit dropping the worst series above 10 mm, then a sagittal-centre
  fallback; out-of-range series sample locally; token geometry no longer requires physical
  sampling. Implemented 2026-09-23 behind `data.robust_study_center` and removed uncommitted the
  same day; preserved as `outputs/analysis/r/robust-center-rescore-20260923/robust_center.patch`,
  which applies cleanly to `d914079`. Its `identity.py` hunk must move into `RELIABILITY`, since
  `identity.py` is protected.
- **Robust-centre gate** (measured 2026-09-23, `image-blend` epoch 8, ×4, τ 1, inference only):
  the refit rescues all 30 rejected fold-0 studies. Robust centre plus bias +0.00010 hard
  (80% [−0.00006, +0.00026]); bias alone with centres unchanged +0.00009 (80% [+0.00003,
  +0.00015]), all from the meniscus and OA pairs. Sampling alone therefore adds about +0.00001;
  the value is the geometry every later mechanism needs
  (`outputs/analysis/r/robust-center-rescore-20260923/`).

## Midline

Measured 2026-09-23 on train (n = 4,229 with both values;
`outputs/analysis/m/midline-vs-centre-20260923/result.md`):

- `knee_center_x_mm` is the median of series centre x over all series (`laterality.py`), not the
  fitted centre. The two differ by a median 1.55 mm, p90 5.7 mm, p95 8.6 mm, max 23.7 mm; 12.5%
  of studies above 5 mm, 3.5% above 10 mm. Fold 0 is the same.
- Side-signed median −0.25 mm: scatter, not a constant offset.
- The 10 worst cases all have sagittal series whose x-centres sit 20–40 mm from the coronal and
  axial ones. The fit ignores sagittal x; the median does not.
- All 178 studies rejected by the fit have a midline.
- Inference: since the fit's x is the one measured against the notch, the midline is probably
  the value that is wrong when they disagree (medium-high confidence; the midline itself was not
  marked).

## Knee rotation

Measured 2026-09-23 from headers, 4,410 studies
(`outputs/analysis/k/knee-rotation-survey-20260923/result.md`):

- The sagittal slice normal deviates from scanner X about the superior-inferior axis by a median
  6.7°, p90 16.2°, p95 19.3°, max 40.9°; 60% of studies above 5°, 30% above 10°, 12% above 15°.
- At a posterior horn 40 mm behind the centre, that shifts X by more than 5 mm in 47% of studies
  and more than 10 mm in 14%, against the side mask's ±10 mm band.
- Several sagittal series in one study agree (median spread 0°). Fold 0 matches the rest.
- Caveat: the normal records the technologist's prescription, which some protocols align with the
  ACL rather than the condylar axis; the survey cannot separate knee rotation from protocol
  obliquity.

## Knee side and bilateral studies

- **Unknown side:** 170 of 4,410 studies (3.9%; 27 of 905 on fold 0): 107 without metadata and
  centred on the scanner origin, 33 conflicting metadata, 15 coordinates contradicting metadata,
  14 text only, 1 bilateral tag (`outputs/analysis/k/knee-side-policy-20260914/studies.parquet`).
- **Bilateral studies**
  ([contralateral knee](../../../docs/research/series-filtering-signals.md#contralateral-knee)):
  seven training studies hold two full knee protocols (Philips, `SENSE_HR_KNEE_AC`, one contiguous
  `SeriesNumber` block per knee, X sign separates the knees); `…42012883` has one stray sagittal
  series from the other side. Which knee the study labels describe is undocumented. Today 7 of
  these 8 resolve to one side for both knees (mostly the L tag on the positive-X block), their
  midline sits between or on one knee (−100 to +108 mm), and 7 of 8 have a rejected centre, so no
  bias or mask. A robust refit alone would validate them with half the tokens on the wrong side.
- Rule-based side from coordinates: centre-X sign agrees with explicit tags in 97.72% of 2,151
  studies, 99.23% with |X| > 20 mm
  ([knee-side coordinates](../../../docs/metadata/knee-side-coordinate-investigation.md#agreement-and-potential-coverage)).

## Cross-talk and the side mask

- **Inference-only masking** on `k05-clean` raised AUC against sibling-positive negatives by
  +0.008 to +0.013 but lowered it against sibling-clean negatives; macro flat (+0.0002)
  ([result](../../../docs/validation/results/2026-09-23-sibling-masking.md)).
- **Receptive field** (medial-lateral-transfer campaign): a stride-32 token 64 px from the
  midline takes about 30% of its input from the other image half, so a pooling mask does not
  remove sibling pixels from each token.
- **Trained side mask, `side-mask`** ([result](../../../docs/validation/results/2026-09-23-side-mask-and-token-attention.md#trained-side-mask-side-mask); 2026-09-23, `model.compartment_evidence: side_mask` on the
  `sibling-masking` branch, patch in
  `outputs/analysis/s/side-mask-vs-k05-clean/side_mask.patch`, applies to `d914079`): the four
  compartment labels attend only to own-side tokens plus the ±10 mm band; unknown side or invalid
  geometry stays visible to both sides. Epoch 8 selected (×1 0.90191), 1,583 s, 20.9 GB.
  - Macro ×4 0.903969 against 0.902760, **+0.0012**, 80% [−0.0004, +0.0028], misses the rule;
    soft +0.0003.
  - Families (95%): meniscus **+0.0046** [+0.0005, +0.0090], OA **−0.0040** [−0.0087, +0.0002],
    other eight +0.0017 [−0.0014, +0.0051].
  - Strata (change, 80%): Medial Meniscus sibling-clean +0.0031, sibling-positive +0.0056;
    Lateral Meniscus +0.0063, +0.0047; Medial OA −0.0038, +0.0003; Lateral OA −0.0034, −0.0093.
  - Learned strengths fell from 1.47 to 1.21 (meniscus) and 1.12 to 0.98 (OA).
  - Reading (hypothesis): trained blind, the menisci improve on both negative groups; OA is a
    knee-level degeneration whose signs on the other side are real evidence. The
    control-to-control spread on this recipe has reached 0.0034.
  - Not run: `side_mask_shared_head`, a hard tie of one classifier per family on the masked pools
    (same patch). The campaign's meniscus head and OA readout replace it.
- **Label and prediction coupling** (medial-lateral-transfer): OA pair logits correlate at 0.854
  against label correlation 0.51–0.56; meniscus 0.389 against 0.13–0.22.

## Sharing between compartments

All from the historical [medial-lateral-transfer campaign](../medial-lateral-transfer/findings.md),
without a mask and on the old evaluation (`all_5fold_v3`, Steven v4 labels), so only a rough
guide. Family deltas:

| Design | Meniscus | OA |
|---|---:|---:|
| Hard-tied meniscus head, position-rich (`anatomical-position-rich-shared-meniscus-readout`, reject) | −0.0071 | −0.0071 |
| Hard-tied meniscus head, appearance-only | −0.0075 | +0.0024 |
| Shared detector with an assignment gate | −0.0114 | −0.0067 |
| Pair contrast / 45° rotation / exchange operator | −0.0105 / −0.0091 / −0.0037 | −0.0022 / −0.0026 / −0.0027 |
| Mirror consistency | −0.0191 | −0.0101 |
| Shared local lesion evidence in the logits (`anatomical-attention-local-lesion-evidence-retry`, reject) | −0.0058 | −0.0025 |
| Forcing evidence apart (`active-gradient-attribution-exclusivity`, reject) | −0.0017 | −0.0053 |
| Shared family-level channel, private pathways kept | +0.0003 | +0.0001 |

- Every run that forced OA evidence apart lost OA; every shared readout in the public path lost
  meniscus. The one non-negative design kept every private pathway and added a shared knee-level
  part. Hypothesis: without a mask both compartment pools mixed both sides, so a tied classifier
  could not separate them.
- **`permutation-invariant-half-knee-mil`** was invalid: a bf16 dtype crash in
  `_segment_smooth_max` on the first step (`total` allocated in bf16 while autocast promotes the
  `exp()` source to fp32), so nothing trained. It took the min over both half assignments because
  side was unknown then. Diff:
  `outputs/autoresearch/medial-lateral-transfer/experiments/permutation-invariant-half-knee-mil-20260921T035443Z-6d769087/changes.diff`
  (weight 0.075, band 10 mm).
- 279 exclusive lateral against 1,150 exclusive medial meniscus training studies; OA 211 against
  558.

## Slice sampling density

- **Usefulness by slice position**
  ([spatial importance map](../../../docs/data-preparation/peripheral-coverage.md#spatial-importance-map),
  `outputs/analysis/s/spatial-importance-20260918/axis_marginals.csv`): sagittal has a middle dip
  (91% of peak) and humps at ±31.5 mm; coronal and axial peak at the centre, coronal staying
  higher anteriorly, axial superiorly. About half the map follows distance from the centre, the
  old sampler's envelope; the dip and humps do not.
- **Per-plane width on frozen checkpoints:** coronal narrower than 40 mm gains (+0.00135 at
  30 mm, sign held to ×4), axial width is inert, a sagittal ×1 gain did not survive higher counts
  ([result](../../../docs/validation/results/2026-09-18-per-plane-sigma-cache.md)).
- **Density gate on `k05-clean`** (measured 2026-09-23, real prediction pipeline with only
  `select_triplets` swapped; baseline reproduces the saved predictions to 5.5e-5; triplet counts
  match; `outputs/analysis/t/two-peak-sagittal-gate-20260923/result.md`):
  - Density proportional to each plane's measured curve (sagittal symmetrized, edge value held):
    ×4 −0.00057, 80% [−0.00120, +0.00008]; ×1 −0.0019. Pool depth is positive toward left,
    posterior and superior, matching the curve file; `sampling_curve.py` there is a working
    implementation.
  - Narrow sagittal two-peak (±31.5 mm, σ 8.3 mm): ×4 −0.00153, 80% [−0.00246, −0.00064];
    ×1 −0.0039. It starves the notch and is dropped.
  - A checkpoint trained on the Gaussian sampler pays a mismatch cost on any new sampler, and
    sampling acts mostly during training, so these gates cannot reject a trained sampler.

## Position encoding

- **Additive 3D embedding** (patient LPS minus the study centre, ÷100 mm, MLP; `unknown_3d` for
  rejected centres) was worth about 0 and was retired; its x was not side-normalized and it sat
  beside the 1D and 2D embeddings
  ([lineage](../../../docs/history/model-lineage.md#five-embedding-predecessor),
  [champion without 3D](../../../docs/inference/submissions/2026-09-15-champion-without-3d-at-inference.md)).
- **Token graph** (kNN message passing with LPS offsets) stayed near initialization in both
  fold-0 runs ([lineage](../../../docs/history/model-lineage.md#retired-token-graph)).
- **`token-attention`** ([result](../../../docs/validation/results/2026-09-23-side-mask-and-token-attention.md#global-token-attention-token-attention); 2026-09-23, branch `token-attention`, patch in
  `outputs/analysis/t/token-attention-vs-k05-clean/token_attention.patch`): one pre-norm transformer
  block over each study's valid tokens with 3D RoPE on raw patient LPS (not side-normalized,
  relative only), 1D and 2D embeddings dropped, reflected studies negate x. Against `k05-clean`:
  macro ×4 0.901284, −0.0015 (95% [−0.0054, +0.0022]); meniscus +0.0027, OA +0.0035, other eight
  −0.0038 with PF OA −0.0152, Fracture −0.0153, MCL −0.0059; selected checkpoint `best-epoch=09`, 1,624 s
  (`outputs/analysis/t/token-attention-vs-k05-clean/comparison.json`). The attention block and the
  dropped embeddings are confounded.

## Out of scope by user decision

- A trained model for side, centre or landmarks; per-label rules outside the four compartment
  labels; per-label coefficients; tuning the midline band or sampler widths to the score;
  reflection-probability experiments; weighting the gold studies.
