# 2026-09-23 sibling masking at inference

**Measured, inference only, no training.** Hiding the other compartment (the sibling) from each
compartment label moves predictions consistently. The model does read the sibling, in both
directions: a diseased sibling raises the score, and a clean sibling lowers it. Removing it
trades one group of negatives against the other and leaves macro AUC flat. Most of the gap
between "negatives with a positive sibling" and "negatives with a clean sibling" remains with
the sibling hidden. The training-time test, `side-mask`, is recorded in the
[knee-geometry prior work](../../../autoresearch/campaigns/knee-geometry/prior-work.md#cross-talk-and-the-side-mask).

## Setup

- **Checkpoint:** `k05-clean` fold 0, epoch 8, ×4 images, attention temperature 1, split
  `all_5fold_scanner_v1`, combined labels. Scored on the `sibling-masking` worktree code
  (main at `d914079`). The checkpoint predates the oblique-code removal, so the script skips the
  preprocessing-identity check; the intact arm reproduces the saved predictions to 5.5e-5.
- **Side:** each token's signed medial position, the side-resolved `medial_axis` that anatomical
  attention already uses. Tokens within ±10 mm of the study centre (the midline band) stay visible
  to both compartments. 27 of 905 studies have no resolved side and stay unmasked; no study-target
  lost all its own-side tokens.
- **Arms:**
  - `intact`: the checkpoint as trained.
  - `token_mask`: each of the four compartment labels pools only own-side and band tokens.
  - `pixel_blank`: `token_mask` plus sibling-side pixels beyond the band set to zero (the value
    outside the prepared rectangle), with one backbone pass per side.
- **Script:** `scripts/adhoc/score_sibling_masking.py` on the unmerged `sibling-masking` branch
  (it needs that branch's `model.compartment_evidence`; a source snapshot is kept with the outputs); outputs
  in `outputs/analysis/sibling-masking-20260923/` (per-arm predictions, per-study token counts,
  summary). Intervals are central 80% paired report-group bootstraps, 2,000 draws.

## Results

Macro AUC: intact 0.902762; `token_mask` +0.00019 [−0.00004, +0.00044]; `pixel_blank` −0.00188
[−0.00266, −0.00117].

Per label, AUC of all positives against one group of negatives (change from intact, 80% interval):

| Label | Negatives | n | Intact | `token_mask` | `pixel_blank` |
|---|---|---:|---:|---:|---:|
| Medial Meniscus | sibling clean | 479 | 0.9327 | −0.0007 | −0.0013 |
| Medial Meniscus | sibling positive | 72 | 0.8932 | **+0.0080** [+0.0055, +0.0105] | −0.0091 |
| Lateral Meniscus | sibling clean | 479 | 0.9251 | **−0.0033** [−0.0052, −0.0016] | +0.0019 |
| Lateral Meniscus | sibling positive | 288 | 0.8657 | **+0.0084** [+0.0059, +0.0109] | −0.0216 |
| Medial OA | sibling clean | 523 | 0.9073 | −0.0011 | −0.0108 |
| Medial OA | sibling positive | 54 | 0.8225 | **+0.0116** [+0.0075, +0.0158] | +0.0006 |
| Lateral OA | sibling clean | 523 | 0.9084 | **−0.0026** [−0.0040, −0.0013] | −0.0074 |
| Lateral OA | sibling positive | 141 | 0.7428 | **+0.0134** [+0.0095, +0.0171] | +0.0109 |

Overall per-label changes under `token_mask` range from +0.0001 to +0.0011.

## Interpretation

- **The pooling reads the sibling.** All four "sibling positive" strata rise when the sibling's
  tokens are removed, with intervals clear of zero.
- **The sibling also helps.** The "sibling clean" strata fall for both lateral labels, so a clean
  sibling was lowering scores on healthy knees. At inference the two effects cancel. A model trained
  without the sibling may not face the same trade; that is the untested part.
- **Most of the gap is not sibling evidence.** Lateral OA against medial-only negatives reaches
  0.756 with sibling tokens removed and 0.754 with sibling pixels removed too, against 0.908
  against clean negatives: about 8% of the gap closes. The lateral meniscus closes about 14%.
  With only its own side visible, the lateral compartment of a knee with medial OA still scores
  above a healthy knee's (hypothesis: sub-threshold degeneration below the report's OA cutoff).
- **Pixel blanking damages the frozen model.** It lowers the "sibling clean" strata as well
  (Medial OA −0.011), so half-black inputs are out of distribution. It gives no clean reading of
  cross-talk through each token's receptive field; only a model trained on such inputs can.
- **Residual shared view:** the ±10 mm band keeps the intercondylar notch, including tibial-spine
  osteophytes, visible to both sides.

## Ceiling

Each label's AUC is a negative-weighted average of its strata. If every "sibling positive"
stratum reached its "sibling clean" level on this checkpoint, macro AUC would rise by about
+0.0059 (Lateral OA +0.0029, Lateral Meniscus +0.0019, Medial OA +0.0007, Medial Meniscus
+0.0004). Hiding the sibling at inference recovers little of it.
