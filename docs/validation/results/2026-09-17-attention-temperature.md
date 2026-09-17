# 2026-09-17 attention temperature and bag size

**Sharpening the pooling softmax at inference gains +0.0024 macro AUC [+0.0012, +0.0037] with no
retraining:** dividing each token's attention score by τ = 0.75 before the per-label softmax, at
today's allocation, on fold 0. All four available checkpoints gain 0.0021–0.0024, on both halves of
the [search/confirm split](2026-09-17-test-time-allocation.md).

The same rescoring answers the bag-size question left open by the
[test-time allocation result](2026-09-17-test-time-allocation.md#caveats): the plateau past about
8 triplets per type is real, while the fall at very large bags is a weighting effect that sharper
or count-normalised attention removes.

## Method

Pooling is `logit[k] = Σ_tokens softmax(score) · contribution + bias[k]` over a study's valid
tokens. The token caches store every candidate triplet's scores and contributions, so pooling can
be recomputed without the backbone.

- **Variants:** τ ∈ {0.4, 0.5, 0.6, 0.7, 0.75, 0.85, 1.0, 1.5} on `softmax(score / τ)`, and
  count-normalised attention, which subtracts `log(valid tokens contributed by that series)` from
  each token's score before the softmax
  ([proposal](../../backlog/small-changes.md#attention-pooling)).
- **Caches:** `outputs/analysis/token-cache-20260917/` (`ranked-filtered-seed{42,43,44}`, means
  over the three) and `outputs/analysis/token-cache-allocation-20260917/`
  (`allocation-concentration`). Both carry the whole-series intensity window, within 1.2e-4 macro
  AUC of the pipeline at the default allocation.
- **Check:** τ = 1 reproduces the caches' own pooling to 6–8e-15 and the published numbers exactly.
- **Outputs and scripts:** `outputs/analysis/pooling-bagsize-20260917/` (`pooling_lib.py`,
  `run_variants.py`, `tau_sweep.py`, `tau_bootstrap.py`, `results_*.json`, `tau_sweep_*.json`,
  `tau_bootstrap.json`, `README.md`).

## Temperature at today's allocation (macro AUC, full fold 0)

| τ | `ranked-filtered` (3 runs) | `allocation-concentration` |
|---:|---:|---:|
| 0.4 | 0.88296 | 0.88490 |
| 0.5 | 0.88889 | 0.88974 |
| 0.6 | 0.89181 | 0.89206 |
| 0.7 | 0.89291 | 0.89300 |
| **0.75** | **0.89304** | **0.89307** |
| 0.85 | 0.89269 | 0.89252 |
| 1.0 (today) | 0.89073 | 0.89066 |
| 1.5 | 0.88026 | – |

- **Δ at τ = 0.75:** +0.0024 over τ = 1, 95% CI [+0.0012, +0.0037] (paired bootstrap over studies,
  500 resamples, seed 20260917, three-checkpoint mean).
- **Per checkpoint:** seed 42 +0.0020, seed 43 +0.0023, seed 44 +0.0026,
  `allocation-concentration` +0.0024.
- **Per half:** search 0.89468 against 0.89244, confirm 0.89191 against 0.88955.
- The curve peaks smoothly at 0.7–0.75 and falls away on both sides, so this is not a single lucky
  point.

## Bag size

Macro AUC by allocation and variant, `ranked-filtered` mean, full fold 0:

| Allocation | Triplets | τ 0.75 | τ 1.0 | τ 1.5 | Count-norm |
|---|---:|---:|---:|---:|---:|
| uniform 8 | 38.7 | 0.89316 | 0.89088 | 0.88060 | 0.88972 |
| default | 50.7 | 0.89304 | 0.89073 | 0.88026 | 0.88982 |
| uniform 12 | 58.0 | 0.89429 | 0.89187 | 0.88130 | 0.89063 |
| uniform 16 | 77.2 | 0.89490 | 0.89214 | 0.88116 | 0.89093 |
| uniform 24 | 110.8 | 0.89566 | 0.89245 | 0.88104 | 0.89129 |
| all candidates | 165.1 | 0.89389 | 0.88963 | 0.87618 | 0.89101 |

- **The plateau is real.** Every variant flattens from about 8–12 triplets per type, so the
  saturating shape is not a softmax-dilution artefact.
- **The fall at 165 triplets is a weighting effect, not information loss.** It shrinks under
  τ = 0.75, grows under τ = 1.5, and disappears under count normalisation: all minus uniform 24 is
  −0.0003 [−0.0008, +0.0003], within the cache's own floor, and +0.00001 on the allocation cache.
  Duplicating tokens would leave pooling unchanged, so this is not mechanical dilution: the extra
  candidates are weaker evidence that takes attention mass away from the better triplets, and
  either sharper or count-normalised attention limits how much they take.
- **Count normalisation is otherwise neutral** at today's allocation (−0.0009 on the three runs,
  +0.0001 on the allocation checkpoint), so it fixes the large-bag penalty without being an
  improvement by itself.
- **τ = 0.75 helps everywhere,** including at today's allocation, so it is not a bag-size fix.
  Sharper attention is simply better for these trained models.

## Reading

- **This is an inference-only change** to `_attention_pool` in
  `src/knee_abnormality/modeling/model.py`. It adds no compute and needs no retraining, so it
  applies to existing checkpoints and their Kaggle bundles.
- **It supersedes the read-only diagnostic** proposed in the
  [attention-pooling entry](../../backlog/small-changes.md#attention-pooling); the diagnostic is
  now done, at a finer grid, on four checkpoints.
- **Untested:** training with the temperature (the trained scorer may adapt, moving the optimum),
  τ between 0.7 and 0.8, per-label temperatures, and the interaction with series dropout.

## Caveats

- Fold 0 only, on the caches' window approximation. No exact rescoring was run for the temperature
  variants, and no submission has tested it. The [×3 evaluation
  count](../../backlog/endgame.md#test-time-augmentation) is a reminder that a fold-0 gain need not
  transfer to the leaderboard.
- Changing pooling at inference is off-distribution for models trained at τ = 1.
- The gain is close to the run-to-run noise of training (0.0015–0.0018), but it is measured on
  frozen checkpoints, where the paired bootstrap is the right uncertainty and excludes zero.
