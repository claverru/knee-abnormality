# Triplet budget allocation (closed 2026-09-18)

A finished investigation, kept for the negative results it settles. Its per-result evidence is in
the linked records.

Ran 2026-09-17 to 18: can a better a-priori split of a study's triplets across plane–contrast
types and series beat today's fixed 6/6/5/5/5/5, at about the same compute? **No rule beat it.**
What the investigation produced instead is the attention temperature, now the best submission.

- **Per-type contributions** are exact per label: fluid types carry about 74% of the
  positive–negative separation with 61% of the tokens; sagittal fluid leads ACL, axial fluid
  Baker's cyst and effusion, coronal fluid MCL and the menisci; coronal and sagittal T1/PD still
  lead the OA labels ([result](../validation/results/pooling-and-attention/2026-09-17-type-contribution.md)).
- **Reallocation at a fixed total does nothing** above the detection floor, on checkpoints trained
  with a fixed split or with random splits; starving a type does hurt
  ([result](../validation/results/sampling/2026-09-17-test-time-allocation.md)). The token caches
  (`outputs/analysis/token-cache-*-20260917/`) score any allocation in about 30 ms and remain
  usable.
- **Training-time triplet counts do matter** while test-time counts barely move a frozen model,
  so the two must not be confused.
- **Rejected:** the specified missing-type fill (worse than losing the budget), budgets A/B,
  count-normalised attention ([result](../validation/results/pooling-and-attention/2026-09-17-count-normalized-attention-result.md)),
  random per-type allocation in training (a tie, option removed), and the repeated-series share
  ([overnight](../validation/results/campaigns-and-queues/2026-09-18-overnight-results.md)).
- **Adopted:** the attention temperature, τ 0.75, which came out of the bag-size check and scores
  0.939 public with ×4 images
  ([submissions](../validation/results/pooling-and-attention/2026-09-18-temperature-and-count-submissions.md)).

Two questions outlived it and are tracked elsewhere: where the evaluation count turns over above
×4 and what it costs in runtime, and whether splitting a slot's own
budget more evenly between its own series helps
([small changes](../backlog/untried-ideas.md#augmentation-sampling-and-intensity)).
