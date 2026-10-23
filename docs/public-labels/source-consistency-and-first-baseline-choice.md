# Source consistency and first-baseline label choice

## Source consistency and remaining limits

The fixed folds contain 54 repeated **normalized** report groups, distinct from
the 46 exact-text groups in the original metadata audit. Label vectors differ
within 37 of those groups for each Steven recipe, three groups for Pilkwang,
and one for lixin73. This is extraction consistency evidence, not proof of
duplicate patients or clinical errors; labels have not been averaged or repaired.

Across Steven v4 and Pilkwang, 736 study-target pairs differ by more than 0.5.
ACL accounts for 206 and Contusion for 188 of those pairs. Their numerical
scales and missingness conventions differ, so these are review candidates,
not adjudicated errors. The local disagreement table contains IDs and values,
without report text or reference-based filtering.

Steven's model/prompt and lixin73's extraction details remain unavailable in
these downloaded snapshots. Pilkwang's producer is incomplete without its
helper module. Consequently, this audit establishes structural validity,
observed transformations and reference agreement; it does not establish full
clinical rubric compliance or regenerate any labels.

Steven's public development reports compare repairs and blends against the
reference studies. Their scores, and comparisons selected using them, are
development evidence. Preserve the project's [fixed folds](../validation/INDEX.md),
but do not claim those folds undo prior public reference-set feedback.
The source claim that teacher extraction AUC bounds image-model performance
is also not a mathematical guarantee; see [supervision evidence](../research/competition-and-prior-art/public-research.md#report-derived-supervision).

## Implication for the first baseline

**Decision, 2026-09-10:** use **Steven Lee's v4 blend** as the report-derived
label source for the first baseline, as selected by the user after this audit.
The pinned file is
`data/labels/public/stevenleehans/version-6/llm_labels_v4_blend.csv`, from Kaggle
dataset version **6**, identified as `steven_v4` in the
[source manifest](../../configs/labels/public-label-sources.json), which records its SHA-256.
Use the published soft targets and retain v1/v2 for missingness and repair
provenance. Baseline3 has since measured one fold using v4, but no paired image
model comparison establishes that source as better than v2 or Pilkwang.

Its 0.0054 advantage over v2 on the reused 58-study endpoint does not establish
that it trains a better image model. v2 remains a useful single-source control,
and Pilkwang is usable on the entire report-only pool.

Fold-aware training later retained official gold separately, uses it only on
eligible training rows, and preserves every held-out study under the
[all-study protocol](../validation/validation-protocol.md). That later
implementation does not change this audit's scope: it altered no labels or
folds and ran no LLM or image-model comparison between label sources.
