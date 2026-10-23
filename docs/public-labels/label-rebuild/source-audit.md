# Label rebuild: source audit

Pre-registered 2026-09-28, before any sampling or judging.

**Question:** are some root sources bad enough to remove from A? Per-source gold macro AUC
(0.835–0.888) cannot separate them on 58 studies, so the audit reads reports instead.

- **Roots audited:** all 19 roots of A, including gold-tuned and gold-copying ones (they vote on
  silver). None exists only on gold.
- **Root side per cell:** positive when the root's value is ≥ 0.5 and its symbol is not a
  not-mentioned code; negative when the value is < 0.5 or its symbol is a not-mentioned code
  (Steven v1 0.5, Vmohit `M`, Yehezkiel −1, JevK5 `not_mentioned`, Josh `NOT_MENTIONED`,
  Hosseini `unk` and every `nm` symbol). Cells with no side (missing,
  Vmohit `U`) are skipped.
- **Disagreement:** on silver only, the root is positive while the mean of the other families
  (B's family mean with the root's own family left out) is < 0.3, or the root is negative while
  that mean is > 0.7. Each root's disagreement rate per target is recorded.
- **Sampling:** 20 disagreements per root, allocated across targets in proportion to that root's
  disagreement counts (largest remainder), drawn at random with seed 20260928. Identical
  (study, target) cells are judged once and shared.
- **Blind judging:** the judge sees only the report text, the target name and the
  [official criteria](../../metadata/official-label-criteria.md), never source values or which side
  anyone took. The judge records what the report states: POS, HEDGED, LOW (explicitly below the
  criteria's threshold), NEG or NOT_MENTIONED, with a quoted evidence span. Non-English reports
  are read in the original.
- **Scoring:** the root is right when the judged state is on its side. POS and HEDGED are the
  positive side; NEG, NOT_MENTIONED and LOW the negative side for every target, because every
  official definition excludes mild disease (borderline graded negative). Outlier precision per
  root with a Wilson 80% interval.
- **Decision rule:** drop a root when n ≥ 15, precision < 0.35 and the Wilson 80% upper bound
  < 0.5. Drop a root for specific targets only when that target has n ≥ 8 and meets the same
  thresholds. Nothing else changes.
- **If anything is dropped:** build A2 (`rebuild_state_v2`) with A's pipeline and settings minus
  the dropped roots or cells (same tempering selection, calibration and pre-registered gold
  floor), report B2 (the family mean without them) for information, and check fold 4 as before.
  If nothing is dropped, A stands.
- **Limit:** this judges the report, not the image. A root can be wrong about the report and
  still correlate with image truth, and outlier precision says nothing about agreeing cells.


## Source audit result (2026-09-28)

380 sampled disagreements (371 unique report/target cells) were judged blind from the report
text alone (judgments (`outputs/analysis/l/label-rebuild-20260928/source_audit/judgments.csv`, deleted),
scores (`outputs/analysis/l/label-rebuild-20260928/source_audit/scores.json`, deleted), script
[`audit_label_sources.py`](../../../scripts/adhoc/audit_label_sources.py)).

| Root | Strong-dissent rate | Outlier precision (80% Wilson) | Precision without HEDGED cells | Rule |
|---|---:|---|---:|---|
| JevK5 (ours) | 5.9% | 0.10 [0.04, 0.22] | 0.40 (n 5) | drop |
| Pearse | 4.9% | 0.05 [0.02, 0.15] | 0.06 | drop |
| Bhavesh | 3.2% | 0.00 [0.00, 0.08] | 0.00 | drop |
| Dreaddev | 3.1% | 0.05 [0.02, 0.15] | 0.07 | drop |
| Hosseini | 2.9% | 0.05 [0.02, 0.15] | 0.00 | drop |
| JEV text | 2.8% | 0.10 [0.04, 0.22] | 0.29 (n 7) | drop |
| Laymond | 2.5% | 0.00 [0.00, 0.08] | 0.00 | drop |
| Qwen3-8B | 2.4% | 0.00 [0.00, 0.08] | 0.00 | drop |
| Pilkwang | 2.4% | 0.00 [0.00, 0.08] | 0.00 | drop |
| Yehezkiel | 2.2% | 0.00 [0.00, 0.08] | 0.00 | drop |
| Steven v1 | 2.2% | 0.05 [0.02, 0.15] | 0.00 | drop |
| Josh | 2.1% | 0.35 [0.23, 0.49] | 0.78 | keep |
| Nartaa | 1.9% | 0.35 [0.23, 0.49] | 0.88 | keep |
| Yunus | 1.5% | 0.05 [0.02, 0.15] | 0.06 | drop |
| Zhen v7 | 1.1% | 0.05 [0.02, 0.15] | 0.07 | drop |
| Ramit | 0.7% | 0.00 [0.00, 0.08] | 0.00 | drop |
| lixin73 | 0.6% | 0.00 [0.00, 0.08] | 0.00 | drop |
| Opus/Sonnet | 0.6% | 0.10 [0.04, 0.22] | 0.20 | drop |
| Vmohit | 0.6% | 0.50 [0.36, 0.64] | 0.91 | keep |

**The pre-registered rule is degenerate and was not applied; A2 was not built.** It fires on
16 of 19 roots because it asks whether a lone dissenter beats the consensus of about 16 other
families, and almost every source loses that contest. That is evidence that the family mean
already outvotes individual errors, not that 16 sources are trash; a table from the three
survivors (two families, one gold-tuned) would contradict the audit's own finding.

What the audit does show (post hoc, not pre-registered):

- **No source is trash.** Strong dissent, an upper bound on gross error, ranges 0.6–5.9% of
  silver cells; about 16 other families outvote each such cell in B's mean and A's z.
- **Below-threshold findings counted as positive:** Pilkwang (19 of 20 wrong cells judged LOW:
  its 0.68 below-threshold code is read as a positive probability), Steven v1 (16), lixin73 (14),
  Hosseini (12). This is the rubric problem A's calibration addresses, not extraction noise.
- **Positives on unmentioned findings:** Bhavesh and Qwen (8 of 20 each judged NOT_MENTIONED),
  Laymond (6).
- **Missed or discounted hedged findings:** JevK5 (15 of 18 wrong cells HEDGED) and JEV text.
  Their precision rises without HEDGED cells, so part of this is the audit's convention that
  unsized effusions or unqualified "OA" count as positive-side.
- **Negative dissent that is often right:** Vmohit, Nartaa and Josh (0.78–0.91 without HEDGED).

Judgment conventions: HEDGED covers unsized effusion or Baker's cyst,
unqualified "OA"/"chondropathy", grade 2 MCL injury, partial ACL tear of unstated grade, focal
full-thickness cartilage defects of unstated size and grade III meniscal signal; edema around a
fracture line is HEDGED for Contusion; insufficiency or subchondral fractures are HEDGED for
Fracture. One judge (the agent) read all reports, including Greek, Cyrillic, Turkish, Croatian,
Dutch, German, French and Spanish ones; there is no second reader.
