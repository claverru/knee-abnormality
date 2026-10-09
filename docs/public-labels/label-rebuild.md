# Label rebuild from public report sources

**Adopted 2026-09-28 (user decision): A2 (`rebuild_state_v2`) with the new split
`all_5fold_scanner_v2` is the recipe's label source** ([status](#status)); built the same day in the
`qwen-lateral-labels` worktree, since merged into `main` and removed. Goal: the best training labels obtainable
from every published Kaggle report-label table (plus our own JevK5 attributes), built as if no
labels existed yet. Until the adoption, main kept `combined_v1`.

## Target definition

The training target is **P(test-style label = 1 | report)**. Test labels are image-adjudicated by
two MSK radiologists with a third adjudicating; borderline findings are graded negative; a report's
silence is an unobserved label, never an absent one
([criteria](../metadata/official-label-criteria.md), host text retrieved 2026-09-17, rechecked
against this design 2026-09-28: no contradiction). Two error sources need different fixes:

1. **Extraction noise** (one source misreads a report): aggregation across independent sources,
   fitted on report data without labels.
2. **Shared semantic mismatch** (what silence, hedging, below-threshold grades and compartment
   wording mean for an image label): every source shares it, so only gold can inform it, through
   a low-dimensional mapping pooled across targets.

Unaddressed cells are about a quarter of all cells (Steven v1 unaddressed: Synovitis 3,689,
Baker's 2,126, Fracture 1,890, Lateral OA 1,090, Contusion 1,058, Medial OA 896 of 4,407), so their
valuation is probably the largest single decision
([encodings](../public-labels/uncertainty-and-confidence-encoding.md)).

## Sources

### Kaggle sweep, 2026-09-28

Searches as in the [09-27 investigation](../public-labels/2026-09-27-public-label-investigation.md)
(knee, RSNA, report, labels, pseudo, soft, weak, LLM, teacher, annotation; updated since Sep 26).
One new label table: `narisettichaitanya/rsna-knee-teacher-soft-labels` v1 (Sep 27 14:16 UTC,
4,407 rows). Its values are image-model predictions (every cell continuous, gold rows not copied),
so it is not report evidence and is excluded. Version-pinned downloads under
`data/labels/public/<owner>/<slug>/downloaded-20260928/`: that table, Bhavesh v2,
`dhiyaneshbabu/rsna-knee-labelled-train` v1 and `yehezkielhaganta/psuedo-labeling-knee` v7
(both byte-identical to the 09-27 investigation copies).

### Lineage and root sources

One vote per root extraction; inferred and copied cells are dropped.

| Root | Model / method | Evidence of lineage | Report states recoverable |
|---|---|---|---|
| Steven v1 | LLM | v2 = v1 with Synovitis inferred from Effusion; v4 = 0.5 v2 + 0.5 lixin73 (exact reconstruction) | 0.5 = unaddressed |
| lixin73 | GPT-5.6-Sol | Half of Steven v4 | mostly binary + confidence |
| Pilkwang v2 | Claude (producer default) | Zhen v8 identical on gold; davidadekanmi copies it | NO / UNK / three YES grades |
| Zhen v7 | not stated | Same verdict schema as Pilkwang; kappa checked below | verdict + score |
| Laymond v2 | Qwen2.5-32B | independent | mentioned / negated / label / confidence |
| Yunus LLM | Gemini Flash | independent (its merges are derivatives) | four severity levels |
| JEV text | OpenJev ensemble | **gold-tuned prompts** | probability |
| Dreaddevelopment | not stated | no gold rows | probability |
| Ramit | GPT-luna | independent | NA / NEG / POS, severity, hedged |
| Josh | gpt-5.6-luna | **gold-tuned (DSPy)** | annotation + confidence |
| Vmohit | not stated | Nartaa Gemini raw agrees 98.9%; no gold rows | P / N / B (below threshold) / U / M |
| Nartaa v1 | Gemini 3 Flash | family with Vmohit if kappa ≥ 0.9 | mention flag + value |
| Qwen3-8B | Qwen3-8B | independent | probability |
| Bhavesh | undocumented | independent | probability |
| Hosseini JEV v2 | Jev-style | same author built gold pairwise comparisons: **treated as possibly gold-tuned** | neg / pos / unknown probabilities |
| Pearse frontier | vote counts of unnamed labelers | unknown constituents | value, positive and negative counts |
| Opus/Sonnet (dhiyaneshbabu) | Claude | **gold copied into gold rows** | binary |
| Yehezkiel | several LLMs | **gold copied into gold rows** | 0 / 1 / −1 unaddressed |
| JevK5 v3 (**ours, not a Kaggle table**) | Qwen3.5-4B LoRA, 40 questions | [extraction](../ongoing/report-attribute-extraction.md) | status with not-mentioned, hedging, grades, extent |

Excluded: Steven v2/v4 and every Steven or Pilkwang derivative (Tonylin, Niklita, Rayan, Shingo,
Fracture hybrids, JEV HYBRID, Yunus merges, Zhen v8, davidadekanmi, saisahanas, Barun, Jezreal,
Saidul, Noisyislands, Zaid, Tasmeem); image-model or fitted outputs (narisetti teacher, tsuyu,
Giovanni, our image blend); tables without a direct per-label report extraction or with unknown
semantics (Nekkon NLI entailment, achelijndiamantidis, lvthanhxun fused teacher, Freshtime,
posterior-calibrated, Manual Label Boost). Families among the kept roots are merged when binary
silver kappa ≥ 0.90 averaged over targets (complete linkage), keeping the member with the richer
state encoding.

## Design

Each item: evidence-backed or guess, with confidence.

- **A. Per-cell report state per root.** Each root keeps its own observation alphabet (its native
  symbols; continuous values in fixed bins 0/0.1/0.3/0.5/0.7/0.9/1 with explicit not-mentioned
  sentinels kept as their own symbol). *Evidence-backed (high):* collapsing Steven v1 to three
  states lost gold ordering (0.878 → 0.809), and sources encode silence differently.
- **B. Anchored Dawid–Skene over five latent report states** per target: NEG, NOT_MENTIONED, LOW
  (explicit below-threshold), HEDGED, POS. Every root has a class-conditional distribution over its
  symbols, fitted by EM on all 4,407 reports without labels. Anchors (JevK5, Ramit, Vmohit,
  Pilkwang) carry hand-written symbol → state maps used for initialisation and as structural
  constraints: a state emits an anchor symbol outside its map with probability scaled by 0.02
  (see pre-gold revisions); other roots are unconstrained with Dirichlet(1) smoothing. Gold-tuned and
  gold-copying roots are dropped on gold rows everywhere. *Guess (moderate):* the earlier binary
  Dawid–Skene over correlated voters lost on gold (0.8903,
  [combined labels](../public-labels/combined-labels.md#alternatives-measured-and-rejected)); this
  version differs by lineage-level roots, multi-state alphabets and tempering.
- **C. Dependence tempering.** Posterior ∝ prior × ∏ likelihood^τ with τ ∈ {1, 0.7, 0.5, 0.35,
  0.25, 0.15, 0.1}, chosen label-free by held-out-source log-likelihood on silver (predict each root's symbol
  from the others). *Caveat:* this criterion rewards shared bias, so it may only set noise
  reduction, never semantics. *Guess (moderate).*
- **D. Semantic calibration** pooled over the twelve targets:
  `P(y = 1) = Σ_k q_k · sigmoid(β_k + α_t + γ·z)`, with q the state posterior, β_k shared state
  effects (N(0, 2.5²)), α_t per-target offsets with strong shrinkage (N(0, 0.5²)), and γ a shared
  slope on z, the centred logit of the root-family mean (N(0, 1²)), which keeps within-state
  ordering. MAP fit on the 696 gold cells. *Evidence-backed (moderate) for the shape:* gold rates
  by report call are 0.08 negative, 0.14 not addressed, 0.54 LOW, 0.78–0.92 higher grades
  ([report attributes](../public-labels/report-attributes.md#gold-rate-by-report-call)); the
  pooling is needed because single targets hold 3–25 gold cells per state.
- **E. Soft targets** stay probabilities for all twelve targets; gold source rows keep
  `combined_v1` values (the loader overrides them with official gold); evaluation targets stay
  frozen.

### Arms, fixed before any gold number

- **A:** state aggregation (B, C) + gold-calibrated mapping (D), final fit on all gold.
- **B:** family-mean control: per target, mean of each root family's own published probability
  (sentinels −1 / U / M missing; a cell no root addresses keeps `combined_v1`).
- **A0 (diagnosis only, not for submission unless the user decides):** A's posteriors with the
  rubric map `q_POS + 0.5·q_HEDGED`, no gold.

### Pre-registered gold evaluation

- Gold-evaluated variants drop JEV text, Josh and Hosseini (gold-tuned) and Opus/Sonnet and
  Yehezkiel (gold copied) on gold rows. `combined_v1`'s own gold rows contain JEV and Josh; that
  contamination favours the baseline.
- A's gold predictions are **cross-fitted over gold report groups** (leave one group out: the
  calibration is fitted on the other groups). The state model uses no labels and is not refitted.
- Metrics: pooled gold log loss over all 696 cells and macro AUC over the twelve targets, for
  A (out-of-fold), B, A0 and `combined_v1` (0.907537 macro AUC). Paired bootstrap over gold
  studies, 10,000 draws, seed 20260927, draws lacking both classes for any target discarded.
- **No gold iteration.** These numbers are reported, not used to choose or tune. Any design change
  forced after seeing them is recorded below as post hoc.

### Pre-gold revisions (label-free, before any gold number)

1. **Anchors as constraints, not priors.** With 20-pseudo-count Dirichlet anchors, EM let latent
   states drift: at τ 0.7 the Synovitis HEDGED state absorbed 1,304 cells that every anchor calls
   unaddressed, and MCL POS emitted JevK5 `partial_unspecified`. Shared state effects in the
   calibration need the same state meaning across targets, so anchors now constrain emissions.
2. **Tempering grid extended.** The held-out-source log-likelihood rose monotonically to the grid
   edge (τ 0.25: −0.4858); a label-free probe found the optimum inside (τ 0.15: −0.4775, τ 0.1:
   −0.5027, τ 0.05: −1.1476), so 0.15 and 0.1 were added. Such strong tempering says the roots
   are heavily dependent, as expected from shared report text and shared model families.
3. **Families** (mean silver binary kappa ≥ 0.90, complete linkage): {Vmohit, Josh} (0.946) and
   {Pilkwang, Yehezkiel} (0.901); every other root is its own family. Vmohit–Nartaa is 0.918 but
   Josh–Nartaa 0.831 blocks the merge under complete linkage. Family representatives (first in
   anchor-priority order) are Vmohit and Pilkwang, so the Vmohit family contributes no gold rows.

## Results (2026-09-28)

**By the pre-registered floor, B passes and A fails.** Both tables exist; only B was recommended
for training unless the user decided otherwise (see [decisions](#decisions-to-check)). The user
trained both and later adopted A2, the fixed version of A ([A2](#a2-common-state-scale-and-the-zhen-lineage-fix-2026-09-28)).

Selected τ = 0.15 (held-out-source log-likelihood −0.4775; 1.0: −0.5111, 0.25: −0.4858,
0.1: −0.5027).

### Gold evaluation (pre-registered)

58 gold studies, 696 cells; gold-tuned and gold-copying roots removed on gold rows; A cross-fitted
over the 58 gold report groups (one study each); 10,000 valid bootstrap draws.

| Arm | Macro AUC | Δ vs `combined_v1` (95%) | P(better) | Log loss | Δ log loss (95%) |
|---|---:|---|---:|---:|---|
| `combined_v1` | 0.9075 | — | — | 0.4471 | — |
| **A**, state + calibration, out-of-fold | 0.8838 | −0.0238 [−0.0372, −0.0109] | 0.000 | 0.3606 | −0.0865 [−0.1286, −0.0447] |
| **B**, family mean | 0.9022 | −0.0054 [−0.0181, +0.0072] | 0.193 | 0.3732 | −0.0739 [−0.1047, −0.0444] |
| A0, state rubric (diagnosis) | 0.9006 | −0.0069 [−0.0200, +0.0059] | 0.137 | 0.7323 | +0.2851 [+0.1474, +0.4343] |

| Target | `combined_v1` | A (OOF) | B | A0 |
|---|---:|---:|---:|---:|
| ACL | 0.996 | 0.994 | 0.995 | 0.995 |
| MCL | 0.996 | 0.975 | 0.986 | 0.982 |
| Medial Meniscus | 0.948 | 0.928 | 0.940 | 0.953 |
| Lateral Meniscus | 0.879 | 0.862 | 0.886 | 0.880 |
| Medial OA | 0.932 | 0.933 | 0.943 | 0.924 |
| Lateral OA | 0.833 | 0.822 | 0.849 | 0.851 |
| PF OA | 0.929 | 0.909 | 0.923 | 0.906 |
| Effusion | 0.877 | 0.826 | 0.867 | 0.853 |
| Synovitis | 0.790 | 0.773 | 0.811 | 0.794 |
| Baker's | 0.944 | 0.909 | 0.917 | 0.946 |
| Contusion | 0.860 | 0.854 | 0.872 | 0.838 |
| Fracture | 0.907 | 0.819 | 0.838 | 0.886 |

The `combined_v1` row is favoured: its gold rows contain JEV and Josh, and its four consensus
targets were chosen on these studies. B uses neither.

**Diagnostics, after the fact and changing nothing:** fitted on all gold, A scores 0.9022
in-sample, the same as B, so most of the out-of-fold loss comes from refitting without each
single held-out study. A likely mechanism is leave-one-out pessimism: removing a positive lowers
its own target offset. This is plausible, not proven. Log loss, which the calibration targets, is
best for A.

### Report states and calibration

Gold-positive rate by argmax state, pooled: NEG 0.03 (185 cells), NOT_MENTIONED 0.13 (193),
LOW 0.34 (87), HEDGED 0.61 (80), POS 0.85 (151). The states keep their intended order and meaning.

Silver cells by argmax state (NEG / NOT_MENTIONED / LOW / HEDGED / POS):

| Target | NEG | NM | LOW | HEDGED | POS |
|---|---:|---:|---:|---:|---:|
| ACL | 2,804 | 347 | 533 | 220 | 445 |
| MCL | 3,187 | 410 | 528 | 175 | 49 |
| Medial Meniscus | 1,674 | 241 | 682 | 116 | 1,636 |
| Lateral Meniscus | 2,774 | 443 | 475 | 79 | 578 |
| Medial OA | 1,751 | 976 | 483 | 793 | 346 |
| Lateral OA | 1,943 | 1,295 | 433 | 480 | 198 |
| PF OA | 1,554 | 807 | 540 | 964 | 484 |
| Effusion | 1,291 | 429 | 1,449 | 593 | 587 |
| Synovitis | 49 | 3,763 | 153 | 0 | 384 |
| Baker's | 1,110 | 2,152 | 534 | 0 | 553 |
| Contusion | 1,774 | 1,280 | 398 | 487 | 410 |
| Fracture | 820 | 3,239 | 0 | 57 | 233 |

Calibration (final fit on all gold): state effects NEG −2.19, NM −2.01, LOW −2.11, HEDGED −1.53,
POS −1.28; slope γ 0.84 on z; target offsets −0.82 (MCL) to +0.98 (Synovitis). Because z carries
much of the signal, the map is read at each state's median silver z:

| Target | NEG | NM | LOW | HEDGED | POS |
|---|---:|---:|---:|---:|---:|
| ACL | 0.02 | 0.10 | 0.18 | 0.63 | 0.93 |
| MCL | 0.02 | 0.11 | 0.24 | 0.72 | 0.93 |
| Medial Meniscus | 0.01 | 0.07 | 0.13 | 0.47 | 0.87 |
| Lateral Meniscus | 0.03 | 0.13 | 0.24 | 0.62 | 0.91 |
| Medial OA | 0.02 | 0.07 | 0.22 | 0.57 | 0.87 |
| Lateral OA | 0.02 | 0.07 | 0.20 | 0.49 | 0.80 |
| PF OA | 0.04 | 0.12 | 0.31 | 0.69 | 0.89 |
| Effusion | 0.03 | 0.14 | 0.32 | 0.76 | 0.87 |
| Synovitis | 0.05 | 0.20 | 0.61 | — | 0.90 |
| Baker's | 0.02 | 0.07 | 0.30 | — | 0.81 |
| Contusion | 0.02 | 0.06 | 0.12 | 0.36 | 0.78 |
| Fracture | 0.02 | 0.08 | — | 0.53 | 0.82 |

### Silver changes against `combined_v1`

Positive rate at 0.5 and changed calls on the 4,349 silver studies:

| Target | `combined_v1` | A | A changed | B | B changed |
|---|---:|---:|---:|---:|---:|
| ACL | 14.7% | 14.3% | 23 | 15.8% | 55 |
| MCL | 5.2% | 5.0% | 14 | 6.0% | 35 |
| Medial Meniscus | 40.4% | 38.3% | 130 | 41.0% | 70 |
| Lateral Meniscus | 15.4% | 15.0% | 25 | 15.4% | 24 |
| Medial OA | 37.3% | 19.4% | 782 | 28.6% | 385 |
| Lateral OA | 27.4% | 9.8% | 763 | 16.6% | 470 |
| PF OA | 31.7% | 31.8% | 68 | 35.2% | 157 |
| Effusion | 59.4% | 27.1% | 1,411 | 29.0% | 1,328 |
| Synovitis | 12.4% | 12.3% | 7 | 12.5% | 13 |
| Baker's | 24.7% | 12.8% | 522 | 22.3% | 107 |
| Contusion | 17.2% | 10.6% | 289 | 17.9% | 197 |
| Fracture | 6.1% | 6.2% | 10 | 6.3% | 11 |

Fold 4 (3,538 training studies): validation targets and gold training targets are identical to
`combined_v1` for both tables. Training positives change most for A in Lateral OA 962 → 345,
Medial OA 1,310 → 689, Effusion 2,097 → 978 and Baker's 868 → 451, and for B in Effusion
2,097 → 1,047, Lateral OA 962 → 590 and Medial OA 1,310 → 1,007. Class-balance weights move with
them (`fold4_training_positives.json`).

### Decisions to check

1. **A is not recommended by the rule.** Training it anyway rests on the in-sample tie with B and
   the better log loss.
2. **Effusion** falls to about 28% positive in both tables, the same direction as the
   [`combined_v2` rule](../public-labels/report-attributes.md#consensus-mean-rule-tested-on-gold)
   that lost gold AUC; B's Effusion AUC is 0.867 against 0.877.
3. **B loses Fracture** (0.838 against 0.907, whose consensus was selected on these studies) and
   Baker's (0.917 against 0.944), and gains OA, Synovitis and Contusion.
4. **Our JevK5 attributes** are a root and an anchor; they are not a Kaggle table.
5. **Hosseini** is treated as possibly gold-tuned; the Tonylin-style derivatives, Nekkon NLI,
   achelijndiamantidis, lvthanhxun, Freshtime and the new narisetti image teacher are excluded.
6. **Josh sits in the Vmohit family,** so the state model never reads it (Vmohit represents it).
7. **The two pre-gold revisions** (anchor constraints, τ grid extension) were made after silver
   diagnostics and before any gold number.

### Outputs and launch

- Tool: [`prepare_rebuilt_labels`](../../scripts/tools/prepare_rebuilt_labels.py), CPU, about
  40 s, deterministic, refuses existing outputs:
  `rtk proxy uv run --locked python -m scripts.tools.prepare_rebuilt_labels --analysis-dir outputs/analysis/label-rebuild-new`
  (move the two prepared directories and manifests aside first).
- Tables: `data/labels/prepared/rebuild-state-v1/labels.csv` (SHA-256 `d47b631d…`) and
  `data/labels/prepared/rebuild-mean-v1/labels.csv` (`0db443ab…`); manifests
  [`rebuild-state-v1`](../../configs/labels/rebuild-state-v1-label-sources.json) and
  [`rebuild-mean-v1`](../../configs/labels/rebuild-mean-v1-label-sources.json).
- Analysis: `outputs/analysis/l/label-rebuild-20260928/` (`report.json` with families, kappa,
  tempering, states, calibration and gold evaluation; `state_model.json`;
  `fold4_training_positives.json`).
- Launch (user, from the worktree, after checking GPU use; pairs with `scanner-confusion-control`,
  fold 4, seed 703791404; compare publicly with S73 at ×4, τ 0.75):

  ```bash
  rtk proxy nohup uv run --locked python -u -m scripts.pipeline.train configs/training.yaml --name rebuild-mean-labels --paths.label-manifest configs/labels/rebuild-mean-v1-label-sources.json --labels.source rebuild_mean_v1 --run-id rebuild-mean-candidate > outputs/training/rebuild-mean-candidate.log 2>&1 < /dev/null &
  rtk proxy nohup uv run --locked python -u -m scripts.pipeline.train configs/training.yaml --name rebuild-state-labels --paths.label-manifest configs/labels/rebuild-state-v1-label-sources.json --labels.source rebuild_state_v1 --run-id rebuild-state-candidate > outputs/training/rebuild-state-candidate.log 2>&1 < /dev/null &
  ```

  Local validation scores against the frozen `combined_v1` targets, which both tables move away
  from; the public score decides.

## Source audit (pre-registered 2026-09-28, before any sampling or judging)

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
  [official criteria](../metadata/official-label-criteria.md), never source values or which side
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


### Source audit result (2026-09-28)

380 sampled disagreements (371 unique report/target cells) were judged blind from the report
text alone ([judgments](../../outputs/analysis/l/label-rebuild-20260928/source_audit/judgments.csv),
[scores](../../outputs/analysis/l/label-rebuild-20260928/source_audit/scores.json), script
[`audit_label_sources.py`](../../scripts/adhoc/audit_label_sources.py)).

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

Judgment conventions the user should check: HEDGED covers unsized effusion or Baker's cyst,
unqualified "OA"/"chondropathy", grade 2 MCL injury, partial ACL tear of unstated grade, focal
full-thickness cartilage defects of unstated size and grade III meniscal signal; edema around a
fracture line is HEDGED for Contusion; insufficiency or subchondral fractures are HEDGED for
Fracture. One judge (the agent) read all reports, including Greek, Cyrillic, Turkish, Croatian,
Dutch, German, French and Spanish ones; there is no second reader.

## Training results

**B (`rebuild-mean-candidate`), fold 4, seed 703791404:** best epoch 9 (×1 0.901624 against the
frozen targets). Checkpoint selection is unaffected by the frozen targets: epoch 9 is best for both
runs against either label set. **Public 0.946 (S75), tying S73 (0.946)**, bundle `1c3e486e`, ×4,
τ 0.75, batch 4.

Cross-evaluation at ×4 (report-group bootstrap, 2,000 draws, seed 20260914):

| Metric, targets | Control | B | Δ (80% / 95%) |
|---|---:|---:|---|
| Hard macro, `combined_v1` | 0.909733 | 0.908746 | −0.0010 ([−0.0030, +0.0012] / [−0.0043, +0.0022]) |
| Hard macro, B | 0.916103 | 0.919106 | +0.0030 ([+0.0009, +0.0050] / [−0.0001, +0.0060]) |
| Soft macro, `combined_v1` | 0.7959 | 0.7913 | −0.0047 |
| Soft macro, B | 0.7800 | 0.7806 | +0.0006 |

Each model wins on its own targets: uninformative by the pre-stated rule. Against
`combined_v1`, B loses where it moved levels (Effusion −0.024, Baker's −0.012 hard) and gains
Fracture +0.013. About 3,300 silver threshold changes across twelve targets left the public score
unchanged. Training BCE is not comparable across tables: target entropy (the BCE floor) is 0.293
for `combined_v1`, 0.348 for B and 0.305 for A.

## A2: common state scale and the Zhen lineage fix (2026-09-28)

A's final probability is mostly `offset + state effect + 0.84 × z`, where z is the logit of the
equal-weight family mean, so the raw values entering that mean set most of A's ranking. Two
defects there, found after S76 by the source audit and a per-code inspection:

- **Inconsistent "not mentioned", below-threshold and hedged values.** Not-mentioned cells
  entered the mean at 0.50 (Steven v1, 13,438 cells), 0.34 (Pearse, Josh), 0.28 (Pilkwang UNK),
  0.20–0.24 (Ramit, Zhen) and 0.02–0.12 elsewhere, and as missing for Vmohit. Below-threshold
  codes entered at 0.68 (Pilkwang "minimal", 6,461 cells), 0.60 (Ramit `pos_LOW`) and 0 (Vmohit
  `B`); hedged at 0.86 (Ramit) or missing (Vmohit `U`). The pooled gold positive rates by state
  from A are 0.135 not mentioned, 0.345 below threshold and 0.613 hedged.
- **Zhen v7 duplicates Pilkwang.** Its verdicts agree with Pilkwang's on 100% of 52,872 cells;
  only the minimal and UNK cells are rescored. The kappa rule missed it (rank correlation 0.976),
  so it voted as a separate family in A.

A2 (`rebuild_state_v2`, maintained tool, v1 generator snapshotted at
`outputs/analysis/l/label-rebuild-20260928/source/prepare_rebuilt_labels_v1.py`, SHA-256
`0a6226d2`) changes only these: codes whose anchor map or silent symbol (`nm`, `unk`,
`NOT_MENTIONED…`) means not mentioned, below threshold or hedged take those three rates before
averaging, negative and positive codes keep each source's value, and Zhen v7 joins the Pilkwang
family. Everything else (τ 0.15 re-selected, calibration refit, gold protocol) is unchanged. The
three rates come from A's gold fit: five pooled numbers, no new fitting on gold.

| Gold (58) | Macro AUC | Log loss |
|---|---:|---:|
| `combined_v1` | 0.9075 | 0.447 |
| A2 out-of-fold | 0.8798 (A: 0.8838) | 0.365 (A: 0.361) |
| B2 family mean | 0.9015 (B: 0.9022) | 0.373 (B: 0.373) |

Gold cannot separate A2 from A. On silver, A2 stays close to A: rank correlation 0.95–0.999 per
target and 264 threshold changes in all, mostly Medial OA (91), PF OA (57) and Lateral OA (34).
Fold 4: validation and gold training targets identical to `combined_v1`; training positives
Medial OA 760 (A 689), Lateral OA 370 (A 345). Table
`data/labels/prepared/rebuild-state-v2/labels.csv` (`5877acaf…`), manifest
`configs/labels/rebuild-state-v2-label-sources.json`, analysis
`outputs/analysis/l/label-rebuild-20260928/v2/`. Launch (user):

```bash
rtk proxy nohup uv run --locked python -u -m scripts.pipeline.train configs/training.yaml --name rebuild-state-v2-labels --paths.label-manifest configs/labels/rebuild-state-v2-label-sources.json --labels.source rebuild_state_v2 --run-id rebuild-state-v2-candidate > outputs/training/rebuild-state-v2-candidate.log 2>&1 < /dev/null &
```

### Reproducing A2

Rerun on 2026-09-28 into scratch directories, the tool and the split builder reproduce the
adopted files byte for byte: `rebuild-state-v2/labels.csv` (`5877acaf…`) and
`rebuild-mean-v2/labels.csv` (`a550cd46…`) in 42 s, and all four files of
`all_5fold_scanner_v2` from `scripts.tools.validation prepare --config configs/training.yaml`.

```bash
rtk proxy uv run --locked python -m scripts.tools.prepare_rebuilt_labels --output-dir <scratch>/prepared --manifest-dir <scratch>/manifests --analysis-dir <scratch>/analysis
rtk proxy uv run --locked python -m scripts.tools.validation prepare --config configs/training.yaml --split-dir <scratch>/split
```

- **Public inputs:** the label manifest pins every input by SHA-256 but records Kaggle versions
  for only some. [`rebuild-public-source-downloads.json`](../../configs/labels/rebuild-public-source-downloads.json)
  gives the dataset, version and download command for all 19 public files; the 11 unversioned
  `downloaded-DATE` files were each matched to their current Kaggle version by download and
  SHA-256 on 2026-09-28.
- **JevK5 v3 attributes:** `outputs/labels/jevk5-attributes-v3/attributes.parquet` (`4e2eb3fc…`)
  is our own extraction and cannot be downloaded or regenerated bit for bit; the folder is
  read-only. JevK5 v4 must build a new label version (a new manifest and control), not replace v3
  under `rebuild_state_v2`.

## Status

- [x] Sweep, pinned downloads, lineage table, design and pre-registration (2026-09-28)
- [x] Family kappa, state model, tempering, calibration, tables, fold-4 check (2026-09-28)
- [x] B trained and submitted: S75 0.946, tie with S73
- [x] A trained: best epoch 9 against either label set (×1 0.897726 on the frozen targets);
  ×4 0.903713 against `combined_v1` (−0.0060, 95% [−0.0093, −0.0025]; Effusion and Baker's
  −0.028, Medial OA −0.011) and +0.0045 (95% [+0.0013, +0.0079]) against its own targets.
  Each model again wins on its own targets. Rank-averaged with the control it scores 0.90950
  against `combined_v1` (control plus B: 0.91131). **Public 0.947 (S76)** against 0.946 for S73
  and S75, bundle `08d56aac`, ×4, τ 0.75, batch 4: one rounding step up, despite −0.0060 locally
  against the frozen `combined_v1` targets.

- [x] A2 built (`rebuild_state_v2`) and adopted by user decision with the new split
  `all_5fold_scanner_v2` (fold 0, seed 1533697213)
- [x] Control `rebuild-state-v2-control` trained: ×4 0.928288 (×1 0.924637, epoch 9 of 10, 912 validation studies); per label from ACL 0.9556 to
  PF OA 0.8766, soft macro 0.7907
- [ ] Export and submit the control at ×4, τ 0.75, batch 4

Estimated effort: about 3–4 hours of agent work for the tables; each arm is then a 32-minute
fold-4 run and one submission.
