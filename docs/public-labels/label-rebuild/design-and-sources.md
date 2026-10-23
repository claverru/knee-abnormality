# Label rebuild: design and sources

**Adopted 2026-09-28 (user decision): A2 (`rebuild_state_v2`) with the new split
`all_5fold_scanner_v2` became the recipe's label source**; it was built the same day in the
`qwen-lateral-labels` worktree, since merged into `main` and removed. Goal: the best training labels
obtainable from every published Kaggle report-label table (plus our own JevK5 attributes), built as if
no labels existed yet. Until the adoption, main kept `combined_v1`. The later `rebuild_state_v3` and
the full table lineage are in [label tables](../label-tables.md); the other parts of this record are
[state and mean arms](state-and-mean-arms.md), [source audit](source-audit.md) and
[training and A2](training-and-a2.md).

## Target definition

The training target is **P(test-style label = 1 | report)**. Test labels are image-adjudicated by
two MSK radiologists with a third adjudicating; borderline findings are graded negative; a report's
silence is an unobserved label, never an absent one
([criteria](../../metadata/official-label-criteria.md), host text retrieved 2026-09-17, rechecked
against this design 2026-09-28: no contradiction). Two error sources need different fixes:

1. **Extraction noise** (one source misreads a report): aggregation across independent sources,
   fitted on report data without labels.
2. **Shared semantic mismatch** (what silence, hedging, below-threshold grades and compartment
   wording mean for an image label): every source shares it, so only gold can inform it, through
   a low-dimensional mapping pooled across targets.

Unaddressed cells are about a quarter of all cells (Steven v1 unaddressed: Synovitis 3,689,
Baker's 2,126, Fracture 1,890, Lateral OA 1,090, Contusion 1,058, Medial OA 896 of 4,407), so their
valuation is probably the largest single decision
([encodings](../uncertainty-and-confidence-encoding.md)).

## Sources

### Kaggle sweep, 2026-09-28

Searches as in the [09-27 investigation](../audits/2026-09-27-public-label-investigation.md)
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
| JevK5 v3 (**ours, not a Kaggle table**) | Qwen3.5-4B LoRA, 40 questions | [extraction](../jevk5-report-attribute-extraction.md) | status with not-mentioned, hedging, grades, extent |

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
  [combined labels](../combined-labels.md#alternatives-measured-and-rejected)); this
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
  ([report attributes](../report-attributes.md#gold-rate-by-report-call)); the
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
