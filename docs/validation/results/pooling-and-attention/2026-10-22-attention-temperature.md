# Attention temperature τ: board, gold and OAI (2026-10-20 to 2026-10-22)

**Answer:** for the 60- and 69-model ensembles at ×4, the public board is best at τ ≈ 0.4–0.45
(S188 at 0.4 first; S197 at 0.43 above S196 at 0.4). Local silver-scored validation always
preferred τ 0.85 because silver labels come from reports: sharpening moves predictions toward
what radiologists see in the images, exactly in the cells where report and image disagree, and
silver scores that as an error. Radiologist (gold) labels make sharpening roughly free; the board
rewards it. Widening Contusion and moving to ×8 did not help on the board.

Scripts and outputs: main checkout `outputs/tau-diag/` (`gold_tau.py`, `gold_gen1_mult.py`,
`oai_width_vs_n.py`, `h1_tokens.py`, `gen_widths.py`, `contusion_width.py`, `label_width.py`),
subagent scripts in `/tmp/claude-1000/tau-agents/` (not kept). How τ is applied:
[model bundle](../../../inference/model-bundle-and-offline-environment.md); it divides the main
attention scorer, view label bias and position prior, and multiplies the shared routing
temperature, so it also sharpens routing.

## Board results

Same members and settings within each block; ties ordered by Kaggle's hidden decimals.

| Ensemble | Setting | Submission | Public | Place among equals |
|---|---|---|---|---|
| 15 v3 gen2 | ×4 τ 0.5 | S185 | 0.967 | best of the 15 |
| 15 v3 gen2 | ×4 τ 0.3 | S186 | 0.966 | |
| 15 v3 gen2 | ×4 τ 0.85 / 0.15 | earlier | 0.965 / 0.963 | |
| 60 (v2+v3 × gen2+gen3) | ×4 τ 0.4 | S188 | 0.968 | 1st of seven 0.968s |
| 60 | ×4 τ 0.5 | S187 | 0.968 | 2nd |
| 69 (60 + 9 gen4) | ×4 τ 0.43 | S197 | 0.968 | 3rd |
| 69 | ×4 τ 0.4 | S196 | 0.968 | 4th |
| 60 | ×4 τ 0.4, Contusion row τ 0.93 | S192 | 0.968 | 5th |
| 60 | ×4 τ 0.6 | S189 | 0.968 | 6th |
| 60 | ×4 τ 0.35 | S194 | 0.968 | 7th |
| 60 | ×8, per-label τ at 0.55 × training width | S191 | 0.967 | |

- The 60-model peak lies between 0.35 and 0.5, with a steeper fall on the sharp side (0.35 ranks
  below 0.6). Seven curve shapes fitted to the rank order put the peak at 0.39–0.44; the best
  mean across shapes was τ 0.43–0.435, which S197 then confirmed against S196.
- Adding nine gen4 students (S196, S197) ranked below S188. Their teachers' predictions were made
  at τ 0.85 and they train mostly on them (share 0.8–1.0); that they carry wide-attention
  behaviour is a guess.
- Ensemble size moves the optimum very little: from the OAI fit below, 60 → 69 models shifts the
  gain of τ 0.35 over 0.4 by about 0.00001.

## Why local validation disagreed

### Gold was scored on silver labels

The 2026-10-21 analyses (`sub12_analysis.py:74`, `proxy_search.py:96`) scored the 58 gold studies
against the v3 silver table, not the radiologist labels in `train.csv`. Those "gold" lines in the
[teacher-student results](../teacher-student/2026-10-19-labels-v3-final-ensembles-and-students.md) of 2026-10-21 are silver numbers.

Only gen1 models can be scored on gold cleanly: gen2, gen3 and gen4 learned the gold labels
through their label tables and teachers. Each gold study is held out by three gen1 models.

| gen1, 3 held-out models, 58 gold studies, ×4 | τ 1.0 | 0.85 | 0.7 | 0.6 | 0.5 | 0.45 | 0.4 | 0.3 |
|---|---|---|---|---|---|---|---|---|
| Gold (radiologist) | 0.9511 | 0.9534 | **0.9542** | 0.9529 | 0.9521 | 0.9519 | 0.9508 | 0.9481 |
| Silver, same studies | 0.9187 | 0.9179 | 0.9137 | 0.9117 | 0.9086 | 0.9057 | 0.9027 | 0.8974 |

At ×8 the same models score about 0.002 lower at every τ (interval −0.005 to +0.001).

### Sharpening follows the image where report and image disagree

Same studies and predictions, τ 0.45 against τ 1.0, pairs split by whether gold and silver agree:

| Pairs | Gold | Silver |
|---|---|---|
| Both cells agree | −0.0034 | −0.0030 |
| At least one cell disagrees (111 cells, 16%) | +0.0041 | −0.0100 |

The gap (+0.0137, bootstrap 10–90% [+0.0013, +0.0268]) sits entirely in the disagreement cells.
A cell's rank change under sharpening is predicted by its gold label (+0.042, [0.023, 0.070]),
not its silver label (−0.009, [−0.027, +0.012]). Positives only radiologists mark rise in rank
(mostly Synovitis, 14, and PF OA, 8); positives only reports mention fall (Effusion 16,
Contusion 12, Lateral OA 8, Medial OA 7). Report-only positives come with mild wording.

- **Gold studies are sicker, but that is not the cause:** on the same 58 studies silver marks
  4.64 positives per study and gold 4.14; other studies average 2.11 in silver. Pairs between
  sick knees still lose −0.003 when the labels agree. Sicker knees only add disagreement cells.
- **Official criteria** (two MSK radiologists, borderline graded negative) are the same for the
  hidden test: [label criteria](../../../metadata/official-label-criteria.md).
- Silver strictness alone does not explain it: on OAI, looser grade cut-offs for OA favour
  sharpening and severe-only ones do not.

## Unseen data: OAI

1,335 OAI knees with MOAKS labels mapped to 9 targets, 15 gen1 models (never saw OAI), ×4,
mirror TTA. Macro AUC fits AUC(τ, N) = a(τ) − v(τ)/N within 0.0003:

| τ | 0.3 | 0.45 | 0.6 | 0.85 | 1.0 |
|---|---|---|---|---|---|
| a(τ), many-model limit | 0.9103 | 0.9153 | 0.9178 | 0.9179 | 0.9167 |
| v(τ), variance | 0.0288 | 0.0134 | 0.0090 | 0.0064 | 0.0056 |
| 1 model | 0.8810 | 0.9020 | 0.9089 | 0.9115 | 0.9111 |
| 15 models | 0.9082 | 0.9146 | 0.9174 | 0.9175 | 0.9163 |

Sharpening adds model variance that ensembling removes; the many-model limit is flat between 0.6
and 0.85. OAI's sharp side is less steep than the board's, so shifted OAI curves cannot fit the
board's order.

## Attention width

Width = participation ratio 1/Σw² of each label's attention. Training (×1, τ 1) widths are 40–205
tokens per label (gen1); ×4 at τ 0.85 attends about 2.5× that. Gen2 and gen3 attend about 30%
fewer tokens than gen1, but the τ that matches training width barely moves (median ×4: gen1 0.55,
gen2 0.52, gen3 0.53); at ×8 it is 0.37–0.61 per label for gen2 (ACL 0.40, MCL 0.37, menisci
0.43–0.44, OA 0.47–0.52, Contusion 0.61). Matching training width (the a-priori hypothesis H1)
fits OAI's many-model limit and sits near the board optimum, but does not explain the board's
gain on its own, and S191's equal per-label widths at ×8 lost a step.

A runtime per-label τ (`--label-attention-temperature`, one encoding pooled at several τ; branch `label-attention-temperature`, 2026-10-12) was never merged: the final bundles bake one τ per member, and both per-label rules tried on the board lost. Its uncommitted code is archived in the main checkout at `outputs/archive/removed-worktrees-20261023/label-temperature.patch`.

Routing temperature is one value shared by all labels, so a per-label τ only scales that label's
own scorer, bias and prior: Contusion's own row at τ 0.93 with routing at 0.4 reaches the width of
a uniform τ 0.85 (2.1× training).

## Contusion

On clean gold, sharpening hurts Contusion at every setting (−0.03 at τ 0.6, −0.057 at τ 0.45,
gen1) while silver barely moves, which suggested keeping Contusion wide. On the board, S192
(S188 with Contusion at τ 0.93) ranked below S188 and S187: the gold signal (19 positives) did not
transfer.

## Consequences

- Never tune τ on silver labels; gold (gen1 own-fold only) and within-burden comparisons are the
  valid local signals, and both are small (58 studies).
- For a 60–69-member ×4 ensemble use τ 0.4–0.43; ×8 did not pay.
- Per-label τ rules (equal widths, Contusion wide) both lost on the board.
