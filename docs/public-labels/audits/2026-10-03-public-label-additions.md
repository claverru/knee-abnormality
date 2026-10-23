# 2026-10-03 public label additions

**Nothing new is worth adding to the label rebuild.** A Kaggle sweep found five label tables
published or changed since the [09-28 sweep](../label-rebuild/design-and-sources.md#kaggle-sweep-2026-09-28). Two are
new independent report extractions, and both read reports worse than `rebuild_state_v2` and
Steven v4 on our blind-judged cells. The other three are copies or derivatives of roots already
held.

- **Search:** Kaggle datasets matching knee, RSNA, report, labels, pseudo, soft, weak, LLM,
  teacher, annotation, gold, OOF and distill, sorted by update date, two pages per term; kept
  those updated from 2026-09-28. Forum and notebooks were not re-read.
- **Snapshots:** `data/labels/public/<owner>/<slug>/downloaded-20261003/`.
- **Checks:** scratch analysis, not a maintained script. Gold macro AUC on the official 58
  studies; binary kappa at 0.5 against held roots on silver; accuracy on the 371 report cells
  judged blind in the [source audit](../label-rebuild/source-audit.md#source-audit-result-2026-09-28) (POS and
  HEDGED positive; NEG, LOW and NOT_MENTIONED negative).

## Tables

| Dataset | What it is | Verdict |
|---|---|---|
| `sebgarciavillalobos/rsna-knee-report-annotations` (Oct 1) | 4,349 silver studies, no gold rows. Per label: `present` / `uncertain` / `absent` / `not_mentioned` plus positive and negative report spans. Extractor undocumented. | New root (kappa 0.81 with Lixin, 0.83 with Pilkwang v2, below the 0.90 family threshold), but weak: see below. Not added. |
| `gperdrizet/rsna-knee-abnormality-pseudolabeled-training-data` (Oct 3) | 4,405 studies, binary calls from Qwen2.5-7B and Phi4-14B (three runs each, majority, then AND; Phi alone for PF OA and Fracture, Qwen alone for Effusion, Synovitis and Contusion), agreement share and evidence spans. Gold copied into the 58 gold rows; the gating was chosen on gold. | New root (kappa 0.77 with Lixin and Steven v4), weak. Not added. |
| `tranphunghia233871/rsna-knee-labels-recalibrated` (Sep 30) | 4,407 studies, per-label monotone remap of Steven v1 (binary kappa 0.85, Spearman 0.92, 0.5 kept for unaddressed cells). | Steven v1 derivative. Gold macro 0.8365 against Steven v4 0.8927. Excluded. |
| `dhiyaneshbabu/rsna-knee-labelled-train` (Oct 1) | `labelled_train.csv` is byte-identical to the 09-28 copy. New `labelled_train_2_softavg_synfill.csv` averages it with another table (Spearman 0.95 with Lixin) and fills Synovitis to a 50% positive rate; gold copied. | Derivative of a held root. Excluded. |
| `ctogaurav/rsna-knee-llm-labels-v4-blend` (Sep 30) | Byte-identical to Steven v4. | Copy. |

Accuracy on the 371 blind-judged report cells:

| Source | Accuracy | Without HEDGED (n 260) | AUC |
|---|---:|---:|---:|
| `rebuild_state_v2` (the label table then) | 0.887 | 0.862 | 0.950 |
| Steven v4 | 0.811 | 0.750 | 0.919 |
| Lixin | 0.798 | 0.735 | 0.859 |
| Seb, `uncertain` as negative | 0.752 | 0.773 | 0.775 |
| Gperdrizet | 0.733 | 0.712 | 0.734 |
| Seb, `uncertain` as positive | 0.695 | 0.612 | 0.775 |
| Pilkwang v2 | 0.679 | 0.546 | 0.849 |

The cells are disagreements among the roots held on 09-28 and that audit shaped
`rebuild_state_v2`, so the comparison favours `rebuild_state_v2`; it is still the only
report-truth panel on silver. Seb marks 32 of 134 below-threshold (LOW) cells `present` and 39
`uncertain`, the same over-call of mild findings that separates silver from gold. Gperdrizet's
own gold confusion tables show the same pattern (Qwen: Effusion 13 false positives, MCL 9;
Phi: Effusion 15, MCL 7), no better than Steven v4's 16 and 7.

## The Seb "audited label corrections" are not corrections

`audited_label_corrections.parquet` holds 64 cells on 40 studies, **all gold studies**, and every
one contradicts the official label: 58 turn a gold 0 into 1 (14 Effusion, 5 Baker's, 6 OA cells
per compartment and others) and 6 turn a 1 into 0. They are a report reader disagreeing with the
image-based labels, mostly on mild findings, which is the
[gold-versus-silver gap](../gold-versus-silver.md) again. The test set is labelled like gold, so
they must not be applied.

## Label quality against the publisher's leaderboard position

**No usable correlation: label quality does not predict where its publisher sits.** Public
leaderboard pulled 2026-10-03 10:19 UTC (4,971 teams), joined by Kaggle username; a team's score
is credited to every member. Scratch analysis.

- **Cohort:** 47 publishers held under `data/labels/public/` or named in earlier sweeps; 40 are
  on the board. Not on it: Ramit, Josh, Bhavesh, Gperdrizet, Giovanni, Juanicarlos, Malikahsan.
  Correlations use the 16 root extractions on the board (15 with all 371 blind cells, 11 with a
  benchmarkable gold score); derivatives, copies and image-model tables are left out because
  their quality is another publisher's.
- **Not matched:** nothing says a publisher trains on their own table, and model, ensemble size
  and effort differ freely. 0.941–0.943 is the public-notebook level (board 75th–90th
  percentile), so scores there say little about the publisher's own work.

| Quality measure (roots) | n | Spearman with LB score | p |
|---|---:|---:|---:|
| Accuracy on the 371 blind-judged cells | 15 | +0.27 | 0.34 |
| Gold macro AUC | 11 | +0.24 | 0.48 |
| Kappa with `rebuild_state_v2` on silver | 16 | +0.05 | 0.87 |
| Silver positive rate | 16 | −0.08 | 0.76 |
| Submission count | 16 | +0.73 | 0.001 |

Submission count correlates +0.50 with score across the whole board, so the one clear signal is
effort, not labels. Restricting to roots with at least 20 submissions (n 7, LB 0.932–0.955)
gives +0.49 for blind accuracy and −0.03 for gold AUC, both unresolved at that size.

Only four publishers are above the public-notebook level: Laymond (rank 45, 0.955, 245
submissions, Qwen2.5-32B root with blind accuracy 0.757), Nartaa (rank 132, 0.948, Gemini 3
Flash root, the best blind accuracy at 0.900), Dreaddevelopment and Achelijn (0.945). The best
gold or blind tables otherwise belong to low-ranked accounts (Dhiyanesh 0.841 blind at 0.926,
JEV text at 0.897). By kind, root publishers have a median score of 0.940 (3 of 16 at or above
0.945) and derivative publishers 0.920 (0 of 18): people who ran their own extraction rank
higher than people who merged others' tables.
