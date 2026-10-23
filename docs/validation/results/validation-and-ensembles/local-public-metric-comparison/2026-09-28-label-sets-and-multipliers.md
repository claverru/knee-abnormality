# Local metrics against the public leaderboard: label sets and multipliers (2026-09-28, S23-S77)

**2026-09-28 update, S23–S77 (41 single-checkpoint submissions in 8 comparable cohorts).** Within
a cohort, the public score's spread (SD 0.0012–0.0023) is about the size of pure seed noise, so
only differences of 0.003 or more say much. On those decisive pairs, scoring the saved validation
predictions against `rebuild_state_v2` agrees with the leaderboard on 78–83% of pairs, against
71–73% for `combined_v1`, the labels most of these runs were validated on. The ×4 and ×1
predictions track equally well on the same submissions. Gold-only AUC is below chance (38–46%),
again. Fixed labels cannot judge a change of training labels: each label set favors the models
trained on labels like itself by about ±0.005, more than the effects being measured.

## Method

[`correlate_submission_metrics.py`](../../../../../scripts/adhoc/correlate_submission_metrics.py) reads
the machine ledger (`outputs/submission/kaggle-ledger.json`), resolves each bundle's checkpoint
to its run, fold and epoch, and scores that checkpoint's saved predictions against every frozen
label set: `steven_v4` (`all_5fold_v3`), `combined_v1` (`all_5fold_scanner_v1`) and
`rebuild_state_v2` (`all_5fold_scanner_v2`). It also scores against their consensus (per label, only
the studies where all three agree on the hard label) and against the official labels of the
fold's gold studies. CPU only; artifacts in `outputs/analysis/l/local-public-metrics-20260928/`.

```bash
env CUDA_VISIBLE_DEVICES= uv run --locked python -m scripts.adhoc.correlate_submission_metrics --output-dir outputs/analysis/local-public-metrics-new
```

**Sources and cohorts.** `val_x<m>` is the per-epoch validation prediction of the submitted epoch
at the run's validation multiplier; `pred_x4` is the ×4 prediction saved for the selected
checkpoint, available for 28 submissions from S41 on; `own` is the metric recorded at training,
against the run's own labels. A cohort shares split, fold and inference setting (image count,
temperature, TTA); `own` metrics also share training labels. Ensembles (S35–S37, S71), the
temperature-folded checkpoint (S28, S29, S31) and S01–S22, whose run directories are gone, are
excluded. Cohorts: `all_5fold_v3` fold 0 at ×2 τ1 (6) and ×4 τ0.75 (5); scanner v1 fold 0 (19),
fold 1 (3), fold 2 (3), fold 3 (6), fold 4 (5); S47 (×5 τ0.7) stands alone.

**Statistics.** Public and metric are centered within each cohort before a pooled Pearson
correlation, with a permutation test that shuffles public scores within cohorts (20,000 draws).
Concordance counts within-cohort pairs ordered the same way, ignoring ties; decisive pairs differ
publicly by at least 0.003, the observed seed swing.

| Metric | n | Pearson (p) | Concordance | Decisive |
|---|---:|---:|---:|---:|
| ×4 against `rebuild_state_v2` | 28 | **+0.48 (0.008)** | 0.70 (86) | **0.78 (41)** |
| ×4 against `steven_v4` | 28 | +0.43 (0.040) | 0.66 (86) | 0.80 (41) |
| ×4 against the consensus | 28 | +0.48 (0.013) | 0.65 (86) | 0.76 (41) |
| ×4 against `combined_v1` | 28 | +0.37 (0.062) | 0.63 (86) | 0.73 (41) |
| ×4 soft AUC, `rebuild_state_v2` | 28 | +0.40 (0.053) | 0.65 (86) | 0.78 (41) |
| ×4 gold-only AUC | 28 | +0.16 (0.45) | 0.40 (86) | 0.46 (41) |
| ×1 soft AUC, `rebuild_state_v2` | 41 | +0.30 (0.075) | 0.64 (181) | 0.75 (80) |
| ×1 against `rebuild_state_v2` | 41 | +0.11 (0.52) | 0.57 (181) | 0.68 (80) |
| ×1 against `combined_v1` | 41 | −0.05 (0.79) | 0.48 (181) | 0.54 (80) |
| Recorded ×1 macro AUC (own labels) | 44 | −0.16 (0.39) | 0.41 (153) | 0.43 (70) |
| Recorded ×1 gold AUC | 44 | −0.06 (0.75) | 0.38 (152) | 0.38 (69) |

On the same 28 submissions, ×1 against `rebuild_state_v2` agrees on 0.66 of pairs and 0.83 of
decisive ones, ×1 against `combined_v1` on 0.58 and 0.71, and the recorded ×1 macro AUC on 0.55
and 0.63. The weaker ×1 rows in the table come from the earlier fold-0 submissions (S40–S47),
not from the multiplier.

**Label switches** (×4 Δ against each label set; public Δ in thousandths):

| Pair | Training labels | Public | `combined_v1` | `rebuild_state_v2` | `steven_v4` | Consensus | Gold |
|---|---|---:|---:|---:|---:|---:|---:|
| S73→S75 | combined → rebuild mean | +0.000 | −0.0010 | +0.0051 | +0.0017 | +0.0035 | +0.0088 |
| S73→S76 | combined → rebuild state | +0.001 | −0.0060 | +0.0039 | −0.0055 | +0.0014 | −0.0110 |
| S48→S50 | combined → image blend | +0.003 | +0.0035 | +0.0026 | +0.0081 | +0.0032 | −0.0057 |
| S51→S52 | combined → image blend | +0.001 | +0.0014 | +0.0017 | +0.0043 | +0.0011 | −0.0140 |

**Reading.**

- **Validate on `rebuild_state_v2`.** It tracks the leaderboard best of the three label sets at
  both multipliers, which supports the 2026-09-28 switch of the shared evaluation. Moderate-weak:
  28–41 submissions, about 30 metrics compared, and the rebuild was chosen partly on S75/S76.
- **The multiplier does not matter for tracking;** ×4 stays the adoption score for its lower
  prediction noise, not for better agreement.
- **The recorded ×1 macro AUC, which picks the checkpoint, tracks worst** (0.43 of decisive
  pairs overall, 0.63 on the recent 28). It is scored on each run's own training labels, mostly
  `combined_v1`.
- **Gold-only AUC stays below chance** with 11–12 fold studies; never use it to choose.
- **Label changes need a submission.** Each label set rates the S73→S76 switch in its own favor
  (`combined_v1` −0.0060, `rebuild_state_v2` +0.0039) while the public score moved +0.001. The
  consensus came closest in these four pairs, but only S48→S50 is decisive, where every label set
  except gold agreed with the public gain.
