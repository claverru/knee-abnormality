# 2026-09-28 campaign comparisons re-scored on `rebuild_state_v2`

**Re-scoring every reproducible autoresearch comparison against `rebuild_state_v2` (the labels
that track the public board best, [comparison](2026-09-18-local-public-metric-comparison.md#2026-09-28-update))
changes 5 of 151 verdicts. The one flip with a decisive public check agrees with it: the
three-compartment OA detector, rejected locally, passes (+0.0022, 80% [+0.0006, +0.0039]) and had
scored 0.944 against 0.940 publicly. The other new passes are token-dropout variants whose
mechanism was nearly inert, and plane-contrast readouts, the only new retry candidate.**

## Method

[`rescore_campaign_comparisons.py`](../../../scripts/adhoc/rescore_campaign_comparisons.py) takes
each decided experiment under `outputs/autoresearch/`, its recorded fold, report groups and the
candidate and reference `validation_predictions.csv`, and first recomputes the recorded macro Δ
with the recorded targets; a comparison that does not reproduce to 1e-9 is skipped (6 early
sampling runs, scored before the ×4 predictions were saved). It then scores the same predictions
against the frozen `all_5fold_scanner_v2` targets with the same paired report-group bootstrap
(2,000 draws, seed 20260914) and the shared adoption rule. CPU only; output in
`outputs/analysis/campaign-rescore-rebuild-20260928/comparisons.csv`.

```bash
rtk proxy env CUDA_VISIBLE_DEVICES= uv run --locked python -m scripts.adhoc.rescore_campaign_comparisons --output outputs/analysis/campaign-rescore-new/comparisons.csv
```

## Result

151 comparisons; 5 passed on their recorded labels, 8 pass on `rebuild_state_v2`. The label
change moves Δ by SD 0.0015 (mean +0.0003), about one bootstrap standard error.

| Campaign | Idea | Recorded Δ [80% low] | Re-scored Δ [80% interval] | Re-scored |
|---|---|---:|---:|---|
| triplet-sampling | Run-level stratified jitter (adopted) | +0.0035 [+0.0013] | +0.0044 [+0.0024, +0.0065] | pass |
| plane-specialization | Dense plane readouts (adopted) | +0.0047 [+0.0024] | +0.0032 [+0.0010, +0.0053] | pass |
| knee-geometry | Half-knee MIL (adopted) | +0.0031 [+0.0015] | +0.0019 [+0.0002, +0.0037] | pass |
| near-misses | Token dropout, random (adopted) | +0.0025 [+0.0005] | +0.0037 [+0.0018, +0.0056] | pass |
| augmentation | Bilateral smoothing (adopted) | +0.0020 [+0.0002] | +0.0008 [−0.0009, +0.0027] | **fail** |
| near-misses | Three-compartment OA detector | +0.0013 [−0.0002] | +0.0022 [+0.0006, +0.0039] | **pass** |
| near-misses | Token dropout, top-attention | −0.0007 [−0.0029] | +0.0024 [+0.0003, +0.0047] | **pass** |
| near-misses | Plane-contrast readouts | −0.0006 [−0.0023] | +0.0018 [+0.0001, +0.0035] | **pass** |
| near-misses | Spatial-cluster token dropout | −0.0002 [−0.0018] | +0.0015 [+0.0000, +0.0030] | **pass** |

Part of each shift belongs to the campaign's references, not the ideas. The mean shift is
+0.0014 in near-misses (19 comparisons) and −0.0009 in augmentation (16), and within ±0.0007
elsewhere except the two small oblique campaigns. Net of the near-misses mean, the idea-specific
shifts are +0.0017 for top-attention dropout, +0.0010 for plane-contrast readouts, +0.0003 for
spatial-cluster dropout and −0.0005 for the OA detector.

## Reading

- **The OA detector flip matches the public board** (S58 0.944 against S55 0.940, decisive). It
  is already on main; this is one decisive check that the re-scored verdicts lean the right way.
- **Token-dropout passes are not evidence for the mechanism.** The adopted variant masked about
  0.3% of tokens ([near-misses campaign](2026-09-25-near-misses-campaign.md)), its public check
  was 0.939 against 0.940, and the variants share its rate and cap. A nearly inert change still
  moves the training path, so these Δs are training noise plus the near-misses reference shift.
  A token-dropout retry would need a realized masking fraction worth testing, which makes it a
  new idea rather than a retry.
- **Plane-contrast readouts is the one retry candidate:** passes narrowly and keeps +0.0010 after
  the campaign shift. Weak evidence: one run, one label re-score, no public check.
- **Smoothing falls below the rule,** largely through the augmentation campaign's −0.0009 shift.
  It stays on main inside the half stack, which gained publicly (S64 0.943 against S63 0.940).
- **The other adoptions hold** on the new labels.

## Main adoptions

The same re-score over the comparisons behind each main adoption
(`outputs/analysis/campaign-rescore-rebuild-20260928/with-main-adoptions.csv`, adding the manual
comparisons below with `--comparisons`). Class-balance k0.5 does not reproduce from its saved
predictions and class-balanced BCE's candidate predictions are gone; label, split and inference
adoptions have no paired comparison.

| Adoption | Comparison | Recorded Δ [80% low] | Re-scored Δ [80% interval] | Basis of adoption |
|---|---|---:|---:|---|
| Stratified jitter | triplet-sampling | +0.0035 [+0.0013] | +0.0044 [+0.0024, +0.0065] | Local pass, holds |
| Half-knee MIL | knee-geometry | +0.0031 [+0.0015] | +0.0019 [+0.0002, +0.0037] | Local pass, holds |
| Three-compartment OA detector | near-misses | +0.0013 [−0.0002] | +0.0022 [+0.0006, +0.0039] | Public (S58), now also local |
| Learned anatomical strengths | `anatomical-strength-vs-control-repeat` | +0.0023 [+0.0001] | +0.0001 [−0.0019, +0.0023] | **Local pass, lost** |
| | `anatomical-learned-vs-control-b` | +0.0009 [−0.0015] | +0.0016 [−0.0008, +0.0039] | |
| Bilateral smoothing | augmentation | +0.0020 [+0.0002] | +0.0008 [−0.0009, +0.0027] | **Local pass, lost** |
| DSU | augmentation (on smoothing) | +0.0022 [−0.0005] | +0.0011 [−0.0014, +0.0035] | Replicated near-miss, **replication lost** |
| | `dsu-without-smoothing-vs-report-attributes-control` | +0.0012 [−0.0006] | −0.0003 [−0.0020, +0.0015] | |
| Half stack | `half-stack-vs-smoothing-dsu` | +0.0011 [−0.0015] | +0.0020 [−0.0005, +0.0046] | Public (S64 0.943 against S63 0.940) |
| Report-attribute heads 3.1 | `report-attribute-heads-v31-vs-oa-detector-control` | +0.0010 [−0.0005] | +0.0007 [−0.0008, +0.0023] | Public (S62 0.940 against S60 0.938) |
| Robust centre and per-knee geometry | `merged-vs-adopted` | +0.0014 [−0.0008] | +0.0009 [−0.0013, +0.0033] | User decision at a fresh seed |
| Scanner confusion head | auxiliary-signals | −0.0008 [−0.0027] | −0.0002 [−0.0022, +0.0016] | Public (S67 0.948 against S68 0.947) |

No adoption turns harmful: the lowest re-scored Δ is −0.0003. Two adoptions that rested on a
local pass lose it (learned anatomical strengths, now +0.0001 against the repeated control, and
bilateral smoothing, +0.0008), and DSU's replicated near-miss no longer replicates (+0.0011 and
−0.0003). All three read as roughly neutral rather than wrong; removing one would need its own
removal run on the current recipe. The public-based adoptions stay where they were: none passed
the local rule before and none passes now.
