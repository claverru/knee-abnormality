# Disk cleanup and checkpoint pruning log

What each cleanup of checkpoints, export weights and caches removed and kept, in the order the entries were recorded. Cleanups ran through the [disk-cleanup skill](../../.agents/skills/disk-cleanup/SKILL.md), which holds the deletion rules, and [`free_disk`](../../scripts/tools/free_disk.py), which applies them and writes a manifest under `outputs/archive/maintenance/disk-cleanup-<UTC>/`; earlier ones were hand-reviewed removals. Weights of submitted models stay on Kaggle as private bundle datasets.

**2026-10-16:** the skill freed **1.59 GiB**
([manifest](../../outputs/archive/maintenance/disk-cleanup-20261016T172321Z/manifest.json)):
eight finished-session scratchpads (1.633 GiB listed), 11 unnamed Hugging Face models (1.2 GiB)
and the weights of 28 export bundles (0.394 GiB); `uv cache prune` removed 0.1 GiB more. The
fastMRI joint line's local runs were named in
[its record](../validation/results/teacher-student/2026-10-15-silver15-round-and-teacher-share.md#local-runs-to-keep-main-checkout-outputstraining)
first, and the six pretraining `last.ckpt` files (start weights of the joint runs, 2.6 GiB) were
moved aside during the run, since the resume-checkpoint rule ignores `docs/ongoing/`. Kept for a
decision: 30 selected checkpoints whose result is recorded nowhere (12.28 GiB), 6 runs without a
selection record (2.5 GiB), 7 export bundles (1.467 GiB). Outside the rules:
`~/.cache/knee-segmenters` (31 GiB, 21 of them `nnunet_data`), `~/.cache/uv` (24 GiB),
`outputs/analysis/pooling-token-cache` (23 GiB), closed campaigns under `outputs/autoresearch`
(23 GiB) and about 66 GiB that `du` without root does not see.
Then, by the user's decision, `~/.cache/knee-segmenters` (31 GiB, segmenter line not pursued; main's
`meniscus_targets` and `balance_anatomy_targets` need its environment and weights set up again),
`outputs/analysis/pooling-token-cache` (23 GiB) and the `source/`, `workspace/` and `jobs/` folders
of every stopped campaign under `outputs/autoresearch` (1,097 folders, 15.23 GiB; each experiment
keeps `changes.diff`, `training/source_snapshot.zip`, predictions, configs and states) were deleted
by hand: WSL free space went from 166 to 230 GiB.

**2026-10-06:** a cleanup limited to `outputs/autoresearch/fixes-and-removals/` (user
instruction: nothing outside the campaign), done by hand under the checkpoint rules, without
`free_disk` and without a manifest: **25 `.ckpt` files (4.25 GiB)** removed, 14 `last.ckpt` and
the selected checkpoints of 11 runs; the folder went from 5.5 to 1.3 GiB. Retained: the selected
checkpoint of the incumbent `slab-rule-and-positive-std-requeue` (the control) and those of
`centre-without-side` (S108) and `oa-detector-off` (S111), with every run's predictions,
metrics, configs and sources.

**2026-10-02:** the first run of the skill removed **180 entries (25.789 GiB)**
([manifest](../../outputs/archive/maintenance/disk-cleanup-20261002T171047Z/manifest.json)); free
space rose by 24.74 GiB while a training run was writing, and a separate `uv cache prune` removed
1.4 GiB more. 81 `.ckpt` files (9.262 GiB) remain under `outputs/`.

- **Checkpoints (16.606 GiB):** 50 `last.ckpt` of finished runs (8.889 GiB) and 42 selected
  checkpoints of recorded runs named neither in `docs/ongoing/` nor by a submitted bundle
  (7.717 GiB): the 28 remaining `smart-crops` and 11 `mixed-field` experiments,
  `plane-contrast-readouts`, `refined-rn50-scaled-backbone-lr` and `rn18-stochastic-pooling-g05`.
  The first two were retained on 2026-10-01 and are gone now.
- **Other:** 29 analysis arrays (5.454 GiB: the `token-cache-20260917`,
  `token-cache-allocation-20260917` and `site-probes-20260930` caches and two smaller files),
  ten cached Hugging Face backbones (1.396 GiB), 23 scratchpads of finished sessions (1.021 GiB),
  Codex build folders (0.866 GiB), the download chunk cache (0.375 GiB) and the wheels of 14
  unsubmitted bundles (0.071 GiB).
- **Retained:** the 25 source checkpoints of submitted bundles, the eight selected checkpoints
  named in `docs/ongoing/` (`no-anatomical-bias`, `control-check`, `budget-learnable-recipe` and
  the five view runs), and every submitted bundle's weights.
- **Kept for a decision:** two crashed `two-stream-paired` attempts and
  `views-budget-jitter-centre`, which have no selection record; the four
  `rn18-l2sp-no-confusion-folds-1-4` teachers, `attributes-v4-w005-candidate` and
  `attributes-v4-w02-candidate`, whose results are recorded nowhere in the docs; and
  `outputs/analysis/s/slice-budget-20261002/scratch/bundle-copy/model.ckpt`, outside any run folder.

The two cleanups below predate the skill and were user-authorized, hand-reviewed removals of
obsolete `.ckpt` files. Each manifest lists every removed and
retained path with its reason, size and file identity; all other artifacts, including predictions,
metrics, configs, source snapshots and submission manifests, were preserved, and the stopped
campaign states were unchanged.

Historical `selected_checkpoint.json` and bundle manifests retain their original provenance paths;
they do not establish that pruned weights are still available. Consult the latest manifest before
attempting to resume, re-export or rerun inference on a historical experiment. Its saved predictions
and metrics remain available for comparisons.

**2026-10-01:** removed **435 files (63.336 GiB)** and retained 84 (10.567 GiB)
([manifest](../../outputs/archive/maintenance/checkpoint-pruning-20261001T164846Z/manifest.json)).

- **Removed:** every `last.ckpt` of a completed run, retained runs included (170 files, 27.8 GiB);
  rejected, invalid and superseded experiments of the stopped campaigns (103, 17.6 GiB) and of
  `outputs/training` and `outputs/kaggle-training` (85, 13.3 GiB); and superseded submission
  exports with their two standalone sources (77, 4.6 GiB): timing, loading-arm and τ-sweep
  bundles and candidates that lost publicly.
- **Retained, selected checkpoint only:** the runs named in the
  [current model](../modeling/current-model.md#adoption-history) (`no-anatomical-bias`, both Refined
  ResNet-50 runs, `plane-contrast-readouts`, `rn18-l2sp-no-confusion` with its fold 1–4 teachers,
  `rn18-oof-blend`, `rn18-no-series-dropout`, `rn18-more-triplets`, `rebuild-state-v2-control`,
  `scanner-confusion-control`); the v1-split models at 0.947–0.948 public
  (`scanner-confusion-detached-head`, `half-stack-control`, `rebuild-state-candidate`); the ten
  2026-09-19 ensemble members; `control-check`; the three `attributes-v4` candidates; and the 31
  export checkpoints of those models, including `aux-site-bias-f3`, whose source was already gone.
- **Not reviewed:** `outputs/autoresearch/mixed-field` (22 files, 4.2 GiB), an active campaign.
- **Superseded retentions:** the held plane-coupling candidate with its parent, the
  attention-temperature cache parents, control A, control B and the reflection reference kept on
  2026-09-20 are gone; their campaigns cannot resume on the current split.

**2026-09-20:** removed **249 files (33.412 GiB)** from completed superseded or rejected
experiments, unused resume checkpoints and stale submission exports, retaining 48 (4.381 GiB)
([manifest](../../outputs/archive/maintenance/checkpoint-pruning-20260920T191234Z/manifest.json)).
Windows C: free space did not grow after either cleanup unless the WSL virtual disk is compacted.

**2026-10-04:** by hand at the user's request, from the stopped `learnable-views` campaign only:
removed **86 checkpoints (14.72 GiB)**, every `last.ckpt` and the selected checkpoint of every
run that is not needed, retaining 10 selected checkpoints (1.70 GiB)
([manifest](../../outputs/archive/maintenance/learnable-views-checkpoints-20261003T220314Z/manifest.json)).
Kept: the campaign reference `hand-views-frozen-with-label-bias` (S100), the five members of S99
(one of them S98), and the four runs above the reference at ×4 (size menu, label depth bias,
label bias at ten times the rate, protected-strata depth transport). Predictions, metrics,
configs, source snapshots and the campaign state are untouched.

**2026-10-04, 22:14 CEST:** `scripts.tools.free_disk`, run after the `refined-radimagenet`
campaign stopped: **19.22 GiB freed**
([manifest](../../outputs/archive/maintenance/disk-cleanup-20261004T201421Z/manifest.json)). The
user asked for the campaign's checkpoints; the script applies its rules to every worktree, so it
also removed what the rules covered elsewhere.

- *From the campaign (6.43 GiB):* 19 `last.ckpt` (3.51 GiB) and 16 selected checkpoints of
  recorded runs (2.92 GiB). Kept: the incumbent `radimagenet-input-statistics` (named in
  `docs/ongoing/`), the two manual slice-teacher runs (results not in the campaign state) and the
  two cancelled runs without a selection record (`crossd-3d-stem`, `acs-gated-stem-block1`).
- *Outside the campaign, by rule (12.79 GiB):* 29 `last.ckpt` of finished runs in
  `outputs/training` (7.39 GiB); 21 selected checkpoints of recorded runs no longer named in any
  `docs/ongoing/` (3.46 GiB): **the ten `learnable-views` checkpoints retained by hand earlier the
  same day** (reference S100, the S99 members, four runs above the reference), the ReXNet-200
  folds 1–4 and `rn18-l2sp-no-confusion` folds 1–4 teachers, `attributes-v4-w005` and `-w02`
  candidates and `budget-learnable-recipe`; 41 export folders of unsubmitted models (1.82 GiB);
  three unused Hugging Face models (0.11 GiB); ten finished-session scratchpads (0.05 GiB).
  Submitted bundles in the ledger and their source checkpoints are untouched.
- *Failed:* one `.ruff_cache` inside a Codex debate (permission denied, 4 KB).
- *Kept for a decision:* six `model.ckpt` outside a fold folder (slice-budget scratch copy and
  the `ens5-learnable-views-f0` dataset), four `mixed-field` two-stream checkpoints and the two
  stopped ResNet-50 heated runs (no selection record), the ten `oof-blend-v2` children, the
  ResNet-10t reference, five `learnable-views`-era runs in `outputs/training` and the
  `oof-blend-v2` export members (results not recorded in `docs/`).
- *uv cache:* `uv cache prune` was skipped: another uv process held the cache lock.

**2026-10-05, about 12:10 CEST:** removed by hand on the user's request, outside the script:
the 72 checkpoint files of the `pretraining-input` campaign (36 runs, 5.7 GiB) and six of the
seven left in `refined-radimagenet` (the two manual slice-teacher runs and the two cancelled
runs). Kept: the `radimagenet-input-statistics` incumbent, the reference of the ResNet-18
comparisons.

**2026-10-06, 23:21 CEST:** removed by hand on the user's request, outside the script: the 48
checkpoint files of the stopped `side-routing` campaign (24 runs, 8.16 GiB) and 58 of the manual
side-routed runs of 2026-10-05 and 06 in `outputs/training` (29 runs, 10.25 GiB), the stopped
ResNet-50 attempt among them. Kept: the reference `20261006-163650-side-routed-spot-frame-ring-sectors`
and `20261006-210555-rn50-side-routed-reference`, which was training. The seven bundles
submitted from this line (S112 to S118) keep their weights in `outputs/submission/exports/`.
Manifest: `outputs/archive/maintenance/disk-cleanup-20261006T212117Z/manifest.json`.

On 2026-10-07 13:28 CEST the user had the checkpoints of the routing-and-shifts campaign removed: 42
files in 21 runs, 7.16 GiB. The two checkpoints of its incumbent
(`height-cuts-from-the-spots-requeue-20261006T232351Z-d2445402`) are kept, as the reference of the
frame-and-targets campaign. Predictions, logs, frozen source and diffs of every run remain.
Manifest: `outputs/archive/maintenance/disk-cleanup-20261007T112812Z/manifest.json`.

On 2026-10-09 07:03 CEST the disk-cleanup rules freed 15.75 GiB, plus 0.38 GiB from `uv cache prune`,
after the Windows drive holding the WSL virtual disk filled up (830 MiB free) and froze WSL twice
during the view-poses campaign: checkpoints 12.01 GiB in 70 files, export weights
0.2 GiB in 9, analysis arrays 3.53 GiB in 11, one Codex build folder and one
session scratchpad. Kept for a decision: the checkpoints and exports whose result is recorded
nowhere or whose run has no selection record (92 checkpoint and 195 export entries kept in all,
with their reasons in the manifest). One `.ruff_cache` under `~/.codex/debates` could not be deleted
(permission denied). The view-poses checkpoints stay: its campaign is running.
Manifest: `outputs/archive/maintenance/disk-cleanup-20261009T050321Z/manifest.json`.

On 2026-10-09 10:03 CEST the user ordered a deep cleanup beyond the rules, 86.9 GiB in all, because
the Windows drive holding the WSL virtual disk was full. Every checkpoint went (18.59 GiB) except
the best epoch of `sagittal-fluid-canvas-fine`, the view-poses incumbent
(`outputs/autoresearch/view-poses/experiments/sagittal-fluid-canvas-fine-20261009T054235Z-c95da7ce/training/fold_1/best-epoch=08-val_macro_auc=0.91939.ckpt`),
and every local bundle (7.20 GiB): the submitted models stay on Kaggle as private bundle datasets,
inference weights only, and runs never submitted can no longer be exported. Also removed: the
package caches the campaign developer jobs left in `jobs/*/scratch/cache` (37.38 GiB), the campaign
developer workspaces except the one claimed in view-poses (8.88 GiB; the frozen `source` of every
run stays), the `.venv` of the `knee-abnormality-jevk5-attributes-v4` worktree (7.43 GiB),
`outputs/previews` and `outputs/profiling`, every file of 1 MiB or more under `outputs/analysis`
except the campaign monitor (3.09 GiB; smaller tables and notes stay), and the segmenter cache's
benchmark folder and weights zip. Predictions, metrics, logs, diffs, the submission ledger and
records remain. Manifest: `outputs/archive/maintenance/deep-cleanup-20261009T080304Z/manifest.json`.

The same day, at the user's request, two more removals outside the rules: the build output and
scratch environment of the `variopinta` project (`rust/target`, `benchmarks/.runs` and one scratch
`.venv`, 6.8 GiB), and the converted input images of the segmenter cache
(`~/.cache/knee-segmenters/{sample550,val,dess-menisci,balanced750}/images`, about 16 GiB). The
segmenter masks, their case lists and the prepared targets stay; segmenting again means converting
the raw series again.

**2026-10-13, targets-and-inputs campaign (user order: keep only the incumbent).** 57 checkpoints
of `outputs/autoresearch/targets-and-inputs/experiments` were deleted, 21 GB: `last.ckpt` and the
selected checkpoint of every run except the incumbent's selected one
(`average-pool-shortcuts-in-the-resnet-20261013T122134Z-1fc0acea/training/fold_2/best-epoch=09-val_macro_auc=0.93090.ckpt`),
by the rules of `free_disk` applied to that campaign alone. The cleanup script was not run across
the worktrees: with the submission ledger stale since 2026-10-03 its export rule would have
removed the wheels of bundles another session was still using. Predictions, metrics, configs,
sources and the campaign state are kept; the weights of S150 and S151 are on Kaggle.

**2026-10-20, before closing the rented machine (user order, by hand).** WSL use fell from 770 to
694 GB: 64 export bundles in `outputs/submission/exports` (old ensemble probes A–D and the 5- and
6-family resplits, the efficiency probes cleared by the inference-speed session, the v3 gen-2
bundles at τ 0.15/0.3/0.5), the checkpoints of the v2 gen-1 runs of all six families,
`nextvit-silver-fold-*` and `overnight-*` (23 files), 55 parity checkpoints in
`outputs/rented-machine-archive-20261020/parity`, 13 TensorRT engine builds in
`outputs/profiling`, and `uv cache clean` (26.8 GiB). The dry run of `free_disk` was not executed:
its rules would have removed the pretraining `last.ckpt` files the students start from and the
wheels of the four candidate bundles, both recorded only in the fastmri-pretraining worktree's
ongoing notes. Kept: the four candidate bundles, the v2/v3 gen-2 and gen-3, v3 gen-1 and six
pretraining checkpoints, `outputs/autoresearch` records, the torch hub cache (RadImageNet weights).
Deleted paths: `outputs/archive/maintenance/manual-cleanup-20261020/`.


**2026-10-23, after the competition (user order).** WSL use fell from 752 to 689 GB. `free_disk`,
with runs named in the [final solution](../final-solution.md#runs-to-keep) protected and the
pretraining `last.ckpt` files kept by rule, freed **3.75 GiB**: the resume checkpoints of the nine
gen-4 and S188-student runs, the wheels of two bundles and finished-session scratchpads
([manifest](../../outputs/archive/maintenance/disk-cleanup-20261023T180503Z/manifest.json)). By
hand: the eleven ensemble bundles still holding weights (S194, S196 and S197, which are on Kaggle,
the rebuilt S192 copy and seven never submitted; every weight is a copy of a kept member
checkpoint), the `experiments` folders of every closed campaign in `outputs/autoresearch` (state
and notes kept), `outputs/analysis`, `outputs/profiling`, `data/raw/.download`, the pip, triton,
inductor and uv caches, old codex debate and review scratch, and the source caches of backbones
outside the final solution. Kept: every run named in the final solution, the six pretrainings with
their `last.ckpt`, the code-only bundles (S188's among them), `outputs/logs` (it holds the local half
of the final chain's scripts), `outputs/labels-v3-reading`, the rented-machine archive and the raw
competition data. Deleted paths: `outputs/archive/maintenance/manual-cleanup-20261023/`.
