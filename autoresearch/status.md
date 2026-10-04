# Implementation status

Current state and pending work only. Finished changes live in git history; campaign outcomes live
in the [campaigns](campaigns/INDEX.md) and the
[validation results](../docs/validation/results/INDEX.md).

## Current state (2026-09-28)

- **No campaign is running.** Every campaign under `outputs/autoresearch/` is stopped, and the
  watchdog is stopped. The last one, [learnable-views](campaigns/learnable-views/INDEX.md),
  was stopped by the user on 2026-10-03 and its line closed; it ran from the `view-optimisation`
  branch's worktree, whose build is not on main.
- **Main recipe:** `configs/training.yaml` trains on the rebuilt labels `rebuild_state_v2` with
  seed 1533697213 on fold 0 of `all_5fold_scanner_v2`
  ([current model](../docs/modeling/current-model.md)); the evaluation gate in
  [`campaign/rules.py`](../src/knee_abnormality/autoresearch/campaign/rules.py) requires both, so
  campaigns sealed on `all_5fold_scanner_v1` or `combined_v1` cannot resume.
- **Fold:** the runner takes the fold from the sealed recipe instead of assuming fold 0
  ([protocol](protocol.md#training-seed-policy)).
- **Activation signature (done 2026-09-28):** prepared developer reports declare a
  logged metric, threshold and epoch; review binds it, and the supervisor checks it mid-run,
  stopping a failed run as `invalid` and returning it as a repair
  ([activation check](lifecycle.md#activation-check)). Not yet exercised by a campaign.
- **Topic rotation (done 2026-09-28):** the topic loop (`loop-start`, `loop-status`, `loop-stop`,
  `loop-resume`) rotates after `backlog.rotate_after` researcher runs without an adoption (six
  for drafted topics), takes topics from
  [the queue](topics/INDEX.md) or a checked researcher draft, and chains each topic on the
  incumbent ([topic rotation](topic-rotation.md)). Exercised only on saved records and a scratch
  copy with launches and agents stubbed; not yet run with a provider or GPU.
- **Loop watchdog, per-campaign rotation streak and label scope (done 2026-09-28):** the
  [watchdog](watchdog.md) started without `--campaign` follows the loop's current and successor
  topics and relaunches or repairs a dead loop process; `backlog.rotate_after` sets each
  campaign's streak (6 in every existing campaign and sealed into their stopped states); the
  split and evaluation targets moved to the protected `data/splits.py`, leaving training-target
  construction in `data/labels.py` editable by topics. Exercised on saved records and a scratch
  copy only; no watcher, loop or provider was started.

- **Per-campaign budget, configuration-only rule, run curves and loop recovery (done
  2026-09-28):** `campaign.yaml` now requires `budget` (`sealed`, or a wall-clock `range` of the
  reference's training time as the only budget rule, with epochs, batch size, steps and input per
  study free and the bounds enforced during and after training;
  [training budget](campaign-setup.md#training-budget)) and `policy.configuration_only_ideas`
  (configuration-only ideas allowed, each nonempty refill needing a `source_change: true` idea).
  Every existing campaign and its stopped state carry `budget: {mode: sealed}` and
  `configuration_only_ideas: false`. Policies can no longer permit validation-frame, split or
  label-source identity differences. Researcher refills receive `run_curves` and proposals declare
  `source_change` and `targeted_curve`. `loop-resume` restarts the watchdog that `loop-stop`
  ended, and the watchdog restarts a live loop stuck on one error for 30 minutes. Exercised on
  saved records, real run timings and a scratch loop directory only; the range budget and the
  stuck-loop restart have not run in a campaign.

## Pending

- **Harness gaps the mixed-field campaign exposed (2026-10-01), awaiting a user decision:**
  - an activation signature of `> 0` passed a mechanism that never acted (`two-stream-paired`,
    observed 0.0013); thresholds should separate an acting mechanism from numerical residue;
  - `data.study_slice_total` is a floor, not a cap, so a range-budget idea that lowers it to save
    time changes almost nothing; cutting input needs `max_images` in `data.series_types`;
  - a configuration-only idea cannot declare an activation column, so it blocks unless the idea
    text allows logging-only instrumentation;
  - developers have no way to estimate epoch time before launch; two paired-view builds were
    stopped at epoch 2 for the budget
    ([campaign](campaigns/mixed-field/INDEX.md#keeper-repairs)).
- **Next campaign reference:** `rebuild-state-v2-control` (fold 0 of `all_5fold_scanner_v2`, seed
  1533697213), ×4 0.928288, trained 2026-09-28; a new campaign, or the topic loop with
  `--reference-run`, can start from it.
- **Near-misses cannot resume:** its sealed recipe no longer validates against its historical
  control on the current main. Its outputs stay a read-only record.
- **Proposed runner changes, awaiting a user decision:** the lower-priority items of the
  [literature review](literature-mechanisms.md#eight-mechanisms-ranked-by-value-per-cost-for-us).
- **Seed pool:** plane-contrast readouts (near-misses `plane-contrast-readouts`) is the first
  entry of the [seed pool](topics/seed-pool.yaml): rejected at −0.0006 but +0.0018, 80%
  [+0.0001, +0.0035] when re-scored on `rebuild_state_v2`, +0.0010 net of the campaign's
  reference shift; weak, one run
  ([re-scoring](../docs/validation/results/2026-09-28-campaign-rescoring.md)).
- **Next campaign:** [RadImageNet](campaigns/radimagenet/INDEX.md), validated 2026-09-28, not started. Start it
  as the loop's first topic while the user is present (`loop-start --campaign radimagenet`), since
  the activation check, topic loop, loop watchdog and range budget have not run live.
