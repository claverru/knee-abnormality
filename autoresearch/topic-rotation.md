# Topic rotation

The topic loop keeps the GPU busy without the user: when a topic stops paying, it moves to a new
topic on its own and carries the incumbent across. Only the user stops autoresearch, with
`loop-stop`; there is no cap on topics and no automatic end. Evidence for the rule and its
parameters is in the
[literature review](literature-mechanisms.md#1-a-six-run-researcher-streak-ends-the-campaign); the
[activation check](lifecycle.md#activation-check) keeps runs whose mechanism never acted out of the
streak. Commands and recovery are in [runtime operation](runtime.md#topic-loop).

## Process model

One detached loop process runs above the per-campaign supervisors
([`topics/loop.py`](../src/knee_abnormality/autoresearch/topics/loop.py)). Each topic is an
ordinary campaign under `autoresearch/campaigns/<id>/` with its own supervisor, seal, GPU lease,
roles and gates; everything a campaign does stays as it is. The loop never trains, reviews or
decides a run. Every 30 seconds it reads the campaigns' `state.json`, and it acts on them only
through files their supervisors already obey: `stop.json`, `resume.json` and a `rotation.json`
mark (below). Its two agent assignments, the researcher's topic draft and the orchestrator's
pick, run through the campaign job worker with the same provider, profiles, sandbox, usage
capture and provider-wait handling. Its state is `outputs/autoresearch/.loop/loop.json`. The
[watchdog](watchdog.md), started for the loop, follows its topics across rotations and
relaunches a dead loop process.

## Trigger

A topic rotates after **`backlog.rotate_after` consecutive researcher-origin decided runs
without `provisional_accept`**
([`topics/history.py`](../src/knee_abnormality/autoresearch/topics/history.py)). The value is a
required per-campaign setting, sealed into the campaign's runtime state at its start. The loop
writes 6 into every researcher-drafted topic, the value the
[literature review](literature-mechanisms.md#1-a-six-run-researcher-streak-ends-the-campaign)
replayed; a queued topic sets its own.
Decided runs are taken in launch order. A `provisional_accept` resets the streak, whatever the
idea's origin, including a user adoption. An `invalid` run counts when one of its run checks
failed (the activation signature, or the training-time budget, including an early projection
stop): the idea itself could not produce a valid run. Other `invalid` runs, such as a failed
comparison or a lost worker, are infrastructure failures and are skipped. Seed ideas neither count nor reset: an idea is a seed when its ID was imported through
the campaign's `imports/` (startup seed files and `import-ideas`, forced or ranked) or it carries
`seeded`. Every other decided run counts. The loop only rotates a `RUNNING` topic.

## Rotation

| Phase | Old topic (`rotation.json`) | New topic | Ends when |
|---|---|---|---|
| sourcing, drafting, picking | `retiring`: no new claims (training repairs included) and no research refills; claimed and prepared work still develops, is reviewed, launches and trains | not yet created | a topic is accepted |
| successor | `retiring` | sealed and started with the mark `successor`: calibration, seeds, development and review run, launches wait | its first candidate is approved |
| handover | `handover`: no launches; when no training runs and no completion is pending, the loop requests a normal stop | `successor` | the old topic is `STOPPED` |
| commit | mark removed, topic retired | mark removed, it takes the GPU | at once |

The old topic's last running training therefore finishes and is decided before the new topic
launches, so the chain never skips a result. The attempt is **cancelled** when:

- the old topic's incumbent changes before the commit: an adoption resets its streak;
- the new topic pauses before its first approval or fails to start;
- three consecutive draft or pick attempts fail (every draft rejected, an invalid report or pick,
  a failed agent call). Provider limits and transient errors wait as in a campaign instead.

On cancellation a created new topic is stopped with abort and recorded `superseded` (its
campaigns-index row says so), the old topic's mark is removed, the loop resumes it if it had
stopped it, and the next attempt waits for one more decided run there. A topic that fails to
materialize therefore never stops the old one.

## Topic sources

1. **User queue.** Topic directories in [`topics/queue/`](topics/INDEX.md), taken in name order,
   any time. The loop never waits for the queue. A queued topic is checked only for validity: it
   is built into a campaign and must pass the same validation as `validate`, the protected list
   included. One that fails moves to `autoresearch/topics/rejected/<name>-<time>/` with
   `rejection.txt`, and the next source is tried. Its campaign ID is the directory name, with
   `-2`, `-3` appended when taken; the directory leaves the queue when its topic commits.
2. **Researcher draft.** With an empty queue the researcher drafts two or three topics from the
   [topic packet](#topic-packet). Code checks each draft; valid ones are written as topic
   directories under `.loop/rotations/<rotation>/drafts/<job>/<draft>/`. One valid draft is taken
   directly; with several, the orchestrator picks one, and a pick outside the valid drafts is
   rejected. A draft is copied into `autoresearch/campaigns/<id>/` and the
   [campaigns index](campaigns/INDEX.md) only once accepted. Drafted topics rank their seeds
   (`seed_order: ranked`), refill below 3 to 5, rotate after 6 and reject configuration-only
   candidates (`configuration_only_ideas: false`).

The **seed pool**, [`topics/seed-pool.yaml`](topics/seed-pool.yaml), holds loose seed ideas not
tied to a topic, in the topic-seed schema plus an `evidence` line. Every draft must take each
untaken pool seed whose `source_paths` and `config_prefixes` fall inside its scope. A pool seed is
taken once a committed topic, of any origin, has it among its seed ideas.

## Chaining

Each new topic is sealed on the chain incumbent, the old topic's incumbent at the trigger. Its
`campaign.yaml` carries a `chain` block (predecessor, incumbent ID, source, run, prediction hash);
its recipe is the incumbent run's `effective_config.json`, taken as recorded because the
incumbent's own configuration schema produced it, and its reference run is that run, with no
equivalences or exceptions. Validation checks that the chain is the predecessor's recorded
incumbent with unchanged predictions and sealed limits, and that epochs, optimizer steps and
bootstrap settings are unchanged. The per-experiment runtime ceiling is inherited from the
predecessor, so a slower adopted incumbent never raises it. The baseline source is the
incumbent's frozen source snapshot with the new campaign's directory added, the seed is the
incumbent's recorded seed and the fold is its recipe's fold; the baseline replay must reproduce the
predecessor's sealed triplet, visit and cohort counts. A chained topic keeps its predecessor's
[training budget](campaign-setup.md#training-budget), written into its `campaign.yaml` by the
loop: with a range budget it inherits the absolute training range and allowance, may start from
an incumbent with a different epoch count, batch size or input per study, and must reproduce
only the cohorts. Switching modes needs a
new chain started by the user; a first topic started with `--reference-run` has a sealed budget. The frozen source carries the chain's first
topic's evaluator, scripts and the harness code that runs inside jobs (calibration, training
accounting); changes to those on main reach a chain only when it restarts from main, while the
supervisor and loop themselves run from the main checkout. No control is retrained and nothing is merged into main: merges
and fold rotation stay manual and outside the loop.

The first topic starts as a campaign does today: either an existing campaign (`--campaign`,
started or resumed if needed) or the main recipe on a given reference run (`--reference-run`),
with the first topic then taken from the sources above.

## Scope

A topic may declare any Python file under `src/knee_abnormality/` and `configs/training.yaml`
except the protected list, which every campaign policy enforces
([`campaign/definition.py`](../src/knee_abnormality/autoresearch/campaign/definition.py)):

- **Sources:** the autoresearch harness (`autoresearch/`), the evaluator (`evaluation/`), the
  input-identity implementation (`identity.py`, `training/checkpoints.py`) and the split and
  evaluation targets (`data/splits.py`: `load_split`, `load_evaluation_targets`, `FrozenSplit`,
  the study-ID checks and the `STUDY` and `TARGET_THRESHOLD` constants). Scripts, dependencies
  and docs are outside `src/` and never editable.
- **Config paths:** `name`, `seed`, `paths`, `labels`, `validation` and
  `data.validation_image_multiplier`, and `trainer.max_epochs`, `trainer.batch_size` and
  `trainer.accumulate_grad_batches` unless the campaign's budget is a wall-clock range, where they
  may vary within the policy's permissions ([training budget](campaign-setup.md#training-budget)). The runtime
  limits are also sealed.
- **Input identity:** a policy cannot permit differences in the validation frame, split or label
  source identity keys (`validation_rows_sha256`, `membership_sha256`, `split_protocol_sha256`,
  `label_manifest_sha256`, `label_source`).

Training-label code is in scope: training-target construction in `data/labels.py` (pinned
label-source loading, the gold/silver fold targets), target processing and losses on targets.
The evaluator and the comparison read the frozen split files, directly or through
`data/splits.py`, never the training frames, so label work cannot change what a run is scored
against. The fold's validation frame, which checkpoint selection reads, is still assembled in
`data/labels.py`; the comparison always rejects a changed one through the `validation_rows_sha256`
input identity, which no policy may list among its permitted differences. The label source
itself stays the recipe's: the config key `labels` is protected, and switching it is a separate
harness change.

## Topic packet

Built by code for each draft ([`topics/packet.py`](../src/knee_abnormality/autoresearch/topics/packet.py))
and saved as `packet.json` in the rotation directory; floats are rounded to six decimals.

- `incumbent`: macro and per-label ×4 AUC from its saved predictions (reproducing the recorded
  prediction AUC), positives, and error slices: `vendor` (per-label AUC per vendor), `plane`
  (studies with the complete recipe series set, and studies missing each recipe series type or
  plane) and `compartment` (medial/lateral margin AUC and cross-talk for meniscus and OA), and
  `curves`, its per-epoch curves ([run curves](records.md#run-curves)).
- `campaigns`: every campaign's family ledger (attempts, decided results, verdicts) with its
  evaluation split, labels and fold; `closed_families`.
- `near_misses`: rejected runs with a positive macro delta whose family has no attempt against the
  current incumbent run.
- `topics`: every campaign with its scope, origin (loop record, else `user`), status, evaluation,
  families and outcome.
- `codebase_map`: modules and config sections no past policy included, and the protected lists.
- `seed_pool`: the untaken pool seeds.

## Checks that reject naive topics

Code-enforced on every researcher draft
([`topics/checks.py`](../src/knee_abnormality/autoresearch/topics/checks.py)); the reasons go back
to the researcher and the old topic keeps training meanwhile.

- **Grounded:** at least one weakness cited as a packet JSON pointer and value; each pointer must
  resolve and match at the cited precision (at least three decimals, integers exactly). The
  generated contract lists them.
- **New:** the scope is not wholly inside one past topic's (every file, directory and config path
  covered), the mechanism families are not all among one past topic's, every family has a seed,
  and closed families stay closed.
- **Substantive seeds:** five to ten seeds spanning the low, medium and high complexity tiers, at
  least two structural, each with Python source paths and config paths inside the scope (none
  configuration-only), no dependencies, a closest precedent that is a packet idea ID, an existing
  repository path or a URL, and a concrete difference; untaken pool seeds inside the scope taken.
- **Researched:** at least two distinct primary-literature links.
- **Valid:** the policy passes the protected list, the campaign ID is free, and the generated
  campaign passes the same validation as `validate`.

Only researcher drafts pass these checks. A user campaign or queued topic with
`configuration_only_ideas: true` has configuration-only seeds and refills by design: its seed files
and refills never pass through the "none configuration-only" seed check, and instead each
nonempty refill must carry an idea declaring `source_change: true`
([campaign setup](campaign-setup.md#declarative-policy)). Drafted topics never enable it.

There is no cross-provider critic.

## Records

`loop.json` holds the provider, GPU, profiles, starting reference, current topic, the active
rotation and a summary of past ones, loop agent jobs, taken pool seeds and the `attention` flag.
Each topic record has its campaign, **origin** (`user`, `queue` or `researcher`), source, title,
predecessor, chain incumbent, status (`active`, `retired`, `superseded`), start and retirement
times, **outcome** (decided, valid, researcher runs, adoptions, invalid, current streak), the
**streak at rotation** and the pool seeds it took, so yield by origin is measured rather than
assumed. Each rotation directory keeps `packet.json`, the drafts with `checks.json`, the staged
campaigns and a final `rotation.json`. After three consecutive retired topics without an adoption
the loop sets `attention` in its status for the user and keeps going.
