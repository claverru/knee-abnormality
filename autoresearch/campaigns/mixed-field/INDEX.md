# Mixed-field campaign

**Ran 2026-10-01 14:25 to 20:10 CEST with Codex, stopped after its forced seeds on the user's
instruction; no researcher idea was added.** Thirteen decided runs, one accepted:
`contrast-split-112-140` (T1/PD series at 112 mm, fluid series at 140 mm), ×4 0.929570 against
0.928062 (+0.0015, 80% [+0.0004, +0.0026]). 160 mm added nothing in any design. The last seed,
`two-stream-paired`, finished inside the budget on its third build but its 140 mm stream stayed
switched off, so it does not test paired views ([note](#two-stream-result)). Adopted on main
after the public check: 120 mm on every series and middle-slice augmentation, not the contrast
split ([result](../../../docs/validation/results/2026-10-01-mixed-field-campaign.md)). Feeds the model a wide 160 mm view beside a narrow one
(120 mm, or the incumbent's 140 mm) at the fixed 224 px input: ten forced seeds from
configuration-only single-field yardsticks to two backbones reading every slice at both fields,
then researcher follow-ups on whichever mechanisms helped.

| Read when | Document |
|---|---|
| Checking the objective, the field-policy rule, seeds, budget, fixed, permitted and excluded mechanisms and evidence requirements | [Contract](contract.md) |
| Checking the field, resolution and multi-stream precedents, the clipping geometry, the 120 mm decision and the code facts the seeds rely on | [Prior work](prior-work.md) |
| Checking the sealed recipe, reference, budget, scope and role instructions | [Campaign definition](campaign.yaml) |
| Reading seeds 1–5 (single 160 mm, single 120 mm, field jitter, mixed 140/160, mixed 120/160) | [Seeds 1](seed-ideas.yaml) |
| Reading seeds 6–10 (per-field readout weights, extent-ruled field, field backbones, paired two streams, co-registered context) | [Seeds 2](seed-ideas-2.yaml) |
| Reading the 160 mm yardstick retake with the field-share columns logged, forced by Claude as campaign keeper on 2026-10-01 after the original seed blocked | [Yardstick retake](yardstick-retake-ideas.yaml) |
| Reading the user seed that gives 160 mm to half of the axial and a quarter of the sagittal slices and keeps the rest at 120 mm, forced on 2026-10-01 after the first four results | [Plane-weighted fields](plane-weighted-ideas.yaml) |
| Reading the user seed that puts T1/PD series at 112 mm (0.5 mm) and fluid series at 140 mm, forced on 2026-10-01 with the 112 mm field added to the scope | [Contrast split](contrast-split-ideas.yaml) |
| Reading the conditional seed that replaces the withdrawn `coregistered-context` (140 mm second stream), forced only if `two-stream-paired` beats 120 mm alone | [Co-registered context at 140 mm](coregistered-context-140-ideas.yaml) |
| Checking results | [Findings](findings.md) |

## Setup

- **Branch and worktree:** `mixed-field` in `../knee-abnormality-mixed-field`, off `00f2f32`; both
  removed on 2026-10-01 after the campaign files were committed on main. Each run keeps its frozen
  source under `outputs/autoresearch/mixed-field/`.
- **Reference:** `rn18-l2sp-no-confusion` (×4 0.928062, 0.943 public as S86), the
  `configs/training.yaml` recipe. It was trained from the since-removed `rn18-l2sp-confusion`
  worktree, so its `paths.label_manifest` and `paths.report_attribute_manifest` are declared config
  exceptions, as in the `revisits` campaign: its recorded `label_manifest_sha256` (`08339ec5…`) and
  `report_attribute_manifest_sha256` (`f712c551…`) match this tree's files.
- **Budget:** range 0.5–1.5 of the reference's 1,222 s training time (611–1,833 s; user,
  2026-10-01).
- **Monitor page:** `scripts.tools.serve_campaign_monitor --campaign mixed-field` on port 8765,
  served from `outputs/analysis/campaign-monitor/mixed-field`.
- **Narrow field:** 120 mm for every seed, from the
  [narrow-field check](../../../docs/research/input-resolution.md#what-a-narrow-field-cuts).
- **Not adopted elsewhere:** the `revisits` campaign's provisional adoptions (plane-contrast
  readouts, no anatomical bias) are not on `main` and are not part of this reference.

## Keeper repairs

- **2026-10-01 14:33–14:40 CEST, yardstick wording.** Seeds 1 and 2 were written as
  configuration-only, but every prepared candidate must declare a logged activation column
  ([lifecycle](../../lifecycle.md#development-and-gates)) and none measures the decoded field.
  `field-160-single` blocked on that conflict in both its attempts; its one-line spacing change was
  reviewed as correct. With the campaign stopped, the keeper corrected `field-120-single` in the
  state and in [Seeds 1](seed-ideas.yaml) to carry logging-only instrumentation
  (`field_share_120`, `field_share_140`, `field_share_160`), forced
  [`field-160-single-logged`](yardstick-retake-ideas.yaml) and moved it to second place in the
  queue, then resumed. State before the edit:
  `outputs/autoresearch/mixed-field/state.before-yardstick-fix-20261001.json`. No result existed.
- **2026-10-01 16:22–16:30 CEST, user seed placed next.** The user asked for a mix with fewer
  160 mm than 120 mm slices, weighted to the planes that can use the wide field. The keeper let
  `mixed-field-120-160` finish under a stop, forced
  [`plane-weighted-fields`](plane-weighted-ideas.yaml), moved it to second place behind the
  prepared `mixed-field-readout-weights` and resumed. State before the edit:
  `outputs/autoresearch/mixed-field/state.before-plane-seed-20261001.json`.
- **2026-10-01 17:35–17:40 CEST, 112 mm added and user seed placed next.** On the user's
  decision the keeper let `extent-ruled-field` finish under a stop, added 112 mm to the permitted
  fields in `campaign.yaml` and in the sealed policy (developer and researcher instructions only;
  decision record `outputs/autoresearch/mixed-field/decisions/user-fields-20261001T153641Z`),
  forced [`contrast-split-112-140`](contrast-split-ideas.yaml) behind the prepared
  `field-backbones-routed` and resumed. Backups: `state.before-112mm-20261001.json`,
  `campaign.before-112mm-20261001.yaml` and `state.before-contrast-seed-20261001.json` under the
  campaign's outputs.
- **2026-10-01 about 18:00 CEST, paired seeds changed (user decision).** After eight results
  the user moved `two-stream-paired`'s second stream from 160 to 140 mm and made
  `coregistered-context` conditional. With the campaign stopped behind `field-backbones-routed`,
  the keeper rewrote `two-stream-paired` in the state and in [Seeds 2](seed-ideas-2.yaml), marked
  `coregistered-context` blocked as withdrawn and removed it from the queue. **Condition:** force
  [`coregistered-context-140`](coregistered-context-140-ideas.yaml) only if `two-stream-paired`'s
  ×4 macro delta exceeds `field-120-single`'s +0.001135; otherwise skip it and let the researcher
  take over. Decision record and `state.before-paired-140-20261001.json` are under the campaign's
  outputs.
- **2026-10-01 18:17 CEST, stop after the forced seeds (user).** The user asked for the campaign
  to stop once the forced seeds finish, with no researcher ideas. A stop is requested the moment
  the last seed launches; since 18:50 researcher refills are also switched off in the state
  (`refill_below: 0`).
- **2026-10-01 18:46–18:50 CEST, two-stream time budget.** `two-stream-paired` (attempt
  `…163136Z-d9118018`, 20 slices, ten epochs) was stopped as invalid at a projected 2,181 s against
  the 1,833 s limit, about 223 s per epoch. Cause: its loader read and decoded every DICOM twice,
  once per field (`prepare_series` called per field with a separate cache). The keeper first set
  the repair to 8 epochs at 18 slices; the user rejected fewer epochs and asked for the same idea
  made faster, so at 18:55 the repair became ten epochs at 20 slices with a loader that reads each
  slice once and prepares both views from it (estimated 150–165 s per epoch, not measured). States
  before the edits: `state.before-two-stream-repair-20261001.json` and
  `state.before-two-stream-read-once-20261001.json`.
- **2026-10-01 19:10–19:25 CEST, two-stream second repair.** The read-once repair
  (`…164742Z-87f55a3b`) was cut again at a projected 1,946 s (194 s per epoch): the loader was not
  the bottleneck (workers at about 20% of a core, 0.02 s wait per batch, main process pinned).
  With the user's go the keeper profiled 30 training steps of that build on the idle GPU with
  synchronous CUDA: augmentation is 26% of step time (bilateral smoothing 14%), backward 25%, the
  two backbones' forward 13%, logging 5–9%. Augmentation runs per view on all three slice channels
  although `repeated_middle` uses only the middle one. The second and last repair selects the
  middle channel before augmentation and moves the idea's own per-step logging to once per epoch,
  at ten epochs and 20 slices (estimated 165–170 s per epoch, not measured). State before the
  edit: `state.before-two-stream-middle-slice-20261001.json`. The same waste exists in the
  reference recipe, outside this campaign's scope.

## Two-stream result

`two-stream-paired-20261001T172835Z-5c197a3e` (ten epochs, 20 slices, 1,736 s): ×4 0.930396, that
is +0.0008 against the incumbent (80% [−0.0005, +0.0022], rejected), +0.0023 against the original
reference (80% [+0.0007, +0.0039]) and +0.0012 against `field-120-single` (80% [−0.0001, +0.0024];
`outputs/analysis/m/mixed-field-two-stream-vs-reference` and `…-vs-120`).

**The 140 mm stream did not take part.** The build pools `narrow + tanh(gate) × (joint − narrow)`
with one scalar gate initialized at zero. The wide backbone and projection receive gradient only
through that gate, so they never trained: the joint ×1 validation AUC equals the narrow-only AUC to
five decimals at every epoch (0.925862 against 0.925864 at the end), the wide-only AUC is chance
(0.506), and the attention mass on wide tokens fell from 29–45% at epoch 0 to 1–9%. The activation
check passed because its threshold was any value above zero (observed 0.0013). The run is in
effect 120 mm alone through the ResNet-18, not a test of two views per slice. It is not an exact
repeat of `field-120-single` either (checked 2026-10-01 from `triplet_accounting.json`):
`data.study_triplet_total: 20` is a floor, not a cap, so it trained on about 25.4 slices per study
(177,416 views per epoch over 3,495 studies) against 26.8; it keeps the reference's slice
selection, whereas the two configuration-only yardsticks read different slices from the reference
(their selection hashes differ, because their ranking radius followed the field); and it augments
the middle slice only. Any of these, or run-to-run noise, can account for its +0.0012 over
`field-120-single`. The zero-start came from the campaign's
developer instruction that new projections start at zero; for a whole stream behind one gate that
blocks learning. `coregistered-context-140` was not forced: its condition is met on the number
(0.930396 above 0.929197) but not in substance.

## Manual runs after the stop: middle-slice augmentation

Run by the keeper on the user's instruction (2026-10-01, 20:28–21:10 CEST) outside the supervisor,
each as the campaign would run a seed: a frozen copy of the parent's source with one change, the
parent's config, seed and fold, the campaign's run limits, and `compare_fold_runs` with 2,000 paired
report-group draws. No sandbox and no agent review; before each launch the plain and mirror-TTA
logits on the parent's checkpoint were checked bit-identical between old and new code (16 studies).
Artifacts: `outputs/autoresearch/mixed-field/manual/`.

**Change:** with `model.slice_channels: repeated_middle`, the middle slice of the images, valid
rectangles and channel indices is selected before reflection, contrast mixing and augmentation
instead of after, so nothing is computed for the two unused neighbour slices
(`changes.diff` in each run folder).

| Run | Parent | ×4 macro | vs parent (80%) | vs reference 0.928062 (80%) | Training s (parent) | Peak GPU |
|---|---|---:|---|---|---:|---:|
| `middle-slice-augmentation-20261001` | incumbent `contrast-split-112-140` (0.929570) | 0.929212 | −0.0004 [−0.0014, +0.0007] | +0.0012 [−0.0002, +0.0025] | 1,103 (1,273) | 3.7 GiB |
| `field-120-middle-slice-augmentation-20261001` | `field-120-single` (0.929197) | 0.929301 | +0.0001 [−0.0011, +0.0013] | +0.0012 [−0.0003, +0.0028] | 1,095 (1,159) | 3.7 GiB |

- **The change is neutral on score in both recipes** and saves about 6–8% per epoch (107–110 s
  against 115–119 s) and most of the GPU memory. No label moved outside noise against either
  parent. The user settled on 2026-10-01 that it goes to main, with mirror TTA handled the same
  way.
- **It does not explain the two-stream run's score.** 120 mm with middle-slice augmentation scores
  0.929301, which is −0.0011 against `two-stream-paired` (0.930396; 80% [−0.0023, +0.0001]). The
  two plain readings of 120 mm alone now agree (0.929197 and 0.929301, +0.0011 and +0.0012 over the
  reference); the two-stream run's extra 0.001 is unexplained and within the usual run-to-run
  noise.

### Public check (user request, 2026-10-01 21:14–21:25 CEST)

Both manual runs were submitted with S86's inference settings (×4, τ 0.9, batch 4, no TTA), each
bundle exported from its own frozen source; weights checked against every earlier bundle.

| Submission | Model | ×4 local | Public |
|---|---|---:|---:|
| S86 | reference `rn18-l2sp-no-confusion` (140 mm) | 0.928062 | 0.943 |
| S90 | `field-120-middle-slice-augmentation-20261001` (120 mm, middle-slice augmentation), bundle `c58fed9b` | 0.929301 | **0.947** |
| S91 | `middle-slice-augmentation-20261001` (T1/PD 112 mm, fluid 140 mm, middle-slice augmentation), bundle `434adacb` | 0.929212 | 0.945 |

One submission each, fold 0, one seed; the board rounds to three decimals. Earlier ResNet-18
variants of the reference recipe scored 0.942–0.944 (S81, S82, S84, S86, S87).
