# Smart-crops campaign

**Closed: ran from 2026-10-01 23:49 to 2026-10-02 13:03 CEST**, stopped by the user after 29
scored runs; do not resume. It added resolution on the joint and tried to let the model choose
where to look: T1/PD coronal and sagittal series at 0.375 mm, first as an 84 mm view in place of
the 120 mm one, then as a 60 mm crop beside it, placed by rule, by the model's own attention and
per compartment, and last with the crop's own loss. Seven forced seeds, then the researcher was
free to try any smart-crop idea inside three limits (the time budget, no more than the 26 slices
per study, no input above 224 px), and the user added 14 seeds on the fly. Outcome: the 84 mm
T1/PD and 93 mm fluid views were adopted on main
([result](../../../docs/validation/results/2026-10-02-smart-crops-campaign.md)); the work
continues as [view optimisation](../../../docs/ongoing/view-optimisation.md).

| Read when | Document |
|---|---|
| Checking the objective, the view policy, seeds, budget, fixed, permitted and excluded mechanisms and evidence requirements | [Contract](contract.md) |
| Checking the crop, resolution and multi-view precedents, the acquired resolution per series type, what the 120 mm field cuts, where the 140 mm control's attention sits, the literature, the code facts the seeds rely on and the expected size | [Prior work](prior-work.md) |
| Checking the sealed recipe, reference, budget, scope and role instructions | [Campaign definition](campaign.yaml) |
| Reading the five crop seeds (T1/PD detail view at 84 mm, rule-placed crop, attention-placed crop, compartment crops, crop with its own loss) | [Seeds](seed-ideas.yaml) |
| Reading seeds 6 and 7, the user's retakes of mixed-field's two-stream run: fluid-sensitive slices also at 140 mm through a ResNet-10t with no gate, beside a main view of 120 mm and of 112 mm | [Two views](seed-ideas-2.yaml) |
| Reading seeds 8 and 9, the user's retakes of `fluid-detail-93-replacement` (S93) with the axial view widened to 130 mm and to 140 mm, force-queued on 2026-10-02 | [Axial width](seed-ideas-3.yaml) |
| Reading seeds 10 to 14, the user's single-type variations of the adopted S93 policy after the spacing review (fluid zoom on one plane only, sagittal fluid without the anterior offset, sagittal fluid at 100 mm, T1/PD at 93 mm), force-queued on 2026-10-02 | [Per-type variations](seed-ideas-4.yaml) |
| Reading seed 15, the user's retake of `crop-compartments` on the adopted policy: two rule-placed coronal T1/PD compartment crops whose tokens enter only the medial and lateral OA labels, force-queued on 2026-10-02 | [OA-routed crops](seed-ideas-5.yaml) |
| Reading seed 16, the stronger version of seed 15 the user asked for so that medial OA also rises: compartment crops on coronal fluid slices too, each routed to the OA label of its own side and trained by its own loss, force-queued on 2026-10-02 | [OA compartment readers](seed-ideas-6.yaml) |
| Reading seed 17, the user's seed on mixed pixel sizes in one encoder: separate BatchNorm statistics per view spacing in the early backbone stages, force-queued on 2026-10-02 | [Per-spacing statistics](seed-ideas-7.yaml) |
| Reading seeds 18 to 21, queued after the fluid two-by-two: a 120 mm sagittal fluid view that only Baker's cyst reads, the T1/PD zoom on one plane only (two seeds), and the sagittal windows without the 10 mm downward shift; force-queued on 2026-10-02 | [Baker's routing and T1/PD split](seed-ideas-8.yaml) |
| Checking results | [Findings](findings.md) |
| Reading which view of which series type serves which label: the ledger of design choices (settled or open), the per-label effect of every view change, what each kind of token reads alone, the slice supply per type and the details and lesson of each run | [View effects by target](view-effects-by-target.md) |

## Setup

- **Branch and worktree:** `smart-crops` in `../knee-abnormality-smart-crops`, off `6a99439`.
- **Reference:** `rn18-field-120-control` (`outputs/training/rn18-field-120-control`, ×4
  0.930666, 0.947 public as S92), the `configs/training.yaml` recipe in force, trained on 2026-10-01 from the main
  checkout at `6a99439`: 120 mm, middle-slice augmentation, plane-contrast readouts, no anatomical
  bias. Its `paths.label_manifest` and `paths.report_attribute_manifest` name the main checkout
  and are declared config exceptions.
- **Budget:** range 0.5–1.5 of the reference's 1,149 s training time (575–1,724 s; user,
  2026-10-01). Epochs, batch size and slices per study are not editable.
- **Seed order:** forced, simple to complex; the order of expected gain is in
  [prior work](prior-work.md#expected-size).
- **No topic rotation:** `rotate_after: 1000` (user, 2026-10-01); after the seeds the researcher
  keeps refilling until the user stops the campaign.

## Keeper repairs

- **2026-10-02 06:46–06:48 CEST, provider switch.** Codex hit its usage limit after 14 decided
  ideas (agent jobs blocked until 2026-10-03 19:28 CEST). On the user's standing instruction the
  keeper stopped the campaign, set `provider` to `claude` in the state and resumed; the role
  profiles for both providers were sealed at the start. State before the edit:
  `outputs/autoresearch/smart-crops/state.before-claude-20261002T064700.json`. Ideas up to and
  including the build of `t1pd-foveated-native-view` were developed by Codex; its review and
  everything after by Claude, until the switch back.
- **2026-10-02 09:01–09:22 CEST, switch back to Codex.** The user asked for Codex as soon as it
  answered again; a test request passed at 09:01. The keeper requested a stop, the supervisor let
  `t1pd-detail-dense-final-grid` finish training, and the keeper set `provider` to `codex` and
  resumed. State before the edit:
  `outputs/autoresearch/smart-crops/state.before-codex-20261002T092209.json`. Claude developed
  and reviewed `crop-full-context-ranking-locator` (repair) through
  `t1pd-detail-dense-final-grid`, and wrote the refill that proposed `fluid-detail-93-replacement`.
- **2026-10-02 09:16 CEST, user seeds.** `axial-130-detail-84-93` and `axial-140-detail-84-93`
  force-queued with `import-ideas --forced` from [seed-ideas-3.yaml](seed-ideas-3.yaml).
- **2026-10-02 09:33 CEST, user adoption.** `fluid-detail-93-replacement` made the incumbent with
  `adopt` after its public score (S93 0.950 against 0.947 for the reference as S92); the rule
  had rejected it at +0.00096. Decision record:
  `outputs/autoresearch/smart-crops/decisions/user-adopt-20261002T073318Z-9ed20e99`.
- **2026-10-02 10:17–10:31 CEST, supervisor restart.** Stopped at a quiet point and resumed on
  Codex to load the new `move-idea` and `remove-idea` commands
  ([runtime](../../runtime.md#importing-moving-and-removing-ideas-and-manual-adoption)).
- **2026-10-02 11:38 CEST, user removals.** `oa-routed-compartment-crops` (superseded by
  `oa-compartment-readers`) and the researcher's `axial-fluid-adaptive-detail` (attention
  placement plus an added view, both shown not to help) retired with `remove-idea`; decision
  records under `outputs/autoresearch/smart-crops/decisions/user-remove-20261002T093819Z-*`.
- **2026-10-02 12:15 CEST, closing after the queued view runs (user).** The user decided to close
  the campaign once `sagittal-fluid-100`, `sagittal-fluid-wide-for-bakers`,
  `coronal-t1pd-84-sagittal-t1pd-120`, `sagittal-t1pd-84-coronal-t1pd-120` and `t1pd-detail-93`
  are decided, and to continue in a new campaign on a flexible per-type view configuration.
  `oa-compartment-readers`, `sagittal-no-vertical-shift` and
  `spacing-specific-batch-statistics` were retired with `remove-idea`; the two split runs were
  moved ahead of `t1pd-detail-93`. The keeper stops the campaign when the last of the five is
  decided, before the researcher refills.
- **2026-10-02 12:30 CEST, requeue after a dependency-gate block.**
  `coronal-t1pd-84-sagittal-t1pd-120` was blocked at its implementation review, not by a
  result: its integration removed `t1pd-detail-84` (sagittal T1/PD no longer zoomed) while
  retaining `fluid-detail-93-replacement`, which declares a dependency on it, and the reviewer
  asked for an authorization. The keeper requeued the three T1/PD runs from
  [seed-ideas-9.yaml](seed-ideas-9.yaml) as `coronal-t1pd-84-only`, `sagittal-t1pd-84-only` and
  `t1pd-detail-93-both`, each stating a complete view policy that supersedes both accepted view
  ideas by the user's authorization, as the fluid split runs superseded
  `fluid-detail-93-replacement`. `t1pd-detail-93` was retired before development because it
  would have met the same gate; `sagittal-t1pd-84-coronal-t1pd-120` was already with the
  developer and is left to its own review, its requeue to be removed if it passes.
- **2026-10-02 12:51 CEST, stop requested by the user.** The user preferred to move on to the
  next campaign, where the spacing of each series type is a parameter and the remaining runs
  become part of it. The stop lets `coronal-t1pd-84-only` finish training and its comparison;
  `sagittal-t1pd-84-only` (built and reviewed) and `t1pd-detail-93-both` (queued) were not run.
  The run decided during the stop has no reporter write-up; its result is in
  [view effects by target](view-effects-by-target.md). Do not resume: a resume would launch
  `sagittal-t1pd-84-only`.
