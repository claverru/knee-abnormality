# Targets-and-inputs campaign

**Started 2026-10-12 18:36 CEST, stopped by the user 2026-10-13 17:58 CEST after 29 runs.** The
campaign changed how the report-derived targets enter the loss, what the intensity pipeline feeds
a Refined-RadImageNet ResNet-50, and, through seeds the user added while it ran, revisited thin
adoptions and tried modules inside the backbone, on fold 2 of `all_5fold_scanner_v2` with fold 2
of `rn50-aligned-mixing-folds-0-2` as the reference. Adopted and on main: the mask decoder, the OA
detector beside the shared OA head and average-pool shortcuts
([result](../../../docs/validation/results/2026-10-13-targets-and-inputs-campaign.md)). Its
artifacts are in `outputs/autoresearch/targets-and-inputs`; only the incumbent's checkpoint is
kept. The worktree it ran from is removed, so it cannot be resumed as it stands.

| Read when | Document |
|---|---|
| Checking the objective, what is settled, the principles a seed follows, how a result is read, what is out of bounds, cost and adoption | [Contract](contract.md) |
| Reading the reference and its per-label scores, the training targets' prevalence and uncertainty per label, the precedent table and what was dropped before the campaign | [Prior work](prior-work.md) |
| Reading the first five seeds: soft-prevalence class weights, stable intensity bounds, uncertain report targets weighing less, hard report targets, the mask decoder | [Seeds](seed-ideas.yaml) |
| Reading the last five seeds: intensity and noise augmentation off, balanced target columns by selection, the hard-pair final phase with every study still seen, plane-contrast readouts off | [Second seeds](seed-ideas-2.yaml) |
| Reading the two seeds the user added on 2026-10-12 after Kaggle's score order: the three-compartment OA detector beside the routed shared OA head, and separate OA heads with the detector in its place | [Third seeds](seed-ideas-3.yaml) |
| Reading the first four of the eight seeds the user added on 2026-10-12 that revisit adoptions with thin evidence: report-attribute loss on, scanner confusion head beside L2-SP, view label bias off, anatomical attention bias back | [Fourth seeds](seed-ideas-4.yaml) |
| Reading the last four of them: sagittal fluid detail view off, detail-view embedding off, GIN off, resolution degradation off | [Fifth seeds](seed-ideas-5.yaml) |
| Reading the two follow-ups to the adopted mask decoder the keeper wrote from the audit of run 5 at the user's request: the decoder's loss weight equal to the map heads', and a 56 by 56 decoder level from the first stage | [Sixth seeds](seed-ideas-6.yaml) |
| Reading the seed the user asked for on 2026-10-13: squeeze-and-excitation in every ResNet-50 bottleneck, started as the identity, with the Refined-RadImageNet weights kept | [Seventh seeds](seed-ideas-7.yaml) |
| Reading the four backbone seeds the user chose on 2026-10-13 from the ResNet-50 modernisation review: global-context blocks, coordinate attention, average-pool shortcuts, blur-pool at the stage transitions | [Eighth seeds](seed-ideas-8.yaml) |
| Reading the seed that applies the user's adoption of the soft-prevalence class weights (2026-10-13, after S150 against S151) to the current incumbent | [Ninth seeds](seed-ideas-9.yaml) |
| Reading the bundled squeeze-and-excitation seed the user asked for on 2026-10-13 after run 22: late stages only, gates pulled toward the identity, no weight decay | [Tenth seeds](seed-ideas-10.yaml) |
| Reading the coordinate-attention seed as it will run, with the pull toward the identity the user added on 2026-10-13; it replaces the one in the eighth seeds | [Eleventh seeds](seed-ideas-11.yaml) |
| Reading the global-context seed with the pull toward the identity, asked for by the user on 2026-10-13 after the free run's logged values | [Twelfth seeds](seed-ideas-12.yaml) |
| Reading the blur-pool seed as requeued on the average-pool incumbent after its first run was cancelled on 2026-10-13 | [Thirteenth seeds](seed-ideas-13.yaml) |
| Checking the reference, the budget, the editable files, the config permissions and the role instructions | [Campaign definition](campaign.yaml) |
| Reading each run's result against the incumbent, per label | [Run notes](run-notes.md) |
| Reading the runtime's decision summary | [Findings](findings.md) |
| Inspecting the runtime's idea and queue projections | [Ideas](ideas.yaml), [backlog](backlog.yaml) |
