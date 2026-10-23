# Fixes-and-removals campaign

**Closed: stopped by the user on 2026-10-05 after 13 decided runs; its incumbent's four changes are on main since 2026-10-06 ([result](../../../docs/validation/results/campaigns-and-queues/2026-10-05-fixes-and-removals-campaign.md)); the worktree and branch are removed.** Codex, fourteen forced seeds, of which seven removals never ran. It repaired defects found by the code audits of
2026-10-05 and tested whether mechanisms active in the recipe then were still needed, by
switching them off one per run on the ResNet-18 recipe. The first seed, the detail-view position fix under reflection (training and TTA), is
adopted by user decision.

| Read when | Document |
|---|---|
| Checking the objective, the forced adoption of the first seed, the seed order and what is fixed and permitted | [Contract](contract.md) |
| Checking what the audits measured and closed, what stays outside the campaign and the candidates not yet seeded | [Prior work](prior-work.md) |
| Checking the sealed recipe, reference, budget, scope and role instructions | [Campaign definition](campaign.yaml) |
| Reading seeds 1–5, the fixes (mirrored detail-view positions, contrast mixing on an aligned donor, centre without side, the slab rule, positive feature-statistics std) | [Seeds 1](seed-ideas.yaml) |
| Reading seeds 6–9 (the 3D position in place of 1D and 2D, and the OA detector, intensity and noise removals) | [Seeds 2](seed-ideas-2.yaml) |
| Reading seeds 10–14, removals (half-knee loss, meniscus own-side mask, plane-contrast readouts, detail-view embedding, report-attribute heads) | [Seeds 3](seed-ideas-3.yaml) |
| Reading the keeper seed that reruns the aligned contrast mixing on the incumbent with seed 1's fix, after its first run was cancelled | [Contrast mixing requeue](contrast-mixing-requeue-ideas.yaml) |
| Reading the keeper seed that keeps the aligned contrast mixing and fades the blend toward the edge of the donor's coverage, forced after the aligned blend scored lower | [Feathered aligned mixing](contrast-mixing-feathered-ideas.yaml) |
| Reading the user seed that applies the three fixes not adopted one by one (slab rule, aligned contrast mixing, positive std) together, forced on 2026-10-05 | [Unadopted fixes together](unadopted-fixes-together-ideas.yaml) |
| Reading the keeper seed that runs the 3D position with the OA detector removed, forced after the detector went silent beside the 3D position | [3D position without the OA detector](position-3d-without-oa-detector-ideas.yaml) |
| Reading the three user seeds that retake the 3D position against its overfitting (position dropped for 15% of training studies, position in the attention path only, wavelengths from 32 mm), forced on 2026-10-05 | [3D position retakes](position-3d-retakes-ideas.yaml) |
| Reading the user seed that applies the slab rule and the positive std without the aligned mixing, adopted by user decision, forced on 2026-10-05 | [Slab rule and positive std](slab-rule-and-positive-std-ideas.yaml) |
| Reading the two keeper seeds from the audit of the 3D runs (the 3D position beside 1D and 2D, and no position embeddings at all), forced on 2026-10-05 | [Position audit seeds](position-audit-ideas.yaml) |
| Reading the keeper requeue of the slab rule and positive std on the incumbent without the OA detector | [Slab rule and positive std requeue](slab-rule-and-positive-std-requeue-ideas.yaml) |
| Checking what must be checked again before a submission (the OA detector's removal, the adoption bar of seeds 4 to 9) | [To revisit](run-notes.md#to-revisit) |
| Checking results | [Findings](findings.md) |
| Reading each run's per-label deltas, curve, mechanism readings and lesson, and the keeper's repairs | [Run notes](run-notes.md) |

## Setup

- **Branch and worktree:** `fixes-and-removals` in `../knee-abnormality-fixes-and-removals`, off
  `7ccf505`. Source is unchanged from main: the seeds carry their own changes, because an edit to
  a hashed source in the worktree changes the baseline's identity.
- **Links:** `outputs` and `data` in the worktree are links to the main checkout's.
- **Reference:** `rn18-fold1` (`outputs/training/20261005-113728-rn18-fold1`), the recipe in
  force trained from main on fold 1 with seed 1110484336 (user decision, 2026-10-05): ×4
  0.923102, ×1 0.914860, 1,179 s. The worktree's `configs/training.yaml` carries that fold and
  seed; main's keeps fold 0. Its two label-manifest paths name the main checkout and are
  declared config exceptions.
- **Budget:** range 0.5–1.2 of the reference's training time, 589–1,415 s (user, 2026-10-05).
- **Adoption:** fixes and removals on an observed delta of zero or more, the 3D position on the
  shared rule, seed 1 always ([contract](contract.md#adoption)).
- **Seed order:** forced, in file order.
