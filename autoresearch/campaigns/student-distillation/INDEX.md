# Student-distillation campaign

**Closed on 2026-10-10 at 16:07 CEST by the user, after 16 forced seeds; not to be resumed unless asked.** Recipe found: 0.75 × silver + 0.25 × the raw joint teacher prediction (incumbent `teacher-share-025-rerun`, fold-0 overall ×4 0.935809, ×1 0.930532, gold ×4 0.938714). Its reference is main's existing run, never scored against; the first seed, half silver labels and half raw teacher predictions, was adopted by force. It ran on a split with one fifth of the data held out, every gold study in it, and four teacher folds in the rest, with ResNet-18 teachers and students. Results, repairs and the closing summary are in the [run notes](run-notes.md#campaign-closed-2026-10-10-1607-cest-by-the-users-instruction).

## Files

| Read when | Document |
|---|---|
| Checking the objective, the four choices of a recipe, the split, the teachers' table, the scope, what is out of bounds, the two scores and the adoption rule | [Contract](contract.md) |
| Checking the first round, what is settled, how fold teachers differ in level, the combinations that do not help, where the adoption numbers come from and the precedent table | [Prior work](prior-work.md) |
| Checking the recipe, the open reference, the budget, the editable files, the `distillation` config permission and the role instructions | [Campaign definition](campaign.yaml) |
| Reading the first five forced seeds: the teacher share at 0.5, 0.25, 0.75 and 1.0, and a share per label | [Seeds](seed-ideas.yaml) |
| Reading the last four forced seeds: reranked fold teachers, recalibrated fold teachers, a share from plane agreement, a target per plane | [Seeds 2](seed-ideas-2.yaml) |
| Reading the two share seeds requeued after the comparison repair of 2026-10-10: 0.25 and 0.75 again | [Rerun seeds](seed-ideas-reruns.yaml) |
| Reading the share seed the keeper added after 0.25 was adopted: 0.125 | [Low share seed](seed-ideas-share-low.yaml) |
| Applying the candidate the keeper prepared for `reranked-fold-teachers`: its diff, activation signature and label-table report | [Prepared reranked candidate](prepared/reranked-fold-teachers/report.md) |
| Reading the recalibration seed requeued after the abort of 2026-10-10 12:57 | [Rerun seeds 2](seed-ideas-reruns-2.yaml) |
| Reading the reranking seed requeued after its run was stopped for time on 2026-10-10 | [Rerun seeds 3](seed-ideas-reruns-3.yaml) |
| Reading the seed the user asked for from the monitor page on 2026-10-10: the teacher only where it disagrees with silver beyond its repeat noise | [Gate seed](seed-ideas-gate.yaml) |
| Reading the second seed the user asked for from the monitor page on 2026-10-10, the campaign's last run: the teacher's residual at each silver level | [Level seed](seed-ideas-level.yaml) |
| Following the overall and the gold score across the runs | [Run notes](run-notes.md) |
| Reading the proposals, the queue and the decision summary once the campaign runs | [Ideas](ideas.yaml), [Backlog](backlog.yaml), [Findings](findings.md) |
