# Anatomy-masks campaign

**Started on 2026-10-08 00:01 CEST with Codex and stopped by the user on 2026-10-08 08:44 CEST after 17 decided runs and three adoptions** (the bicubic cut and the map's joint centre by the contract's exception, then the joint centre with equal view exposure by the two-multiplier rule: ×4 0.927262, ×1 0.917892; the user doubts that last one, which trains 7% slower and is a rerun in effect). Results per run are in the [run notes](run-notes.md). The work of 2026-10-07 on exploiting a pretrained
segmenter's masks, as one campaign: the anatomy map taught per token, sections and routing read
from it, sagittal and coronal crops shifted and turned to the knee's own pose, and measurements
from the masks as targets. It starts on `rn18-anatomy-crops-sample550`, the first run of the line
above run 24 at both multipliers (×4 0.925885, ×1 0.917130; 0.953 public as S122, equal to run 24's S118). The segmenter is never run by the
campaign and nothing of it is needed at prediction. A run is adopted only when it is at least
level with the incumbent at both ×1 and ×4 and predicts in at most 1.3 times its time.

| Read when | Document |
|---|---|
| Checking the objective, the rule about the segmenter, the scope of each of the four topics, the eight principles, how a seed is written, what is out of bounds and closed, the guardrails and the two-multiplier adoption | [Contract](contract.md) |
| Checking the incumbent's design and results per target, the yardstick, the prepared targets and how the joints were derived, the six runs of 2026-10-07 and what each taught, the cost to Synovitis, the measurements on the masks, the cost of the crops and the open leads | [Prior work](prior-work.md) |
| Checking the sealed recipe, reference, budget (0.5 to 1.2), editable files, the two config permissions and the role instructions with the developer-environment paragraph | [Campaign definition](campaign.yaml) |
| Reading the first five forced seeds: turned crops cut with bicubic interpolation, the joint centre from the map instead of the spots, a tenth of the map's gradient to the backbone, the map's loss without Dice, the first look on a third of the slices | [Seeds](seed-ideas.yaml) |
| Reading the last two forced seeds: the fluid sagittal crops moved forward for the kneecap, the joint gap per side as a target | [Seeds 2](seed-ideas-2.yaml) |
| Reading the forced seed added by the user on 2026-10-08: the first look on the whole canvas resized to 224 px instead of its central crop | [First-look seed](first-look-ideas.yaml) |
| Reading the requeued third seed, a tenth of the map's gradient to the backbone, cancelled when it launched before the second seed's adoption | [Gradient requeue](gradient-requeue-ideas.yaml) |
| Reading the two forced seeds added by the user on 2026-10-08: sagittal crops a ninth wider and coronal crops a ninth wider, with the extents that set the field | [Field seeds](field-ideas.yaml) |
| Reading the proposals, the queue and the decision summary once the campaign runs | [Ideas](ideas.yaml), [Backlog](backlog.yaml), [Findings](findings.md) |
| Reading what each run gave after it finished (both multipliers, the per-target table, map and crop diagnostics), the repairs and the adoptions | [Run notes](run-notes.md) |
