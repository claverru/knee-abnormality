# Teacher-student results

| Read when | Document |
|---|---|
| Reading the first fastMRI joint runs on fold 3: the ResNet-50 joint model against its control (S148), ConvNeXt-Tiny on the joint setup, fastMRI-only pretraining then joint (run B) or Kaggle-only (run C), the ConvNeXt + ResNet-50 blend (S152) | [2026-10-12 fastMRI joint first runs](2026-10-12-fastmri-joint-first-runs.md) |
| Checking main's ResNet-50 recipe at `288e50a` on folds 0–4 as pseudo-label teachers, and fold-0 ablations of the OA detector, average-pool shortcuts and mask decoder | [2026-10-13 main recipe on five folds](2026-10-13-main-recipe-on-five-folds.md) |
| Reading the second round: averaged teacher labels, fastMRI bone masks, ConvNeXt pretrained-then-joint against the ResNet-50 on five folds, the τ and multiplier search with its caching pitfall, S154/S155 and the distilled students S156/S157 | [2026-10-14 second round](2026-10-14-second-round-averaged-labels.md) |
| Reading the 15-silver-teacher round: fastMRI + OAI pretraining, the ConvNeXt fold-0 fine-tuning grid C1–C8, the OAI label rule, gold against silver for ensemble choices, stacking, the teacher share (S167) and ConvNeXt × 5 (S170), and where those runs are kept | [2026-10-15 silver-15 round and teacher share](2026-10-15-silver15-round-and-teacher-share.md) |
| Reading the generations: share children as teachers, all5 against mixed tables, the selection error on leaky gold, per-family splits and seeds, NextViT as sixth family, the rebuild's gen-1 teacher choice and the family probes S175–S182 | [2026-10-17 generations, resplits and rebuild](2026-10-17-generations-resplits-and-rebuild.md) |
| Reading the v3 labels and their chain, the final 60-model ensembles and their τ probes (S183–S197), the ×8 per-label build and its out-of-memory failure, gen 4 and the efficiency students S193 and S195 | [2026-10-19 labels v3, final ensembles and students](2026-10-19-labels-v3-final-ensembles-and-students.md) |
