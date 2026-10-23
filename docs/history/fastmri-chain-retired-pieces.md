# Retired pieces of the fastMRI teacher-student chain

Code and configs built during the fastMRI/OAI teacher-student line (2026-10-12 to 2026-10-22)
that the final recipe did not adopt, removed on 2026-10-23 before the branch was committed. The
adopted recipe is pretraining on fastMRI + OAI pseudo-labels, then Kaggle-only students
(gen2, gen3, gen4) for ConvNeXt-T, ResNet-50 and NextViT-S with v2 and v3 labels; its log is
[fastMRI joint training](../validation/results/teacher-student/INDEX.md).

**Recovering the code:** the removed code was never committed. The full uncommitted state just
before the removal is in the main checkout at
`outputs/archive/removed-non-adopted-20261023/`: `worktree-diff-before-removal.patch` (tracked
files, against `5bb5cd2`) and `untracked-before-removal.tgz` (the configs and other new files).

## Joint training with external batches

**What it was:** `extra_training` without `alone` interleaved every Kaggle batch with a batch of
external pseudo-labelled studies (`PairedStudies` and `PairedBatches` in
`training/lightning.py`, `EXTRA_BATCHES = 2`, gradient accumulation doubled). The external loss
was weighted by `extra_training.weight_schedule`: `constant`, or `quarter_cosine`, which faded
it as cos(π t / 2) over training. L2-SP anchoring skipped the external batches.

**Why it was dropped:** fine-tuning on Kaggle only, from the same pretraining, was as good or
better on the board.

| Comparison | Joint | Kaggle-only fine-tune |
|---|---|---|
| ConvNeXt fold 0, pretrained | S159 0.961 | S160 0.961, ranked above |
| ConvNeXt with teacher share 0.75 | C8, S168 0.964, ranked below | C6, S167 0.964 |

Per label, joint won the menisci and OA and Kaggle-only won bone and fluid labels (S159 vs S160,
contrast +0.0046, study bootstrap [+0.0010, +0.0092]); the board did not reward the difference.
One fade bug found on the way: progress was first computed from
`estimated_stepping_batches`, which counts both batches of a pair, so the weight ended at 0.71
instead of 0.

**What stays:** `extra_training` now trains on the external studies only (the pretraining
stage), with the same batch order as the old `alone` mode, so the existing pretraining
checkpoints remain reproducible.

## Per-label ensemble weights

**What it was:** `export --label-weights` took a JSON with one weight per member and label; the
bundle stored it as `models/label_weights.json`, and `predict_loader` averaged members per label
with those weights instead of equally.

**Why it was dropped:** weights tuned on the 58 gold studies overfit them. S164 (S163's members
with best-gold per-label weights, in-sample gold +0.0046) scored 0.963 against S163's 0.964 with
equal weights. Every later ensemble used equal weights.

## Configs of rounds and families not in the final recipe

| Removed configs | What they ran | Why retired |
|---|---|---|
| `{convnext,resnet50,mambavision}-silver15-{pretrain,joint,joint-oai-soft-tissue}` | First external round (2026-10-14/15): 15 silver teachers label fastMRI + OAI, pretraining, then joint with a fade | Joint training dropped; superseded by the teachers2 round and the rebuild chain |
| `{convnext,resnet50,mambavision,efficientvit}-teachers2-{pretrain,joint}`, `nextvit-teachers2-pretrain` | Second round on the selected teachers' labels | Superseded by the rebuild chain (pretraining, then Kaggle-only gen2/gen3) |
| `{efficientvit,focalnet,mambavision}-rebuild-{pretrain,gen2,gen3}` | The rebuild chain for three more families | Not in the final ensemble: the default families are ConvNeXt, ResNet-50 and NextViT. MambaVision with ConvNeXt (S162, 0.963) lost a step to the three-family S161 (0.964); adding five FocalNet students (S166, 0.963) lost to S163 (0.964). See [FocalNet and EfficientViT](../modeling/backbones/focalnet-and-efficientvit-backbones.md) and [MambaVision](../modeling/backbones/mambavision-backbone.md) |
| `convnext-tiny-tuned`, `convnext-tiny-tuned-fastmri-only`, `resnet50-joint`, `nextvit-gen3mixed-finetune` | Early fastMRI probes and a one-off gen3 fine-tune | Superseded by the rebuild chain |
| `convnext-tiny-distilled-student`, `rn50-distilled-student` | Earlier single-model students | Superseded by the `*-s188-student` configs |
| `configs/labels/rebuild-state-v3-1-label-sources.json` (removed) | The v3.1 silver label table | No kept config used it; v3 stayed the label table of the final recipe |
