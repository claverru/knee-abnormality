# Backbone families campaign

**Ran 2026-10-15 18:11 to 2026-10-16 15:12 CEST with Codex, stopped by the user.** FocalNet-T SRF
became the incumbent (×4 0.938816) and EfficientViT-B2 with stochastic depth 0.2 came level
(0.938773); both are merged as `configs/focalnet.yaml` and `configs/efficientvit.yaml`
([result](../../../docs/validation/results/2026-10-16-backbone-families.md)). Brings in backbone families different in kind from those
tried (SAM2.1 Hiera-T, HRNet-W18, PVTv2-b2, FocalNet-T SRF, RDNet-T, and DINOv3 ConvNeXt-T, DaViT-T, EdgeNeXt-B, EfficientViT-B2 and WTConvNeXt-T added at the user's request), each with starting settings
set from its own weight scale and layers, against the decoder-free MambaVision-T reference
`mamba-silver-fold-2` (fold 2, ×4 0.935160, ×1 0.930081). Ten forced seeds, then the researcher.
Training time 0.35 to 0.85 of the reference's 4,812 s (about 0.54 to 1.32 of a local MambaVision-T
run); ten epochs by default, a family setting. Runs from the worktree `knee-abnormality-backbone-families`.

Start: `rtk proxy uv run --locked just autoresearch start backbone-families --provider codex` from
the worktree.

| Read when | Document |
|---|---|
| Checking the objective, the seeds, what is settled, the principles, how a result is read, what is out of bounds and the cost | [Contract](contract.md) |
| Reading the reference and its exceptions, the five families measured (channels, weight scale, depthwise convolutions, BatchNorm, licence), how the starting settings were set, padding of image counts, DINOv3 ConvNeXt-T's weight scale and settings, the four families added later (and why RMT-S was left out), the precedents | [Prior work](prior-work.md) |
| Reading the five seeds: Hiera-T, HRNet-W18, PVTv2-b2, FocalNet-T SRF, RDNet-T | [Seeds](seed-ideas.yaml) |
| Reading the sixth seed, DINOv3 ConvNeXt-T, added after the start | [DINOv3 ConvNeXt-T seed](seed-dinov3-convnext.yaml) |
| Reading the PVTv2-b2 relaunch after its transient CUDA failure | [PVTv2-b2 relaunch seed](seed-pvt-retry.yaml) |
| Reading the RDNet-T relaunch after its transient CUDA failure | [RDNet-T relaunch seed](seed-rdnet-retry.yaml) |
| Reading the four seeds added on 2026-10-15: DaViT-T, EdgeNeXt-B, EfficientViT-B2, WTConvNeXt-T | [Four families seeds](seed-four-families.yaml) |
| Reading the DINOv3 ConvNeXt-T relaunch at ten epochs, the new default | [DINOv3 ten-epoch seed](seed-dinov3-10ep.yaml) |
| Reading PVTv2-b2 and RDNet-T with activation checkpointing of their early stages, after their out-of-memory failures | [Checkpointed seeds](seed-checkpointed.yaml) |
| Reading the FocalNet-T incumbent rerun at ten epochs (user's request) | [FocalNet-T ten-epoch seed](seed-focalnet-10ep.yaml) |
| Reading EfficientViT-B2 with checkpointed stages 0 and 1 after its preflight ran out of memory | [EfficientViT-B2 checkpointed seed](seed-efficientvit-checkpointed.yaml) |
| Reading the WTConvNeXt-T relaunch of its reviewed candidate after the read-only Triton cache failure | [WTConvNeXt-T relaunch seed](seed-wtconvnext-relaunch.yaml) |
| Reading the repaired WTConvNeXt-T at nine epochs, after its ten-epoch preflight projected 2.3% over the bound | [WTConvNeXt-T nine-epoch seed](seed-wtconvnext-9ep.yaml) |
| Reading the user's EfficientViT-B2 seeds of 2026-10-16: LayerNorm on the reads, bfloat16 attention | [EfficientViT-B2 seeds](seed-efficientvit-user.yaml) |
| Reading the user's second EfficientViT-B2 seeds: stochastic depth 0.1, checkpointing stage 0 only | [EfficientViT-B2 seeds 2](seed-efficientvit-user-2.yaml) |
| Reading EfficientViT-B2 with stochastic depth and stage 0 only checkpointed combined | [EfficientViT-B2 stochastic depth, stage 0](seed-efficientvit-sd-stage0.yaml) |
| Reading EfficientViT-B2 with stochastic depth 0.2 and only stage 0's first block checkpointed | [EfficientViT-B2 stochastic depth 0.2](seed-efficientvit-sd02.yaml) |
| Rerunning the keeper's GPU parity of the vendored WTConv layer against the official one | [WTConv GPU parity](evidence/wtconv_gpu_parity.py) |
| Checking the reference, the budget, the editable files, the config permissions and the role instructions | [Campaign definition](campaign.yaml) |
| Reading the runtime's decision summary | [Findings](findings.md) |
| Reading each run's details: epoch curves, token maps, per-label changes, causes (Hiera-T aborted at epoch 7) | [Run notes](run-notes.md) |
