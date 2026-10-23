# Backbone families 2 campaign

**2026-10-17 13:55 to 2026-10-18 15:02 CEST, Codex, stopped by the user after 25 trainings; NextViT-S adopted
(`configs/nextvit.yaml`, `318eba2`).** Backbone families against the [first round's](../backbone-families/INDEX.md)
incumbent FocalNet-T SRF (`focalnet-silver-fold-2`, fold 2, ×4 0.938816): the user's six of 2026-10-17
(OverLoCK-XT, TransNeXt-Micro, VSSD-Tiny, FlashInternImage-T, MedSAM2 Hiera-T, LingBot-Vision ViT-S/16), the
original RadImageNet DenseNet-121, InceptionV3 and InceptionResNetV2, and measured new families of 2026-10-18
(FocalNet-T LRF, NextViT-S, EfficientNet-B3 NS, TinyViT-21M, InceptionNeXt-T run; HGNetV2-B4, MambaOut-T and six second-batch families seeded but not run). NextViT-S
(SSLD 6M) was provisionally adopted (×4 0.938851, ×1 0.934578, blend with the ResNet-50 `resnet50-silver-fold2`
+0.003267); TinyViT-21M came level at ×4 (+0.003223). RadImageNet-pretrained families all lowered the blend.
Training bound 1,684 to 4,881 s (raised from 4,092 s by the user on 2026-10-18). Results:
[run notes](run-notes.md), [2026-10-18 backbone families 2](../../../docs/validation/results/backbones/2026-10-18-backbone-families-2.md).

| Read when | Document |
|---|---|
| Checking the objective, the seeds, what is settled, the principles, how a result is read, the adoption rule, what is out of bounds and the cost | [Contract](contract.md) |
| Reading the reference and the blend partner, the six families measured (stages, geometry, norms, custom ops, weight scale, licences, where weights and sources are), the starting settings, padding of image counts, Kaggle inference, the time bound, the precedents | [Prior work](prior-work.md) |
| Reading the first five seeds: OverLoCK-XT, TransNeXt-Micro, VSSD-Tiny, MedSAM2 Hiera-T, LingBot-Vision ViT-S/16 | [Seeds](seed-ideas.yaml) |
| Reading the sixth seed: FlashInternImage-T (DCNv4), last because its op must be built first | [Seed 6](seed-ideas-2.yaml) |
| Reading the keeper's requeue of LingBot ViT-S/16 with the augmentation at its stride-16 map after block 3 (2026-10-17) | [Seed 7](seed-ideas-3.yaml) |
| Reading the keeper's LingBot stage-decayed rate seed (0.8 per stage, from the stem's drift) | [Seed 8](seed-ideas-4.yaml) |
| Reading the original RadImageNet DenseNet-121, InceptionV3 and InceptionResNetV2 added by the user: weights converted from Keras and checked against it, input range, geometry, weight scale and settings, measured time | [RadImageNet prior work](prior-work-radimagenet.md) |
| Reading the three RadImageNet seeds | [Seeds 9 to 11](seed-ideas-5.yaml) |
| Reading the keeper's LingBot early-anchor follow-up (block 0's falling drift), removed by the user before it ran | [Seed 12](seed-ideas-6.yaml) |
| Reading the keeper's DenseNet-121 normalised middle-read seed (raw concatenated stride-16 map) | [Seed 13](seed-ideas-7.yaml) |
| Reading the re-described InceptionResNetV2 seed (pretrained_cfg and the non-2:1 grids spelled out) | [Seed 14](seed-ideas-8.yaml) |
| Reading the keeper's DenseNet-121 stage-decay seed (the refined ResNet-50's depth profile of relative steps) | [Seed 15](seed-ideas-9.yaml) |
| Reading the requeued InceptionResNetV2 seed (aborted for a disk compaction) | [Seed 16](seed-ideas-10.yaml) |
| Reading the keeper's FlashInternImage-T eight-epoch seed (nine epochs did not fit) | [Seed 17](seed-ideas-11.yaml) |
| Reading the 20 candidate backbones measured on 2026-10-18 (time, memory, weight scale, structure, pretraining) and why seven were seeded | [Candidates](prior-work-candidates.md) |
| Reading the new-family seeds FocalNet-T LRF, NextViT-S, EfficientNet-B3 NS and TinyViT-21M | [Seeds 18 to 21](seed-ideas-12.yaml) |
| Reading the new-family seeds InceptionNeXt-T, HGNetV2-B4 and MambaOut-T | [Seeds 22 to 24](seed-ideas-13.yaml) |
| Reading the keeper's FocalNet-T LRF twelve-epoch seed (the best new family) | [Seed 25](seed-ideas-14.yaml) |
| Reading the keeper's NextViT-S twelve-epoch seed (the incumbent's follow-up) | [Seed 26](seed-ideas-15.yaml) |
| Reading the second-batch seeds TResNet-M, CS3DarkNet-L, GhostNetV2-160, Twins-SVT-S and Res2Net50d | [Seeds 27 to 31](seed-ideas-16.yaml) |
| Reading the second-batch seed BiT ResNetV2-50x1 | [Seed 32](seed-ideas-17.yaml) |
| Checking the reference, the budget, the editable files, the config permissions and the role instructions | [Campaign definition](campaign.yaml) |
| Reading every run's ×4, ×1 and blend gain, the invalid and blocked runs, NextViT-S per label and the lessons (pretraining and the blend, optimisation against exploration, speed guesses, the time bound and its seal, integration defects, original RadImageNet weights) | [Run notes](run-notes.md) |
| Reading the runtime's decision summary | [Findings](findings.md) |
