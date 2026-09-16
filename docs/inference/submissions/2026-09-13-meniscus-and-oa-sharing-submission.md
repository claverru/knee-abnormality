# Meniscus and OA sharing submission

The selected 1B `meniscus-oa-classifiers` checkpoint (epoch 7, index 6;
0.892022 local macro AUC) was exported on 2026-09-13 to
`outputs/submission/exports/knee-meniscus-oa-20260913` using its archived training source,
2× images and batch size 2. Checkpoint SHA-256 is
`fc7ced812d323f6109887a941c5d3693f5f8c7dafcd49c9e3eb568d6f4744a4e`.
Strict CPU loading and bundle checks passed; source matches the training snapshot
and identifies two shared classifier pairs with independent attention. Export
preparation took 0.62 seconds; optional local inference was not requested.

The user-authorized automatic submission job launched at 21:21:19 UTC with
private dataset `claverru/knee-meniscus-oa-20260913`, version 1, and private offline
notebook `claverru/knee-meniscus-oa-20260913-inference`, version 1, script version
`349627071`. Submission **56215113** completed with **0.925 public AUC**, versus
reflection's **0.931**. The runner recorded completion at 21:38:08 UTC in the
[workflow record](../../../outputs/submission/records/knee-meniscus-oa-20260913.json).

The subsequent [CPU-only audit](../../history/sharing/2026-09-13-1b-public-result-and-overnight-code-choice.md)
found no bundle mismatch: manifest and file hashes, original checkpoint identity
and every bundled Python file match the submitted record and training snapshot.
Both exports have identical effective configuration, 2× density and batch 2;
both visible notebook runs use FP32. The downloaded three-study example outputs
have the correct target order and finite probabilities. Hidden scored predictions
and per-target public AUCs are unavailable. No fresh inference was run.
