# Labels v3, the final ensembles and the students (2026-10-19 to 2026-10-22)

**The v3 labels lifted the gen-1 teachers' gold AUC by +0.0067 (0.95340 against 0.94666) but the
v3 gen-2 ensemble tied its v2 counterpart on the board (S183 0.965, one place below S181). What
moved the board was the attention temperature: the same 15 v3 gen-2 models went from 0.965 at
τ 0.85 to 0.967 at τ 0.5 (S185), and all 60 v2 and v3 gen-2 and gen-3 models reached 0.968 at
τ 0.5 (S187) and, first among the 0.968s, at τ 0.4 (S188), the final best submission.** Students
distilled for the efficiency board scored 0.960 (S193) and 0.964 (S195). The τ analysis is in
[attention temperature](../pooling-and-attention/2026-10-22-attention-temperature.md); the chain
in the [teacher-student chain](../../../modeling/external-data-and-distillation/teacher-student-chain.md);
the label table in [label tables](../../../public-labels/label-tables.md).

## Silver labels v3

- **Why:** the gen-1 error analysis (`outputs/logs/rebuild/gen1-error-analysis.md` in the main
  checkout) found most top errors are label problems shared by all three families. Six audit
  agents proposed per-label rules (`outputs/logs/rebuild/label-audit/{ligaments,menisci,osteoarthritis,fluid,bone,cross-cutting}.md`);
  a final-rules agent drafted the set and the drop-in `apply_v3_rules` code. Claude subagents read
  all 4,407 reports for the rule questions instead of regexes or JevK5 (state and procedure in
  `outputs/labels-v3-reading/README.md`).
- **Build (2026-10-19):** `rebuild_state_v3` = post-calibration overrides on silver rows only, gold
  untouched, folds of `all_5fold_scanner_v2` kept as `all_5fold_scanner_v3` (build report
  `outputs/logs/rebuild/label-audit/final/v3-build.md`): gold cross-fitted macro AUC 0.880 → 0.906
  (optimistic, some rules were written after reading gold reports), 748 silver flips, 40 studies
  masked; a verdict blend was rejected (weight 0). Manifest
  `configs/labels/rebuild-state-v3-label-sources.json`.
- **v3 over v3.1:** a second soft pass, v3.1, moved 62 silver cells to 0.45/0.55. The choice used
  the board-order correlation of local silver scores over 26 re-scorable submissions
  (`outputs/labels-v3-reading/lb-correlation/results.md`): Spearman v2 0.20, v3 0.43, v3.1 0.41; v3 >
  v2 in 99.6% of bootstrap draws; v3.1 − v3 = −0.017 [−0.046, +0.005], lower in 81% of draws; v3.1
  also mixed the v2-trained ensemble into gen-1 targets. A v3.2 built in parallel was worse
  ([v3.2 labels](../../../public-labels/audits/2026-10-19-v32-labels.md)). A final review
  (`outputs/labels-v3-reading/pass3/`) found 69 large label/ensemble disagreements on trauma knees
  were model errors, 2 of 45 MCL low-grade cells borderline, 14 of 425 size-dependent cells wrong
  (mostly sub-1 cm OA lesions gold graders often call positive), and a known rule gap of 5
  positive cells for old fracture plus avulsion or osteochondral findings; v3 was kept unchanged.

## The v3 chain

- **Gen 1** (`{convnext,resnet50,nextvit}-v3-gen1-fold-{0..4}`, new seeds in
  `outputs/logs/rebuild/rebuild-seeds.tsv`), three-family ensemble at ×4, τ 0.85, TTA, own folds:
  gold 0.95340 against 0.94666 for the v2 gen 1 (+0.0067, bootstrap 10/90% +0.0017/+0.0122, P
  0.95); per family ConvNeXt +0.0107, ResNet-50 +0.0058, NextViT +0.0049; silver on v3 targets
  −0.0005 (seed noise), on v2 targets −0.0066 (expected). Gold per label: Synovitis +0.072, Lateral
  Meniscus +0.014, Fracture +0.013, Medial Meniscus +0.005, others within ±0.002 except Lateral OA
  −0.010 and Contusion −0.013. The gate (gold ≥ old gen 1 − 0.002, silver on v3 better) passed.
- **Share:** recalculated on `rebuild-v3-t2-x4-tau085-tta` with v3 cross-fitted gold values:
  optimum 0.80 ([share choice](../../../modeling/external-data-and-distillation/teacher-student-chain.md#choosing-the-teacher-share));
  gen 2 and gen 3 trained at 0.80.
- **External table and pretraining:** 15 v3 external predictions (~55 min each), table
  `rebuild-v3-gen1-x4-tau085-tta-oai-soft-tissue` (19,265 studies, every teacher's correlation with
  the mean ≥ 0.935); three pretrainings on the external studies only, about 30 minutes per epoch.
- **Gen 2** (15 runs, complete 2026-10-20 15:17 CEST): gold 0.95702 against 0.95195 for v2 gen 2
  (+0.0051, both partly leaky on gold; Synovitis +0.051, Fracture +0.007, Contusion +0.007, PF OA
  −0.008); silver on v3 targets equal (−0.00002), on v2 targets −0.0064. Mixed table share curve
  (information only): 0.75 0.9594, 0.78 0.9595 (peak), 0.80 0.9593.
- **Gen 3** (15 runs) followed at share 0.80; its fold-4 out-of-fold predictions were made on the
  rented machine after a local PC crash.

## Board results

Kaggle's order among tied scores is the `public_rank` of the
[ledger](../../../inference/submissions/ledger.md). All at ×4 with mirror TTA, equal member weights.

| Submission | Members | τ | Public | Place |
|---|---|---|---|---|
| S181 | v2 gen 2, ConvNeXt + ResNet-50 + NextViT × 5 folds (15) | 0.85 | 0.965 | 8th of the eleven 0.965s once S183 scored |
| S182 | v2 gen 3 (15) | 0.85 | 0.965 | first of the 0.965s |
| S183 | v3 gen 2 (15) | 0.85 | 0.965 | directly below S181 |
| S184 | S183's weights | 0.15 | 0.963 | last of five 0.963s |
| S186 | S183's weights | 0.3 | 0.966 | only 0.966 |
| S185 | S183's weights | 0.5 | 0.967 | best at the time; 2nd of two 0.967s once S191 scored |
| S187 | v2 + v3, gen 2 + gen 3 (60) | 0.5 | 0.968 | 2nd of seven 0.968s |
| **S188** | the same 60 | **0.4** | **0.968** | **1st of seven 0.968s, best submission** |
| S189 | the same 60 | 0.6 | 0.968 | 6th |
| S191 | the same 60 at ×8, per-label τ at 0.55 × each label's training width, backbones in FP16 | per label | 0.967 | 1st of two 0.967s |
| S192 | S188 with Contusion's attention rows at τ 0.93 | 0.4 / 0.93 | 0.968 | 5th |
| S194 | the same 60 | 0.35 | 0.968 | 7th |
| S196 | S188's 60 + nine gen-4 models (69) | 0.4 | 0.968 | 4th |
| S197 | the same 69 | 0.43 | 0.968 | 3rd |

- S190, the ×8 per-label bundle of S191 with FP32 weights at batch 2, failed in the hidden rerun
  after about 12 minutes, almost certainly out of GPU memory (worst training pair estimated at
  14.2 of 14.6 GiB). S191 held the backbones in FP16 (`model.backbone.half()`), cutting the
  resident weights from 6.7 to 3.2 GiB, with predictions within 1e-4 per label of S190's visible
  run; full `model.half()` and Conv/Linear-only half both failed in the side head (FP32 input in
  an autocast-off region), and `requires_grad_(False)` alone did not lower memory, so the saving
  is the FP16 backbones. S191's visible run peaked at 7.76 and 5.18 GiB of GPU memory (S190's: 10.77
  and 8.19) at 0.252 s per image.
- **Per-label τ of S190 and S191.** Rule: each label attends c times its own training width (×1,
  τ 1), with c = 0.55, set per attention channel by scaling the exported weights' rows (no code
  change). Basis: at one global τ (×4, τ 0.40) the attended width differed 2.8× across labels
  (Contusion 0.36× of its training width to MCL 0.99×), because routing gives labels different
  token counts, and locally the most squeezed labels lost most when sharpened (Spearman −0.67,
  p 0.018). Channel τ: ACL 0.27, MCL 0.24, Effusion 0.41, Synovitis 0.35, Baker's 0.41, Contusion
  0.44, Fracture 0.34, meniscus 0.305, OA 0.38 (PF OA prior row 0.35), routing temperature 0.34.
  Per-study τ for the uneven ×8 slice gains was estimated at about 1e-5 AUC and not built. The
  rule scored 0.967 against S188's 0.968 (table above). The brackets c 0.45 and 0.65, planned
  only for a result above S188, were not run (width analysis:
  [attention temperature](../pooling-and-attention/2026-10-22-attention-temperature.md#attention-width)).
- Bundles that mix members baked at τ 0.85 with raw members carry a per-member scale: S188's
  description shows 0.4706 (= 0.4 / 0.85) for the 45 baked members and 0.4 for the 15 v3 gen-3
  members exported at τ 1.0.
- Hidden-test studies are smaller than training studies (about 104 images per study at ×4 from
  S187's runtime against 123 in training).
- Local entries of 2026-10-21 that called a score "gold" were scored against the v3 silver
  table, not the radiologist labels; the corrected analysis is in
  [attention temperature](../pooling-and-attention/2026-10-22-attention-temperature.md#gold-was-scored-on-silver-labels).
  No local proxy tried (v2/v3 silver, k = 1/3/12 held-out members, scanner groups, study size,
  per-study rules) reproduced the board's preference for τ 0.4–0.5 over 0.85.

## Efficiency students and gen 4

- **S188 students (2026-10-21/22):** `convnext-s188-student`, `resnet50-s188-student`,
  `nextvit-s188-student` (configs `configs/{family}-s188-student.yaml`). Fold-0 scores are
  inflated by training on fold 0: ConvNeXt epoch 9 ×1 0.944085 / ×4 0.946344 (3,613 s, peak 20.3
  GB); ResNet-50 epoch 9 ×1 0.942194 / ×4 0.944665 (3,110 s); NextViT epoch 8 ×1 0.943229 / ×4
  0.946361 (3,664 s). The teacher table took about 4 h of prediction; its parity at τ 0.85 against
  `outputs/predictions/rebuild` passed. Only the ConvNeXt was submitted:
  **S193, ×4, τ 0.5, no mirror TTA, compiled CUDA-graph backbone with a Kaggle-built compile
  cache, FP16 weights: 0.960, 4th of five 0.960s**, hidden scoring 544 s (bundle
  `outputs/submission/exports/convnext-s188-student-x4-tau05-fast`).
- **Gen 4 (2026-10-22):** fold-0 ×1 / ×4, inflated by training on fold 0, v2 runs scored on v2
  targets and v3 runs on v3: `convnext-gen4` 0.945970 / 0.947657, `convnext-v3-gen4` 0.949640 /
  0.951167, `resnet50-gen4` 0.942347 / 0.944240, `resnet50-v3-gen4` 0.946678 / 0.948590,
  `nextvit-gen4` 0.948305 / 0.949971, `nextvit-v3-gen4` 0.950721 / 0.952660 (about 55–65 min each
  on the 3090). The v2v3 gen-4 models, trained side by side on a rented RTX PRO 6000 Max-Q (Vast
  57177970) in about 1 h 35 min: `convnext-v2v3-gen4` 0.946133 / 0.948109, `resnet50-v2v3-gen4`
  0.944133 / 0.945919, `nextvit-v2v3-gen4` 0.946361 / 0.947807. Their v2 pretraining start beat
  the v3 pretraining on v2 targets by about 0.004 and on v3 targets by 0.0002–0.0007 for all three
  families on Kaggle fold 0, where it never trained; the 12 gold studies there favoured v3 by
  +0.002–0.004.
- **S195, `resnet50-v2v3-gen4`, ×4, τ 0.85, no mirror TTA, accelerated like S193: 0.964, 2nd of
  thirteen 0.964s**, above every other single model at 0.964 (S167 4th); submit-to-scored 483 s
  (bundle `outputs/submission/exports/resnet50-v2v3-gen4-x4-tau085-fast`).
- The nine gen-4 models added to S188's 60 ranked below S188 at τ 0.4 and 0.43 (S196, S197).
