# 2026-09-23 trained side mask and global token attention

Two fold-0 training runs against `k05-clean` (×4, τ 1 macro 0.902760), both at seed 2263738815 on
`all_5fold_scanner_v1` with combined labels. Neither was adopted; both fed the
[knee-geometry campaign](../../../autoresearch/campaigns/knee-geometry/INDEX.md), which adopted a
meniscus-only mask with a half-knee loss. They ran on the unmerged `sibling-masking` and
`token-attention` branches, retired 2026-09-24; their code is preserved as patches that apply to
`d914079`.

## Trained side mask (`side-mask`)

The four compartment labels (Medial/Lateral Meniscus and OA) attend only to own-side tokens plus
the ±10 mm midline band; unknown side or invalid geometry stays visible to both sides, and a
study-target with no own-side token falls back to all tokens. It is the trained counterpart of the
[inference-only masking](2026-09-23-sibling-masking.md).

- **Macro:** 0.903969, **+0.0012**, 80% paired report-group interval [−0.0004, +0.0028]; misses
  the adoption rule. Soft macro +0.0003. Epoch 8 selected (×1 0.90191), 1,583 s, 20.9 GB.
- **Families (95%):** meniscus **+0.0046** [+0.0005, +0.0090], OA **−0.0040** [−0.0087, +0.0002],
  other eight +0.0017 [−0.0014, +0.0051].
- **Strata** (AUC of all positives against one negative group, change, 80%):

| Label | Sibling clean | Sibling positive |
|---|---:|---:|
| Medial Meniscus | +0.0031 [+0.0003, +0.0059] | +0.0056 [−0.0000, +0.0110] |
| Lateral Meniscus | +0.0063 [+0.0022, +0.0108] | +0.0047 [−0.0012, +0.0110] |
| Medial OA | −0.0038 [−0.0067, −0.0010] | +0.0003 [−0.0080, +0.0085] |
| Lateral OA | −0.0034 [−0.0080, +0.0011] | −0.0093 [−0.0169, −0.0018] |

- **Learned anatomical strengths** fell from 1.47 to 1.21 (meniscus) and 1.12 to 0.98 (OA).
- **Reading:** trained blind, the menisci improve on both negative groups; OA loses, including the
  sibling-positive group (hypothesis: OA is a knee-level degeneration whose signs on the other side
  are real evidence). This split motivated the adopted meniscus-only mask.
- **Not run:** `side_mask_shared_head`, a hard tie of one classifier per family on the masked
  pools.
- **Artifacts:** `outputs/training/side-mask/`, `outputs/analysis/side-mask-vs-k05-clean/`
  (`side_mask.patch`).

## Global token attention (`token-attention`)

One pre-norm transformer block (4 heads, MLP 1024, zero-initialized outputs) over each study's
visually valid tokens before the attention pooling, with 3D RoPE on raw patient LPS in mm
(10 frequencies per axis, wavelengths 8–512 mm; relative, not side-normalized). The 1D and 2D
position embeddings are dropped; reflected studies negate x.

- **Macro:** 0.901284, **−0.0015**, 95% [−0.0054, +0.0022].
- **Families:** meniscus +0.0027, OA +0.0035, other eight −0.0038; PF OA −0.0152, Fracture
  −0.0153, MCL −0.0059.
- **Reading:** the attention block and the dropped embeddings are confounded. The losses sit on
  labels that need absolute position (patellofemoral, fracture), consistent with relative-only
  position.
- **Artifacts:** `outputs/training/token-attention/`, `outputs/analysis/token-attention-vs-k05-clean/`
  (`comparison.json`, `token_attention.patch`).
