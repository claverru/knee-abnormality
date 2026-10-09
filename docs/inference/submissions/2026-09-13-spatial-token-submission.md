# Spatial-token submission

Submission **56206027** completed on 2026-09-13 with public score **0.929**,
confirmed through the Kaggle API. This improves on `crop140mm` at 0.925 by 0.004
and was subsequently surpassed by [3dsampling](2026-09-13-3dsampling-submission.md) at 0.930.
It uses the completed `spatial-tokens-five-embeddings` fold-0 run's selected
epoch-10 checkpoint, local macro AUC 0.88950565, with 2× image limits and
inference batch size two. See the
[validation comparison](../../validation/results/2026-09-13-spatial-token-validation-comparison.md).

The checkpoint SHA-256 is
`c58fd47aada2c32ab5cb0fee9f168e0271ebf68bdba05beed1c2dc260c26348f`.
The [bundle](../../../outputs/submission/exports/knee-spatial-tokens-five-embeddings-20260913)
passed metadata, strict CPU model loading and integrity checks; bundled source
matches the training snapshot. Source SHA-256 is
`6e68f3370f878715dad77d7af33cd95d22035865786f70e8e3b5895a7e7b731f`.
Optional local inference was not requested.

Private dataset `claverru/knee-spatial-tokens-five-embeddings-20260913`, version 1,
feeds private offline notebook `claverru/knee-spatial-tokens-20260913-inference`,
version 1 (Kaggle script version `349500954`). The notebook completed successfully,
and its three example rows passed output validation before that exact notebook
version was submitted. Completed competition scoring establishes successful
hidden-data execution; the API does not provide official inference runtime.
The runner recorded completion at 11:50:08 UTC in the
[workflow record](../../../outputs/submission/records/knee-spatial-tokens-five-embeddings-20260913-v2.json).

The initial longer notebook name was rejected with HTTP 400 after dataset
upload; the successful retry reused the same private dataset version. The
failed attempt created no competition submission. Exact export, both workflow
launches and monitoring commands are retained in the
[command log](../../../outputs/analysis/s/spatial-tokens-vs-crop140mm-20260913/submission-commands.md).
