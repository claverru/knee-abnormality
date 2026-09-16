# Baseline3 fold-0 submission artifact

Local preparation on 2026-09-11 selected
`outputs/training/baseline3/fold_0/best-epoch=07-val_macro_auc=0.84982.ckpt` (deleted on 2026-09-16), SHA-256
`b955aefe710905e47abf8e5ec8f65d2c6ec009108e61eeae02cb448d31bcaf54`.
The exported state dictionary has the same 341 keys and tensor values as the
training checkpoint, with the same fold and ordered targets. Fresh BF16
inference reproduced all 903 saved validation rows exactly, with maximum
absolute probability difference `0.0` and macro-AUC
`0.8498192406240211`. Prediction took 37.42 seconds on the local RTX 3090,
or 24.13 studies per second, excluding the 0.76-second model load.

The three-study example prediction matches the official 13-column schema,
study order and probability range. Repository, frozen-bundle and generated
notebook execution produced identical probabilities. The notebook installed
its wheels with index access disabled, reported 0.69 seconds for setup and
5.43 seconds for the prediction subprocess, and validated its output. These are
local notebook measurements.

Kaggle notebook version 3 installed the same wheels offline and completed on a
CUDA runtime using FP32. Its three-study prediction took 11.54 seconds and the
whole visible notebook took 82.12 seconds. This validates the visible Kaggle
path, not hidden-test throughput.

The Python 3.12 manylinux2014 x86-64 bundle contains `pylibjpeg==2.1.0`,
`pylibjpeg-libjpeg==2.4.0` and `pylibjpeg-openjpeg==2.5.0`. Those exact wheels
decoded pydicom's real JPEG Lossless fixture with transfer syntax
`1.2.840.10008.1.2.4.70` and JPEG 2000 Lossless fixture with transfer syntax
`1.2.840.10008.1.2.4.90`. A complete header scan found all 819,078 local
training DICOMs use uncompressed Explicit VR Little Endian, so no local
competition file could exercise the compressed paths. Hidden-test codec mix and
full runtime remain remote checks.

The submitted bundle is `outputs/submission/exports/baseline3-fold0-v3`, with bundled-source
SHA-256 `f4976ae171bc3c363e1aba55356a6d9bd1689c37d6ad6ed7d2e184e298e63578`
and no unmet manifest prerequisites. The remote workflow started on 2026-09-11:
private Dataset version 2 is ready and offline notebook version 3 completed.
Its validated output was submitted as `56164711` and scored **0.878** publicly;
the state record is
`outputs/submission/records/baseline3-fold0-20260911-v6.json`.

After that submission launched, the local project migrated from the absent
`lightning` distribution to Kaggle's installed
`pytorch-lightning==2.6.5`. Full validation inference again matched all 903
saved rows exactly. The ready `outputs/submission/exports/baseline3-fold0-v5` bundle has
bundled-source SHA-256
`4a270850d0594612ec913504f38dd5e988f349ea6fa7e67fc70cc4c1922e54f0`,
records `pytorch-lightning==2.6.5`, and contains only the three offline DICOM
codec wheels. Its generated notebook completed locally without the
`lightning` distribution.

Training and prediction both select PyTorch's `high` float32 matrix
multiplication policy. This keeps the float32 target-attention calculation
consistent while BF16 autocast remains active for CUDA inference.

Baseline4's ReXNet-200 epoch-7 checkpoint reproduced all 903 saved validation
probabilities exactly at **0.872510 macro AUC**. The ready wheel-free bundle is
`outputs/submission/exports/baseline4-fold0-v1`, with bundled-source SHA-256
`6cedf5a92eed77fe40a6064eb55357d5266cfdf84f364c45a488b5865a7e258b`.
Kaggle notebook version 1 completed its visible three-study run in 54.14
seconds and submission `56167669` scored **0.910** publicly. Kaggle scoring was
observed complete 23 minutes 41 seconds after submission; that interval
includes queue and evaluation time and is accurate only to the runner's
30-second polling interval. The state record is
`outputs/submission/records/baseline4-fold0-20260911.json`.
