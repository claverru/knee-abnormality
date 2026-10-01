# RadImageNet-pretrained weights

Public weights trained on the [RadImageNet](https://www.radimagenet.com/) corpus (about 1.35M CT,
MRI and ultrasound slices, 165 classes, 11 anatomical regions) or on its VQA derivative. Surveyed
2026-09-28; Hub revisions and licences come from the Hub API on that date. Only the files marked
**inspected** were downloaded and loaded (with `weights_only=True`); the rest are card-level
claims. The ResNet-50 in use is the [RadImageNet backbone](../ongoing/radimagenet-backbone.md)
line; ViT integration costs are in the [encoder survey](huggingface-image-encoders.md#radiodino-and-omnirad).

## Summary

| Weights | Kind | Where | Licence | Fits our pipeline |
|---|---|---|---|---|
| Original ResNet-50, DenseNet-121, InceptionV3, InceptionResNetV2 | Supervised CNNs | Official Google Drive, mirrors below | MIT (repository) | ResNet-50 **in use** |
| Refined-RadImageNet ResNet-10t, ResNet-18, ResNet-50 | Supervised CNNs, timm definitions | `convergedmachine/RadImagenet` | MIT (Hub), Apache-2.0 (GitHub) | ResNet-18 **in use** (control), ResNet-50 **in use** (submissions) |
| RadioDINO-S/16, S/8, B/16 | DINO self-supervised ViTs | `Snarcy/RadioDino-*` | CC BY 4.0 | Need a patch-token adapter |
| OmniRad-small, OmniRad-base | DINOv2-style self-supervised ViTs | `Snarcy/OmniRad-*` | CC BY 4.0 | Need a patch-token adapter |
| RadInSight | LoRA adapters for Llama-3.2-11B-Vision | `ahmetcangunay/RadInSight` | Apache-2.0 | Not a backbone; VLM uses only |
| MobileNetV3-Small RadioCaptioner | Captioning model | `erfanasgharieyan/mobilenet_radiocaptioner` | Unknown | Not a backbone; repository returned 401 |

## Original RadImageNet CNNs

The [official repository](https://github.com/BMEII-AI/RadImageNet) (MIT) releases ResNet-50,
DenseNet-121, InceptionV3 and InceptionResNetV2 as Keras files on Google Drive, plus a PyTorch
conversion on [Google Drive](https://drive.google.com/file/d/1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR/view?usp=sharing).
Swin Transformer weights are announced but not released. The official fine-tuning example
([`acl/acl_train.py`](https://github.com/BMEII-AI/RadImageNet/blob/main/acl/acl_train.py)) uses
Keras `preprocess_input` then `rescale=1/255`.

- **Kaggle mirror, ResNet-50, inspected:**
  [`marwanmath/resnet-50-radimagenet-marwan`](https://www.kaggle.com/datasets/marwanmath/resnet-50-radimagenet-marwan)
  `ResNet50.pt`, 94,345,733 bytes, SHA-256
  `08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734`. A plain state dict of 318
  tensors (`backbone.{0,1,4–7}.*`, torchvision layout without `fc`). It loads strictly into timm
  `resnet50.tv_in1k` after renaming, and is the file the public 0.943 stack pins. Listed as
  CC-BY-NC-SA-4.0. Measured on 2026-09-28:
  - The convolution-weight RMS is 0.314 in every layer (0.30–0.36), about 20× torchvision's ImageNet
    ResNet-50.
  - [−1, 1] input scaling reproduces its first BatchNorm's running statistics on knee slices, and the
    Keras example's normalization does not
    ([details](../ongoing/radimagenet-backbone.md#what-is-implemented)).
- **Hugging Face mirror:** [`Lab-Rasool/RadImageNet`](https://huggingface.co/Lab-Rasool/RadImageNet)
  (MIT, revision `14460ee4c1276f6925611a63aa9a54a05d39eae0`) has `ResNet50.pt` (94.3 MB),
  `DenseNet121.pt` (28.4 MB) and `InceptionV3.pt` (87.5 MB). Its README loads them with
  `torch.load` as whole pickled models, which needs unsafe unpickling. The ResNet-50 file is the same
  size as the Kaggle one; not inspected.

## Refined-RadImageNet ResNets

[`convergedmachine/RadImagenet`](https://huggingface.co/convergedmachine/RadImagenet), revision
`7fa62dccee2b14e49188d11c4b0e66d88c29b41e`, copied from `convergingmachine/RadImagenet`. The
[GitHub repository](https://github.com/convergedmachine/refined-radimagenet) re-organizes RadImageNet
into a "refined and stratified" 165-class set without further detail, and reports RadImageNet macro
F1 of 0.3998 (ResNet-10t), 0.4545 (ResNet-18) and 0.5097 (ResNet-50). The related paper is
[arXiv 2406.14568](https://arxiv.org/abs/2406.14568). No downstream results are reported.

Inspected 2026-09-28: `resnet10t.pth` (40,126,400 bytes, SHA-256
`f5219dff128ca752d47e8acd07817564c10bacbbc4241b3a31845e1a67a079ab`) and `resnet18.pth` (90,192,118
bytes, SHA-256 `c8c9259ac24068d1986bce1d1828f32b396642fc5ac86d17ab86baa0a13d344f`). Each is a
training checkpoint with `model`, `optimizer`, `lr_scheduler`, `epoch` and `args`; the optimizer
state explains the file sizes. Inspected 2026-10-01: `resnet50.pth` (191,145,887 bytes, SHA-256
`b6daa90de92589c404e9da30f4c0ce49182763d4a952f1222caeec647237bacf`, epoch 89, same recipe,
median convolution RMS 0.0136), loading strictly into timm `resnet50` the same way; it is
`configs/refined-resnet50.yaml` (`resnet50.a1_in1k`, cached as
`refined_radimagenet_resnet50.pth`).

- **Loading:** `weights_only=True` works once `argparse.Namespace` is allow-listed. Model keys carry
  the `torch.compile` prefix `_orig_mod.`. Stripped, they load strictly into timm `resnet10t` and
  `resnet18` with `num_classes=165` and three input channels.
- **Recipe (from `args`):** torchvision's reference classification script, run on 8 GPUs:
  - SGD, lr 0.1, momentum 0.9, weight decay 1e-4, step decay ×0.1 every 30 epochs, 90 epochs;
  - AutoAugment `imagenet` and CutMix 1.0;
  - train crop 224, validation resize 256 with crop 224, bilinear interpolation.

  The saved epochs are 89 (ResNet-10t) and 83 (ResNet-18). That script normalizes with ImageNet
  mean/std by default. A BatchNorm check on knee slices matched the variance under ImageNet
  normalization (ratio 1.56) but no candidate matched the means, most likely because the training
  images are brighter than our quantile-normalized slices. So normalization rests on the recipe,
  not on that check.
- **Weight scale:** convolution RMS 0.0158 (ResNet-10t) and 0.0124 (ResNet-18), the same range as
  torchvision's ImageNet ResNet-50 (0.0159). The 20× scale problem of the original weights does
  not apply.
- **Integration:**
  - ResNet-18 uses basic residual blocks and ResNet-10t a three-convolution deep stem with
    average-pool downsampling; `_resnet_mapping` handles both.
  - Both run at roughly ReXNet-200's cost, about 1.8 and 1.1 GMACs per 224 px triplet against 1.53.
  - `resnet50.pth` (191 MB) is not inspected.
  - Integrated on branch `radimagenet` as `resnet18.a1_in1k` and `resnet10t.c3_in1k`
    ([RadImageNet backbone](../ongoing/radimagenet-backbone.md#refined-radimagenet-resnet-18-and-resnet-10t-integrated-2026-09-28-untrained)).

## Self-supervised ViTs: RadioDINO and OmniRad

All are CC BY 4.0, safetensors, loadable as `timm.create_model("hf_hub:Snarcy/<name>")`, 224 px
with ImageNet mean/std according to their cards. RadioDINO is from
[Zedda, Loddo and Di Ruberto, 2025](https://doi.org/10.1016/j.compbiomed.2025.110583); OmniRad's
code is on [GitHub](https://github.com/unica-visual-intelligence-lab/OmniRad).

| Model | Revision | Architecture | Tokens at 224 px | Card results |
|---|---|---|---:|---|
| [RadioDino-s16](https://huggingface.co/Snarcy/RadioDino-s16) | `773d28d8…` | ViT-S/16, DINO, 384-d | 196 | Already in the [encoder survey](huggingface-image-encoders.md#practical-2d-shortlist) |
| [RadioDino-s8](https://huggingface.co/Snarcy/RadioDino-s8) | `710fdbb9f3a4be699e39496e56cc0662664e6853` | ViT-S/8, DINO, 21.7M params | 784 | MedMNIST F1 75–96 |
| [RadioDino-b16](https://huggingface.co/Snarcy/RadioDino-b16) | `2c0994dbed5e1c0fa257974b54f941cd6f8cb511` | ViT-B/16, DINO, 85.8M params | 196 | MedMNIST F1 78–97, BUSI 91.7 |
| [OmniRad-small](https://huggingface.co/Snarcy/OmniRad-small) | `4650430a…` | ViT-S/14, DINOv2-style, 21.6M params | 256 | Already surveyed |
| [OmniRad-base](https://huggingface.co/Snarcy/OmniRad-base) | `3e42381b…` | ViT-B/14, DINOv2-style without local crops, 85.7M params | 256 | No numbers on the card |

None has knee-task evidence. All need the patch-token adaptation and spatial geometry described in
the [encoder survey](huggingface-image-encoders.md#radiodino-and-omnirad).

## Vision-language weights

These are not image backbones. They are recorded for other uses, such as zero-shot image
annotation or teacher labels; such uses are hypotheses, and none is evaluated.

- **RadInSight:** [`ahmetcangunay/RadInSight`](https://huggingface.co/ahmetcangunay/RadInSight),
  Apache-2.0, revision `47f7605a783962db888b0aecd22b2c6ca728eae5`. LoRA adapters
  (`adapter_model.safetensors`) for `unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit`, trained with
  Unsloth in two stages (alignment, then instruction) on RadImageNet-VQA (1.68M items) for
  multi-turn radiology visual question answering. The card reports training losses only, no
  benchmarks; it needs the 11B base model.
- **MobileNetV3-Small RadioCaptioner:** `erfanasgharieyan/mobilenet_radiocaptioner`,
  `pytorch_model.bin`, reportedly a captioning model trained on RadImageNet-VQA. The Hub returned
  401 on 2026-09-28 (gated or removed), so nothing about it is verified.
