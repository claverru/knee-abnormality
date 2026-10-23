# Environment and dependency versions

The local `uv` environment is installed, using the Kaggle GPU v171 reference
and the explicitly documented additions below. It targets Linux x86-64
(including WSL2) with an NVIDIA GPU. Kaggle submissions are recorded in the
[inference notes](../inference/INDEX.md).

## Setup and daily use

From the repository root:

```bash
uv sync --locked
uv run --locked python
```

[`pyproject.toml`](../../pyproject.toml) owns the direct dependency pins and
dependency groups; [`uv.lock`](../../uv.lock) records the full resolved dependency
graph. [`.python-version`](../../.python-version) pins the local interpreter.
`uv sync --locked` installs that interpreter when needed and recreates `.venv`
without changing the lockfile. Run project commands with `uv run --locked`;
activation is optional.

`uv sync` installs dependencies and the local `knee_abnormality` package in
editable mode from `src/knee_abnormality/`, using the pinned setuptools build
backend. Scripts and notebooks can import the shared modules, and source
edits take effect without reinstalling. The [architecture](INDEX.md)
defines module, config and cache conventions. Inspect the starting settings
with `uv run --locked python -m knee_abnormality.config configs/resnet50.yaml`.

The `dev` group (Plotly, Ruff and just), `notebooks` group (IPython kernel and Matplotlib)
and `tools` group (Kaggle CLI and fold generation) install by default. JupyterLab is not installed.
To install just the core stack,
use `uv sync --locked --no-default-groups` and the same group flag on `uv run`.

When changing dependencies, edit them through `uv add`/`uv remove` or update
`pyproject.toml`, then run `uv lock` and `uv sync --locked`.
Version changes were reviewed against the Kaggle reference.
The CUDA index is restricted to Torch and TorchVision following
[uv's PyTorch guidance](https://docs.astral.sh/uv/guides/integration/pytorch/);
generic dependencies resolve from PyPI.

## Host

- Ubuntu 26.04 under WSL2, x86-64.
- NVIDIA RTX 3090, 24 GB VRAM, driver 610.88 when last checked; visible through `nvidia-smi`.
- `uv` and `uvx` 0.12.12, installed in `/home/claudio/.local/bin` using the
  [official installer](https://docs.astral.sh/uv/getting-started/installation/).
  Both commands resolve on the host PATH; shell startup files were not modified.
- The project uses uv-managed CPython 3.13.15 (`.python-version`), the version of the v171 image's
  `libpython3.13` (3.13.15-1+noble1).

## Kaggle reference

**Since 2026-10-17 the pins follow the v171 GPU image** (released 2026-10-01, on Kaggle notebooks
since about 2026-10-03; Ubuntu 24.04, Python 3.13), from its
[release diff](https://github.com/Kaggle/docker-python/releases/tag/v171-GPU-22be155a37fc52bc8254ebfd0bb894643d00a1283e2446e212597ae12aec45f9)
against v170: torch 2.11.0+cu128, torchvision 0.26.0+cu128, timm 1.0.29, numpy 2.1.3, pillow
12.3.0, pytorch-lightning 2.6.6 and kornia-rs 0.2.0 changed; the diff lists only changed packages, so
every other pin below kept its v170 version. pydantic-settings left the image (the submission bundles
it). Submission notebooks set no `docker_image`, so they run Kaggle's current default image. With
torch 2.11 the lock also moves to cuDNN 9.19.0.56, NCCL 2.28.9 and setuptools 81.0.0 (Kaggle pins
setuptools below 82); triton stays 3.6.0. Checked on the CPU in float32: ResNet-50, ConvNeXt-T,
FocalNet-T and EfficientViT-B2 checkpoints load strictly and their backbone outputs are bit-identical
to torch 2.10 with timm 1.0.26; MambaVision-T loads strictly.

The earlier reference was the public image `gcr.io/kaggle-gpu-images/python:v170` (Ubuntu 22.04, CUDA
toolkit 12.8.1, built 2026-06-29, Python 3.12, cu128 PyTorch wheels), immutable reference
`gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`.
Its versions came from the image's pinned GPU base requirements, PyTorch freeze and installed package
metadata; Polars 1.35.2 is in the v170 GPU base lock and the
[v167 GPU release](https://github.com/Kaggle/docker-python/releases/tag/v167-GPU-8777803d62f21da3e88e1062b5cc6973d9612768ca7bc59613d418958084e3cc),
with no later Polars change through v171. Versions were not inferred from unpinned requirements or
PyPI releases. Sources: [Kaggle's image repository](https://github.com/Kaggle/docker-python) and the
[v170 GPU release](https://github.com/Kaggle/docker-python/releases/tag/v170-GPU-bdf9e0538555f90453619adefb49ba40cfa136db44a9c9be7a42ea715c0aa068).

This is a reference for matching Python packages. It does not establish the
environment selected by a particular competition notebook or reproduce the
container's OS and system libraries inside the host virtual environment.

## Core dependencies

These installed dependencies support the implemented PyTorch/timm image baseline
and DICOM preparation workflow. Use Polars for tabular data processing.
The following pins match the inspected image; other transitive versions were resolved
by uv and are not claimed to match Kaggle's full environment.

| Package | Kaggle GPU v171 version | Purpose |
|---|---|---|
| `torch` | `2.11.0+cu128` | Training, GPU tensors, and inference |
| `torchvision` | `0.26.0+cu128` | Vision operations and timm dependency |
| `timm` | `1.0.29` | Pretrained vision backbones, including ReXNet candidates |
| `numpy` | `2.1.3` | Pixel arrays, preprocessing, and prediction arrays |
| `polars` | `1.35.2` | Labels, study manifests, and submission tables |
| `pydicom` | `3.0.2` | DICOM metadata and pixel access |
| `scikit-learn` | `1.6.1` | Validation splits and ROC-AUC calculations |
| `scipy` | `1.16.3` | Numerical support, including scikit-learn's dependency |
| `pillow` | `12.3.0` | Image support required by the vision stack |
| `tqdm` | `4.67.3` | Progress reporting for preprocessing and training |

Kornia 0.8.3 and kornia-rs 0.2.0 are used by the implemented affine, intensity and noise
augmentations. kornia-rs 0.2.0 is the v171 image's version; Kornia is not in the v171 diff.

MambaVision-T adds `einops` 0.8.1, `mambavision` 1.2.0 and `mamba-ssm` 2.3.2.post1 from Astral's GPU
index (CUDA 12.8, torch 2.11, cp313; T4 and 3090 code), with trimmed dependency metadata so that
neither pulls in `transformers` or its own pins; described in
[MambaVision-T dependencies](../modeling/backbones/mambavision-backbone.md#dependencies).

## Lightning training framework

Installed and locked as project dependencies, aligned with the inspected
Kaggle GPU image on 2026-09-11:

| Package | Version | Purpose |
|---|---|---|
| `pytorch-lightning` | `2.6.6` | Provides the project's `pytorch_lightning` namespace |
| `torchmetrics` | `1.9.0` | Lightning's metrics dependency |
| `lightning-utilities` | `0.15.3` | Shared Lightning utilities |

Kaggle's [v170 GPU release](https://github.com/Kaggle/docker-python/releases/tag/v170-GPU-bdf9e0538555f90453619adefb49ba40cfa136db44a9c9be7a42ea715c0aa068)
lists `pytorch-lightning==2.6.5`; v171 moved it to 2.6.6. The
[v167 GPU release](https://github.com/Kaggle/docker-python/releases/tag/v167-GPU-8777803d62f21da3e88e1062b5cc6973d9612768ca7bc59613d418958084e3cc)
lists TorchMetrics 1.9.0 and Lightning Utilities 0.15.3; the v168–v170 GPU diffs
record no subsequent changes to those two packages. These pins use published
release evidence rather than a fresh inspection of image layers.

The project previously installed the separate `lightning` distribution and
imported `lightning.pytorch`. Kaggle notebook version 1 showed that distribution
is absent from the selected image. The project now imports
`pytorch_lightning`, matching Kaggle's installed distribution and version, and
has removed `lightning` from the project and lockfile. The selected baseline3
checkpoint still loads, and full-fold inference reproduced every saved
probability exactly after the migration.

## Configuration dependencies

Installed and locked on 2026-09-10 as core project dependencies:

| Package | Version | Purpose |
|---|---|---|
| `pydantic` | `2.13.5` | Typed models and configuration validation |
| `pydantic-settings` | `2.15.0` | Parse explicit CLI arguments and validate CLI settings |
| `pyyaml` | `6.0.3` | Read experiment YAML with `safe_load` |

The Pydantic packages are additional local packages, not verified against the Kaggle image; the
experiment loader uses Pydantic models for the YAML and `pydantic-settings` for explicit CLI overrides,
with environment, dotenv and secrets sources disabled. See
[configuration behavior and commands](configs-and-experiment-artifacts.md#configs-and-development-commands).

## Notebook and competition tools

The default groups install Matplotlib, ipykernel, and the Kaggle CLI.
| Package | Version | Purpose |
|---|---|---|
| `matplotlib` | `3.10.0` | Inspect slices, preprocessing, and training curves |
| `ipykernel` | `6.17.1` | Run notebooks against the project interpreter |
| `kaggle` | `2.0.2` | Competition downloads and submission tooling |

The `tools` group also pins
[iterative-stratification 0.1.9](https://pypi.org/project/iterative-stratification/)
for offline multilabel fold generation. This is an added local preparation
dependency, not a Kaggle pin.

Do not add an experiment tracker, LLM stack, or extra DICOM codec solely because
Kaggle includes it. Compressed DICOM decoder wheels are bundled for submission
rather than installed in the training environment; their verification is
recorded in [inference](../inference/submissions/early-submission-details.md).

## Development tools

Installed and locked on 2026-09-10 in the default `dev` dependency group:

| Package | Version | Command | Purpose |
|---|---|---|---|
| `ruff` | `0.16.6` | `ruff` | Python linting and formatting |
| `rust-just` | `1.58.0` | `just` | Run recipes from a justfile |

The [just installation guide](https://just.systems/man/en/packages.html)
identifies `rust-just` as the package providing the command runner through
Python package managers. Both tools are managed inside the project environment:

```bash
uv run --locked ruff --version
uv run --locked just --version
```

These are local development pins, selected independently of the Kaggle GPU
image. Adding them preserved all previously locked package versions.
Both version commands and `uv lock --check` passed; `uv pip check` reported
all 126 installed packages compatible.

## Local verification

Checks completed on 2026-09-09 and 2026-09-10, before the Lightning installation: the original 13
Kaggle-reference pins matched the installed versions; a smoke script (later removed from the
repository, last in commit `fc90227`) passed a Polars Parquet round trip, synthetic uncompressed DICOM
decoding, scikit-learn scoring, a timm ResNet-18 BF16 forward/backward pass, an AdamW update and
TorchVision CUDA NMS on the RTX 3090, with random weights and synthetic targets; and
`uv run --locked kaggle --version` reported 2.0.2. These checks did not validate real MRI
preprocessing, compressed DICOM codecs, training throughput or accuracy.

## Confirm the selected Kaggle notebook

In the intended Kaggle GPU notebook, before adding or upgrading packages, this printed the inventory:

```python
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

print("Python:", platform.python_version())
for name in ("git_commit", "build_date"):
    path = Path("/etc") / name
    print(name, path.read_text().strip() if path.exists() else "unavailable")
for name in (
    "torch",
    "torchvision",
    "timm",
    "numpy",
    "polars",
    "pydicom",
    "scikit-learn",
    "scipy",
    "pillow",
    "tqdm",
    "matplotlib",
    "ipykernel",
    "jupyterlab",
    "kaggle",
    "lightning",
    "pytorch-lightning",
    "torchmetrics",
    "lightning-utilities",
    "pydantic",
    "pydantic-settings",
    "pyyaml",
):
    try:
        print(f"{name}=={version(name)}")
    except PackageNotFoundError:
        print(f"{name}: not installed")
```

Comparing that output with the local pins was the parity check before relying on a notebook image; when the image differed, the reference and project pins were updated and the lockfile regenerated.
