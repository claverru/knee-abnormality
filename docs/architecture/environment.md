# Environment and dependency versions

The local `uv` environment is installed, using the Kaggle GPU v170 reference
and the explicitly documented additions below. It targets Linux x86-64
(including WSL2) with an NVIDIA GPU. The visible Kaggle notebook run is recorded
in the [inference notes](../inference/INDEX.md); hidden-run parity remains a
separate check.

## Setup and daily use

From the repository root:

```bash
rtk proxy uv sync --locked
rtk proxy uv run --locked python
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
with `rtk proxy uv run --locked python -m knee_abnormality.config configs/training.yaml`.

The `dev` group (Ruff and just), `notebooks` group (IPython kernel and plotting),
and `tools` group (Kaggle CLI and fold generation) install by default.
Select `.venv/bin/python` as the
notebook interpreter in your editor. JupyterLab is not installed. To install just the core stack,
use `uv sync --locked --no-default-groups` and the same group flag on `uv run`.

When changing dependencies, edit them through `uv add`/`uv remove` or update
`pyproject.toml`, then run `uv lock` and `uv sync --locked`.
Review version changes against the Kaggle reference and update this document.
The CUDA index is restricted to Torch and TorchVision following
[uv's PyTorch guidance](https://docs.astral.sh/uv/guides/integration/pytorch/);
generic dependencies resolve from PyPI.

## Host

- Ubuntu 26.04 under WSL2, x86-64.
- NVIDIA RTX 3090, 24 GB VRAM, driver 591.86; visible through `nvidia-smi`.
- `uv` and `uvx` 0.12.12, installed in `/home/claudio/.local/bin` using the
  [official installer](https://docs.astral.sh/uv/getting-started/installation/).
  Both commands resolve on the host PATH; shell startup files were not modified.
- System Python remains 3.14.4. The project uses uv-managed CPython 3.12.14,
  matching the published image's Python minor version. This local patch
  version has not been verified against Kaggle's interpreter.

## Kaggle reference

The inspected public image is `gcr.io/kaggle-gpu-images/python:v170`, also
tagged `latest` when checked. Its immutable reference is:

```text
gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461
```

Image metadata identifies Ubuntu 22.04, CUDA toolkit 12.8.1, build date
2026-06-29, and source commit `fc61d5cda7da39530055bae9bd0e92865f995cd9`.
Python is 3.12; its exact patch version was not established by this inspection.
PyTorch's wheel build is CUDA 12.8 (`cu128`).

Sources: [Kaggle's image repository](https://github.com/Kaggle/docker-python),
[v170 GPU release](https://github.com/Kaggle/docker-python/releases/tag/v170-GPU-bdf9e0538555f90453619adefb49ba40cfa136db44a9c9be7a42ea715c0aa068),
and the [Dockerfile at the image's source commit](https://github.com/Kaggle/docker-python/blob/fc61d5cda7da39530055bae9bd0e92865f995cd9/Dockerfile.tmpl).
The release also names a private runtime image with a different digest; the
digest above belongs to the public image actually inspected.

The original core versions were checked using the image's pinned GPU base
requirements, PyTorch freeze, and installed package metadata in its later
Python-install layers. Polars 1.35.2 is present in the v170 GPU base lock and
the [v167 GPU release](https://github.com/Kaggle/docker-python/releases/tag/v167-GPU-8777803d62f21da3e88e1062b5cc6973d9612768ca7bc59613d418958084e3cc);
the v168–v170 GPU release diffs report no subsequent Polars version change.
Versions are not inferred from unpinned requirements or current PyPI releases.

This is a reference for matching Python packages. It does not establish the
environment selected by a particular competition notebook or reproduce the
container's OS and system libraries inside the host virtual environment.

## Core dependencies

These installed dependencies support the implemented PyTorch/timm image baseline
and DICOM preparation workflow. Use Polars for tabular data processing.
The following pins match the inspected image; other transitive versions were resolved
by uv and are not claimed to match Kaggle's full environment.

| Package | Kaggle GPU v170 version | Purpose |
|---|---|---|
| `torch` | `2.10.0+cu128` | Training, GPU tensors, and inference |
| `torchvision` | `0.25.0+cu128` | Vision operations and timm dependency |
| `timm` | `1.0.26` | Pretrained vision backbones, including ReXNet candidates |
| `numpy` | `2.0.2` | Pixel arrays, preprocessing, and prediction arrays |
| `polars` | `1.35.2` | Labels, study manifests, and submission tables |
| `pydicom` | `3.0.2` | DICOM metadata and pixel access |
| `scikit-learn` | `1.6.1` | Validation splits and ROC-AUC calculations |
| `scipy` | `1.16.3` | Numerical support, including scikit-learn's dependency |
| `pillow` | `11.3.0` | Image support required by the vision stack |
| `tqdm` | `4.67.3` | Progress reporting for preprocessing and training |

Kornia 0.8.3 and kornia-rs 0.1.14 are additional local dependencies used by
the implemented affine, intensity and noise augmentations. They are pinned in
`pyproject.toml`; matching Kaggle-image versions have not been established.

## Lightning training framework

Installed and locked as project dependencies, aligned with the inspected
Kaggle GPU image on 2026-09-11:

| Package | Version | Purpose |
|---|---|---|
| `pytorch-lightning` | `2.6.5` | Provides the project's `pytorch_lightning` namespace |
| `torchmetrics` | `1.9.0` | Lightning's metrics dependency |
| `lightning-utilities` | `0.15.3` | Shared Lightning utilities |

Kaggle's [v170 GPU release](https://github.com/Kaggle/docker-python/releases/tag/v170-GPU-bdf9e0538555f90453619adefb49ba40cfa136db44a9c9be7a42ea715c0aa068)
lists `pytorch-lightning==2.6.5`. The
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

The Pydantic packages are additional local packages; their versions have not
been verified against the Kaggle GPU v170 image. Adding them preserved all
previously locked package versions. At installation, `uv lock --check` passed,
and `uv pip check` reported all 124 installed packages compatible.

The YAML migration promoted the existing transitive PyYAML 6.0.3 pin to a
direct dependency, preserving all locked package versions. The experiment
loader uses Pydantic models for the YAML and `pydantic-settings` for explicit
CLI overrides, with environment, dotenv and secrets sources disabled. See
[configuration behavior and commands](configs-and-experiment-artifacts.md#configs-and-development-commands).

## Notebook and competition tools

The default groups install Matplotlib, ipykernel, and the Kaggle CLI.
JupyterLab's reference version is retained for a future browser interface;
it is not currently declared in the project or lockfile.

| Package | Kaggle GPU v170 version | Purpose |
|---|---|---|
| `matplotlib` | `3.10.0` | Inspect slices, preprocessing, and training curves |
| `ipykernel` | `6.17.1` | Run notebooks against the project interpreter |
| `jupyterlab` | `3.6.8` | Browser notebook interface, if used locally |
| `kaggle` | `2.0.2` | Competition downloads and submission tooling |

The `tools` group also pins
[iterative-stratification 0.1.9](https://pypi.org/project/iterative-stratification/)
for offline multilabel fold generation. This is an added local preparation
dependency, not a verified Kaggle v170 pin. It adds no new transitive packages
to the existing stack.

Do not add an experiment tracker, LLM stack, or extra DICOM codec solely because
Kaggle includes it. Compressed DICOM decoder wheels are bundled for submission
rather than installed in the training environment; their verification is
recorded in [inference](../inference/submissions/2026-09-11-baseline3-fold-0-submission-artifact.md).

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
rtk proxy uv run --locked ruff --version
rtk proxy uv run --locked just --version
```

These are local development pins, selected independently of the Kaggle GPU
image. Adding them preserved all previously locked package versions.
Both version commands and `uv lock --check` passed; `uv pip check` reported
all 126 installed packages compatible.

## Local verification

Historical checks completed on 2026-09-09, before the Lightning installation:

- The original 13 Kaggle-reference pins matched installed versions. After
  adding the stratifier, `uv pip check` found no incompatibilities among the
  107 installed packages.
- [`scripts/adhoc/check_environment.py`](../../scripts/adhoc/check_environment.py) passed:
  Polars Parquet round-trip, synthetic uncompressed DICOM pixel decoding,
  scikit-learn scoring, a timm ResNet-18 BF16 forward/backward pass, an AdamW
  weight update, and TorchVision CUDA NMS on the RTX 3090. The smoke model
  uses random weights and synthetic targets, not a competition configuration.
- The project notebook kernel started and executed Polars code; Matplotlib
  rendered a PNG without a display; `uv run --locked kaggle --version`
  reported 2.0.2. Competition file listing and five individual CSV downloads
  subsequently succeeded; see the [metadata audit](../metadata/INDEX.md).

The smoke script downloads no weights or data and uses FP32 on GPUs without
BF16 support. It does not validate real MRI preprocessing, compressed DICOM
codecs, training throughput, or competition accuracy.

## Confirm the selected Kaggle notebook

In the intended Kaggle GPU notebook, before adding or upgrading packages, run:

```python
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

print("Python:", platform.python_version())
for name in ("git_commit", "build_date"):
    path = Path("/etc") / name
    print(name, path.read_text().strip() if path.exists() else "unavailable")
for name in (
    "torch", "torchvision", "timm", "numpy", "polars", "pydicom",
    "scikit-learn", "scipy", "pillow", "tqdm", "matplotlib",
    "ipykernel", "jupyterlab", "kaggle", "lightning", "pytorch-lightning",
    "torchmetrics", "lightning-utilities", "pydantic", "pydantic-settings",
    "pyyaml",
):
    try:
        print(f"{name}=={version(name)}")
    except PackageNotFoundError:
        print(f"{name}: not installed")
```

Compare that output with the local pins before relying on notebook parity.
If the selected image differs, update the reference and project pins, regenerate
the lockfile. Follow [project guidance](../../AGENTS.md) for subsequent work;
the checks recorded above are historical. Update this document when the
target notebook image or project dependency choices change.
