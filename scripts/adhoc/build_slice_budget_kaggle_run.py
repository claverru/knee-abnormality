import json
import shutil
from pathlib import Path

from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.kaggle import COMPETITION, KaggleId

SOURCE = Path(__file__).parent / "slice_budget_kaggle"
SOURCES = ("sitecustomize.py", "run_passes.py")
NOTEBOOK = """import json
import os
import subprocess
import sys
import time
from pathlib import Path

SOURCES = __SOURCES__
started = time.perf_counter()
subprocess.run(["nvidia-smi"], check=False)
input_root = Path("/kaggle/input")


def locate(name):
    matches = sorted(
        {
            path
            for pattern in (f"*/{name}", f"*/*/{name}", f"*/*/*/{name}")
            for path in input_root.glob(pattern)
        }
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one mounted {name}, found {matches}")
    return matches[0].parent


bundle = locate("manifest.json")
inputs = locate("protocol.json")
wheels = sorted((bundle / "wheels").glob("*.whl"))
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"]
    + [str(wheel) for wheel in wheels],
    check=True,
)
work = Path("/tmp/knee-slice-budget")
patch = work / "patch"
patch.mkdir(parents=True, exist_ok=True)
(patch / "sitecustomize.py").write_text(SOURCES["sitecustomize.py"])
(work / "run_passes.py").write_text(SOURCES["run_passes.py"])
environment = os.environ.copy()
environment["PYTHONPATH"] = str(bundle / "code")
subprocess.run(
    [
        sys.executable,
        str(work / "run_passes.py"),
        "--bundle",
        str(bundle),
        "--inputs",
        str(inputs),
        "--patch-dir",
        str(patch),
        "--adapter",
        str(work / "data"),
        "--output-dir",
        "/kaggle/working",
    ],
    check=True,
    env=environment,
)
print(json.dumps({"notebook_seconds": time.perf_counter() - started}))
"""


class Settings(CLISettings):
    model_config = describe(
        "Build the Kaggle folders of the slice-budget sensitivity run: a private "
        "inputs dataset with the fold definition and label table, and a notebook "
        "that predicts fold-0 validation with each series type's budget halved and "
        "doubled from an attached bundle."
    )

    output_dir: Path = Field(description="Directory for the dataset and notebook")
    bundle: Path = Field(description="Local copy of the attached bundle")
    bundle_dataset: KaggleId = Field(description="Kaggle dataset holding the bundle")
    label_manifest: Path = Field(description="Label manifest of the bundle's run")
    data_root: Path = Field(default=Path("data"), description="Local data root")
    inputs_dataset: KaggleId = Field(
        default="claverru/knee-slice-budget-inputs",
        description="Private Kaggle dataset to create for the inputs",
    )
    notebook: KaggleId = Field(
        default="claverru/knee-slice-budget-sensitivity",
        description="Kaggle notebook to push",
    )


def build_dataset(settings: Settings) -> Path:
    directory = settings.output_dir / "dataset"
    directory.mkdir(parents=True, exist_ok=True)
    experiment = json.loads((settings.bundle / "config.json").read_text())["experiment"]
    split = settings.data_root / "folds" / experiment["validation"]["split"]
    for name in ("studies.csv", "targets.csv", "protocol.json"):
        shutil.copyfile(split / name, directory / name)
    manifest = json.loads(settings.label_manifest.read_text())
    (record,) = [
        file
        for source in manifest["sources"]
        for file in source["files"]
        if file.get("label_table") == experiment["labels"]["source"]
    ]
    shutil.copyfile(
        settings.data_root.joinpath(*Path(record["path"]).parts[1:]),
        directory / "labels.csv",
    )
    shutil.copyfile(settings.label_manifest, directory / settings.label_manifest.name)
    metadata = {
        "id": settings.inputs_dataset,
        "title": "Knee slice budget inputs",
        "licenses": [{"name": "other"}],
        "description": "Private fold definition and label table for the "
        "slice-budget sensitivity measurement",
    }
    (directory / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return directory


def build_notebook(settings: Settings) -> Path:
    directory = settings.output_dir / "notebook"
    directory.mkdir(parents=True, exist_ok=True)
    sources = {name: (SOURCE / name).read_text() for name in SOURCES}
    code = NOTEBOOK.replace("__SOURCES__", repr(sources))
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(keepends=True),
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (directory / "slice_budget.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    metadata = {
        "id": settings.notebook,
        "title": "Knee slice budget sensitivity",
        "code_file": "slice_budget.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "competition_sources": [COMPETITION],
        "dataset_sources": [settings.bundle_dataset, settings.inputs_dataset],
    }
    (directory / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return directory


def main() -> None:
    settings = Settings()
    print(
        json.dumps(
            {
                "dataset": str(build_dataset(settings)),
                "notebook": str(build_notebook(settings)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
