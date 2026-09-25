from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

COMPETITION = "rsna-knee-abnormality-detection"
KAGGLE_PYTHON = "3.12"
KAGGLE_PLATFORM = "manylinux_x86_64"
KAGGLE_WHEELS = Path("outputs/submission/dependencies/kaggle-v170")
CODEC_WHEELS = (
    "pylibjpeg==2.1.0",
    "pylibjpeg-libjpeg==2.4.0",
    "pylibjpeg-openjpeg==2.5.0",
)
KaggleId = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
]


class NotebookRun(BaseModel):
    accelerator: str = Field(default="NvidiaTeslaT4", description="Kaggle accelerator")
    notebook_timeout_seconds: int = Field(
        default=32400, gt=0, le=32400, description="Kaggle notebook time limit"
    )
    poll_seconds: int = Field(
        default=30, gt=0, description="Kaggle status poll interval"
    )


class SubmissionRequest(BaseModel):
    bundle: Path = Field(description="Ready exported bundle directory")
    dataset: KaggleId = Field(description="Private Kaggle dataset, owner/slug")
    notebook: KaggleId = Field(description="Kaggle notebook, owner/slug")
    record: Path = Field(description="New submission record JSON")
    label: str = Field(
        description="Submission title after the S number, 2 to 45 characters"
    )
    sequence: int | None = Field(
        default=None,
        gt=0,
        description="Submission number; defaults to one after the live submission count",
    )
