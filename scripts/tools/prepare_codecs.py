import subprocess
from pathlib import Path

from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.kaggle import CODEC_WHEELS, KAGGLE_PYTHON, KAGGLE_WHEELS


class PrepareCodecsCLI(CLISettings):
    model_config = describe(
        "Download the pinned offline DICOM codec wheels for the Kaggle Python runtime"
    )

    output: Path = Field(
        default=KAGGLE_WHEELS, description="Wheel directory that export bundles"
    )


def main() -> None:
    args = PrepareCodecsCLI()
    subprocess.run(
        [
            "uvx",
            "--from",
            "pip==25.1.1",
            "pip",
            "download",
            "--no-deps",
            "--only-binary=:all:",
            "--python-version",
            KAGGLE_PYTHON,
            "--platform",
            "manylinux2014_x86_64",
            "--dest",
            str(args.output),
            *CODEC_WHEELS,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
