import subprocess
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import CliApp, CliSubCommand

from knee_abnormality.cli import CLISettings, command, describe

ROOT = Path(__file__).resolve().parents[2]
BUILDERS = ROOT / "scripts/adhoc/meniscus_masks"
SEGMENTER_PYTHON = Path.home() / ".cache/knee-segmenters/venv/bin/python"


def run(module: str, *arguments: str) -> None:
    subprocess.run(
        [str(SEGMENTER_PYTHON), str(BUILDERS / f"{module}.py"), *arguments],
        cwd=ROOT,
        check=True,
    )


class Step(BaseModel):
    name: str = Field(
        default="dess-menisci",
        description="Working folder under ~/.cache/knee-segmenters",
    )


class Convert(Step):
    model_config = describe(
        "Convert the DESS series with most slices of every study that has one into "
        "one volume per study on the CPU: 0.5 mm in-plane, cropped to the body with a "
        "10 mm margin and turned to the segmenter's axis order"
    )
    workers: int = Field(default=4, gt=0, description="Conversion processes")

    def cli_cmd(self) -> None:
        run("convert_dess", self.name, str(self.workers))


class Segment(Step):
    model_config = describe(
        "Segment the converted volumes with aagatti/nnunet_knee on the GPU "
        "(full-resolution model, fold 1, no mirroring); volumes already segmented "
        "are skipped"
    )

    def cli_cmd(self) -> None:
        run("segment_dess", self.name)


class Export(Step):
    model_config = describe(
        "Test every meniscus mask without labels (volume, one main piece, next to "
        "the tibia and the femur of the same mask, medial nearer the body's midline) "
        "and write the accepted knees as 1 mm volumes in patient coordinates under "
        "data/labels/prepared, with the per-knee table of the test"
    )
    target: str = Field(
        default="meniscus-nnunet-knee-dess265",
        description="Target folder under data/labels/prepared",
    )
    workers: int = Field(default=4, gt=0, description="Export processes")

    def cli_cmd(self) -> None:
        run("export_dess", self.name, self.target, str(self.workers))


class MeniscusTargetsCLI(CLISettings):
    model_config = describe(
        "Build meniscus masks from the 3D DESS series with aagatti/nnunet_knee, "
        "one step per subcommand"
    )
    convert: CliSubCommand[Convert] = command(Convert)
    segment: CliSubCommand[Segment] = command(Segment)
    export: CliSubCommand[Export] = command(Export)

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    CliApp.run(MeniscusTargetsCLI)


if __name__ == "__main__":
    main()
