import json
from pathlib import Path

import numpy as np
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe

TYPES = (
    "Axial T1/PD",
    "Axial fluid",
    "Coronal T1/PD",
    "Coronal fluid",
    "Sagittal T1/PD",
    "Sagittal fluid",
)
QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
WINDOWS = ((93, 93), (93, 67), (80, 67), (67, 67), (60, 60))


class Settings(CLISettings):
    model_config = describe(
        "Summarise an input-saliency dump per series type: where the sensitivity of "
        "each label group sits and how much of it fixed windows hold"
    )

    dump_dir: Path = Field(description="Directory written by dump_input_saliency")
    vertical_centre_mm: float = Field(
        default=10.0, description="Window centre below the image centre"
    )
    sagittal_centre_mm: float = Field(
        default=7.0, description="Sagittal window centre anterior of the knee centre"
    )


def quantiles(axis: np.ndarray, profile: np.ndarray) -> list[float]:
    cumulative = np.cumsum(profile) / profile.sum()
    return [float(axis[np.searchsorted(cumulative, q)]) for q in QUANTILES]


def run(settings: Settings) -> dict:
    summary = json.loads((settings.dump_dir / "summary.json").read_text())
    maps = np.load(settings.dump_dir / "saliency.npz")
    canvas = summary["canvas"]
    axis = (np.arange(canvas) - canvas / 2 + 0.5) * summary["spacing_mm"]
    slices = maps["slices"]
    result = {"studies": summary["studies"], "groups": {}}
    for group in summary["groups"]:
        values = maps[group]
        total = values.sum()
        entry = {}
        for k, name in enumerate(TYPES):
            centre = settings.sagittal_centre_mm if name.startswith("Sagittal") else 0.0
            inside = {}
            for width, height in WINDOWS:
                columns = np.abs(axis - centre) <= width / 2
                rows = np.abs(axis - settings.vertical_centre_mm) <= height / 2
                inside[f"{width}x{height}"] = float(
                    values[k][np.ix_(rows, columns)].sum() / values[k].sum()
                )
            entry[name] = {
                "share_of_group": float(values[k].sum() / total),
                "per_slice_against_uniform": float(
                    values[k].sum() / total / (slices[k] / slices.sum())
                ),
                "horizontal_mm_quantiles": quantiles(axis, values[k].sum(0)),
                "vertical_mm_quantiles": quantiles(axis, values[k].sum(1)),
                "share_inside_window": inside,
            }
        result["groups"][group] = entry
    (settings.dump_dir / "analysis.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> None:
    result = run(Settings())
    for group, entry in result["groups"].items():
        print(group)
        for name, values in entry.items():
            inside = " ".join(
                f"{key} {value:.2f}"
                for key, value in values["share_inside_window"].items()
            )
            print(
                f"  {name}: horizontal {values['horizontal_mm_quantiles']} "
                f"vertical {values['vertical_mm_quantiles']} | {inside}"
            )


if __name__ == "__main__":
    main()
