import json
from math import ceil
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image, ImageDraw
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from scripts.tools.prepare_combined_labels import fingerprint


class RenderCLI(CLISettings):
    model_config = describe(
        "Render a selected DICOM series as CPU contact sheets for the v3 label error audit"
    )
    series: Path = Field(description="Read-only source DICOM series directory")
    output: Path = Field(description="Fresh PNG path inside this worktree")


def render(settings: RenderCLI) -> None:
    if not settings.output.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError("Outputs must stay inside this worktree")
    if settings.output.exists():
        raise FileExistsError(settings.output)
    files = sorted(settings.series.glob("*.dcm"))
    slices = [(path, pydicom.dcmread(path)) for path in files]
    orientation = np.asarray(slices[0][1].ImageOrientationPatient, dtype=float)
    normal = np.cross(orientation[:3], orientation[3:])
    slices.sort(key=lambda item: float(np.dot(item[1].ImagePositionPatient, normal)))
    arrays = [
        ds.pixel_array.astype(np.float32) * float(ds.get("RescaleSlope", 1))
        + float(ds.get("RescaleIntercept", 0))
        for _, ds in slices
    ]
    low, high = np.percentile(np.stack(arrays), [1, 99.5])
    width, height, columns = 320, 344, 6
    sheet = Image.new(
        "RGB", (width * columns, height * ceil(len(slices) / columns)), "black"
    )
    draw = ImageDraw.Draw(sheet)
    records = []
    for i, ((path, ds), array) in enumerate(zip(slices, arrays)):
        pixels = np.uint8(np.clip((array - low) / max(high - low, 1e-6), 0, 1) * 255)
        if ds.get("PhotometricInterpretation") == "MONOCHROME1":
            pixels = 255 - pixels
        view = Image.fromarray(pixels).resize((width, width))
        x, y = i % columns * width, i // columns * height
        sheet.paste(view, (x, y + 24))
        position = [round(float(v), 2) for v in ds.ImagePositionPatient]
        draw.text((x + 3, y + 4), f"{i:02d} | LPS {position}", fill="white")
        records.append(fingerprint(path) | {"index": i, "position_lps": position})
    settings.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(settings.output)
    settings.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "series": str(settings.series),
                "description": str(slices[0][1].get("SeriesDescription", "")),
                "orientation_lps": orientation.tolist(),
                "slice_order": "Ascending projection onto cross product of DICOM orientation vectors",
                "display": "Native pixel orientation, shared intensity percentiles 1 and 99.5, all slices",
                "window": [float(low), float(high)],
                "slices": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(settings.output)


def main() -> None:
    render(RenderCLI())


if __name__ == "__main__":
    main()
