import base64
import hashlib
import json
import shutil
import sys
from dataclasses import replace
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from pydantic import Field
from pydantic_settings import CliPositionalArg, SettingsError

from knee_abnormality.augmentation import (
    AugmentationParameters,
    ImageAugmentation,
    sample_augmentation_parameters,
)
from knee_abnormality.cli import NonExitingCLISettings
from knee_abnormality.config import load_config
from knee_abnormality.inference import prediction_data
from knee_abnormality.provenance import sha256_file

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MRI augmentation viewer</title>
<style>
:root {color-scheme:dark;font-family:system-ui,sans-serif;background:#11151d;color:#e9eef6}
body {max-width:1240px;margin:auto;padding:28px}
h1 {margin:0 0 8px;font-size:28px} h2 {font-size:17px;margin:0 0 6px}
p {line-height:1.55;color:#b6c2d2} .controls {display:flex;gap:24px;flex-wrap:wrap;
align-items:end;background:#1a2230;padding:18px;border-radius:12px;margin:22px 0}
label {display:block;font-size:14px} select {display:block;margin-top:6px;
padding:8px;max-width:100%;background:#11151d;color:inherit;border:1px solid #607088;
border-radius:5px} input {accent-color:#78bdff} input[type=range] {width:100%}
#cards {display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:20px}
.card {background:#1a2230;border:1px solid #2b394d;border-radius:12px;padding:18px}
.pair {display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:15px 0}
figure {margin:0} img {width:100%;aspect-ratio:1;display:block;background:#000}
figcaption,.small {font-size:13px;color:#b6c2d2;margin-top:7px}
.status {color:#8dc8ff;font-size:14px;min-height:22px;margin-top:12px}
.position {font-size:13px;color:#b6c2d2;margin-bottom:8px}
details {font-size:12px;overflow-wrap:anywhere;margin-top:12px;color:#91a2b8}
.missing {color:#91a2b8;padding:45px 0;text-align:center}
@media(max-width:540px) {body {padding:14px} #cards {grid-template-columns:1fr}}
</style></head><body>
<h1>MRI augmentation viewer</h1>
<p id="intro"></p>
<div class="controls">
<label>Example study<select id="study"><option value="all">All six types across examples</option></select></label>
<label>Augmentation draw<select id="variant"></select></label>
<label>Display<select id="display"><option value="composite">Three-channel composite</option><option value="0">Channel 1 · first slice</option><option value="1">Channel 2 · middle slice</option><option value="2">Channel 3 · last slice</option></select></label>
<label><input type="checkbox" id="force" checked>Force enabled transforms</label>
</div>
<p id="explanation"></p><div id="cards"></div>
<script>
const data = __DATA__;
const study = document.getElementById('study');
const variant = document.getElementById('variant');
const force = document.getElementById('force');
const display = document.getElementById('display');
const triplets = data.slices_per_input === 3;
display.disabled = !triplets;
if (!triplets) display.options[0].textContent = 'Grayscale slice';
const positions = new Map();
const acquisitions = new Map();
data.studies.forEach((uid,i) => study.add(new Option(`Example ${i+1} · …${uid.slice(-8)}`,uid)));
data.seeds.forEach((seed,i) => variant.add(new Option(`Seed ${seed}`,i)));
document.getElementById('intro').textContent = `${data.image_size} px · ${(data.image_size*data.spacing_mm).toFixed(1)} mm crop · ${data.sampling} slice sampling · ${triplets ? 'three consecutive slices per input' : 'one slice per input'} · Kornia · affine ${100*data.settings.affine.probability}% · gamma/contrast ${100*data.settings.intensity.probability}% · noise ${100*data.settings.noise.probability}%`;
function render() {
  const draw = Number(variant.value);
  const cards = document.getElementById('cards');
  cards.replaceChildren();
  document.getElementById('explanation').textContent = force.checked
    ? 'Enabled transforms are forced on for comparison. Uncheck the box to respect their independently sampled training probabilities. Images use a fixed display scale before backbone normalization.'
    : 'Training probabilities are active. Geometry, gamma, contrast and noise strength are shared within each acquisition; pixel noise varies between slices. Images use a fixed display scale before backbone normalization.';
  if (triplets) document.getElementById('explanation').textContent += ' The composite maps consecutive MRI slices to red, green and blue; select a channel to inspect its grayscale slice.';
  data.types.forEach((type,slot) => {
    const candidates = data.series.filter(row => row.slot === slot && (study.value === 'all' || row.study_uid === study.value));
    const acquisitionKey = `${study.value}/${slot}`;
    const entry = candidates.find(row => row.series_uid === acquisitions.get(acquisitionKey)) ?? candidates[0];
    const card = document.createElement('section');
    card.className = 'card';
    const title = document.createElement('h2');
    title.textContent = type.plane;
    card.append(title);
    const subtitle = document.createElement('div');
    subtitle.className = 'small';
    subtitle.textContent = `Fluid-sensitive ${type.fluid_sensitive} · Fat suppression ${type.fat_suppression}`;
    card.append(subtitle);
    if (!entry) {
      const missing = document.createElement('div');
      missing.className = 'missing';
      missing.textContent = 'This type is absent from the selected example.';
      card.append(missing); cards.append(card); return;
    }
    const choiceLabel = document.createElement('label');
    choiceLabel.textContent = 'Acquisition';
    const choice = document.createElement('select');
    candidates.forEach(row => choice.add(new Option(`Example ${data.studies.indexOf(row.study_uid)+1} · …${row.series_uid.slice(-8)} · ${row.inputs} inputs`, row.series_uid)));
    choice.value = entry.series_uid;
    choice.addEventListener('change', () => { acquisitions.set(acquisitionKey, choice.value); render(); });
    choiceLabel.append(choice); card.append(choiceLabel);
    const content = document.createElement('div');
    content.innerHTML = '<div class="status"></div><div class="pair"><figure><img class="original" alt="Prepared original MRI input"><figcaption>Prepared original</figcaption></figure><figure><img class="augmented" alt="Augmentation preview of the same MRI input"><figcaption class="caption"></figcaption></figure></div><label>Input<input type="range" min="0" step="1"></label><div class="position"></div><details><summary>Example study and preparation</summary><div class="identity"></div></details>';
    const slider = content.querySelector('input');
    const key = `${entry.study_uid}/${entry.series_uid}`;
    slider.max = entry.original.length-1;
    slider.value = positions.get(key) ?? Math.floor(entry.original.length/2);
    const sample = entry.variants[draw];
    const active = name => sample[name].applied || (force.checked && data.settings[name].probability > 0);
    const effects = [];
    if (active('affine')) effects.push(`Rotation ${sample.affine.angle_degrees.toFixed(2)}° · Shift ${sample.affine.translation_fraction.map(x => (100*x).toFixed(2)+'%').join(', ')} · Scale ${sample.affine.scale.toFixed(3)}`);
    if (active('intensity')) effects.push(`Gamma ${sample.intensity.gamma.toFixed(4)} · Contrast ${sample.intensity.contrast.toFixed(4)}`);
    if (active('noise')) effects.push(`Noise σ ${sample.noise.std.toFixed(4)}`);
    content.querySelector('.status').textContent = effects.length ? effects.join(' · ') : 'All transforms skipped for this acquisition';
    content.querySelector('.caption').textContent = force.checked ? 'Forced enabled transforms' : 'Sampled training transforms';
    content.querySelector('.identity').textContent = `Study: ${entry.study_uid}. Acquisition: ${entry.series_uid}. ${data.acquisition_sampling} acquisition selection, ${data.slice_sampling} slice sampling; epoch ${data.epoch}. Center sampling weight ${data.center_sampling_weight}. This acquisition supplies ${entry.inputs} inputs from ${entry.slices} distinct slices; the type budget is ${type.max_images/data.slices_per_input} inputs across acquisitions. ${entry.failures.length} preparation conditions.`;
    function update() {
      const index = Number(slider.value);
      positions.set(key,index);
      const channel = display.value === 'composite' ? null : Number(display.value);
      const original = channel === null ? entry.original[index] : entry.original_channels[channel][index];
      const augmented = channel === null
        ? (force.checked ? sample.images[index] : sample.sampled_images[index])
        : (force.checked ? sample.channels[channel][index] : sample.sampled_channels[channel][index]);
      content.querySelector('.original').src = original;
      content.querySelector('.augmented').src = augmented;
      const unit = data.position_encoding === 'physical_mm' ? `mm from acquisition ${data.position_reference}` : 'ordered slice index';
      content.querySelector('.position').textContent = `${triplets ? 'Triplet' : 'Slice'} ${index+1} / ${entry.original.length} · ${triplets ? 'middle slice: ' : ''}${entry.positions[index].toFixed(2)} ${unit}`;
    }
    slider.addEventListener('input',update); update(); card.append(content); cards.append(card);
  });
}
[study,variant,force,display].forEach(control => control.addEventListener('change',render));
render();
</script></body></html>
"""


class PreviewCLI(NonExitingCLISettings):
    config: CliPositionalArg[Path] = Path("configs/baseline.yaml")
    output_dir: Path = Path("outputs/previews/augmentations")
    study_uid: str | None = None
    seed: int | None = Field(default=None, ge=0, lt=2**32)
    variants: int = Field(default=4, ge=1, le=12)
    sampling: Literal["validation", "training"] = "validation"
    epoch: int = Field(default=0, ge=0)


def encode_image(pixels: torch.Tensor) -> str:
    stream = BytesIO()
    if pixels.ndim == 3:
        pixels = pixels[0] if pixels.shape[0] == 1 else pixels.permute(1, 2, 0)
    Image.fromarray(pixels.numpy()).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def encode_channels(images: torch.Tensor) -> list[list[str]]:
    if images.ndim != 4 or images.shape[1] != 3:
        return []
    return [[encode_image(image[channel]) for image in images] for channel in range(3)]


def overview(data: dict, output: Path) -> None:
    size = data["image_size"]
    width = max(520, size * 2 + 48)
    row_height = size + 112
    canvas = Image.new("RGB", (width, 65 + row_height * len(data["types"])), "#11151d")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=17)
    font = ImageFont.load_default(size=13)
    draw.text(
        (16, 12),
        "Prepared original / forced augmentation",
        fill="white",
        font=title_font,
    )
    draw.text(
        (16, 38),
        f"Example test · {data['sampling']} sampling · seed {data['seeds'][0]}",
        fill="#b6c2d2",
        font=font,
    )
    for slot, kind in enumerate(data["types"]):
        top = 65 + slot * row_height
        label = f"{kind['plane']} · Fluid-sensitive {kind['fluid_sensitive']} · Fat suppression {kind['fat_suppression']}"
        draw.text((16, top), label, fill="white", font=font)
        entry = next((row for row in data["series"] if row["slot"] == slot), None)
        if entry is None:
            draw.text(
                (16, top + 40), "Absent from these examples", fill="#b6c2d2", font=font
            )
            continue
        index = len(entry["original"]) // 2
        sample = entry["variants"][0]
        for column, encoded in enumerate(
            (entry["original"][index], sample["images"][index])
        ):
            pixels = Image.open(BytesIO(base64.b64decode(encoded.split(",", 1)[1])))
            canvas.paste(pixels, (16 + column * (size + 16), top + 28))
        unit = "Triplet" if data["slices_per_input"] == 3 else "Slice"
        intensity_detail = (
            f"gamma {sample['intensity']['gamma']:.3f} · contrast {sample['intensity']['contrast']:.3f}"
            if data["settings"]["intensity"]["probability"] > 0
            else "intensity disabled"
        )
        detail = f"{unit} {index + 1}/{len(entry['original'])} · {intensity_detail}"
        draw.text((16, top + size + 38), detail, fill="#8dc8ff", font=font)
        affine_detail = (
            f"Angle {sample['affine']['angle_degrees']:.1f}° · scale {sample['affine']['scale']:.3f}"
            if data["settings"]["affine"]["probability"] > 0
            else "affine disabled"
        )
        noise_detail = (
            f"noise std {sample['noise']['std']:.4f}"
            if data["settings"]["noise"]["probability"] > 0
            else "noise disabled"
        )
        draw.text(
            (16, top + size + 58),
            f"{affine_detail} · {noise_detail}",
            fill="#8dc8ff",
            font=font,
        )
        draw.text(
            (16, top + size + 78),
            f"Study …{entry['study_uid'][-12:]}",
            fill="#b6c2d2",
            font=font,
        )
    canvas.save(output)


def create_preview(args: PreviewCLI) -> dict:
    config = load_config(args.config)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new --output-dir")
    torch.set_num_threads(1)
    dataset, _, _ = prediction_data(config, "test", 0)
    seed = config.seed if args.seed is None else args.seed
    dataset.training = args.sampling == "training"
    dataset.seed = seed
    dataset.set_epoch(args.epoch)
    indices = [
        index
        for index, uid in enumerate(dataset.studies["StudyInstanceUID"])
        if args.study_uid is None or uid == args.study_uid
    ]
    if not indices:
        raise ValueError("Requested study is absent from the example-test table")
    entries = []
    stacks = []
    for index in indices:
        item = dataset[index]
        for slot, acquisitions in enumerate(item["series"]):
            for series in acquisitions:
                stacks.append(series.images)
                entries.append(
                    {
                        "study_uid": item["study_uid"],
                        "series_uid": series.uid,
                        "slot": slot,
                        "inputs": len(series.images),
                        "slices": series.slice_count,
                        "positions": series.positions.tolist(),
                        "prepared_sha256": hashlib.sha256(
                            series.images.numpy().tobytes()
                        ).hexdigest(),
                        "failures": list(series.failures),
                        "original": [encode_image(pixels) for pixels in series.images],
                        "original_channels": encode_channels(series.images),
                        "variants": [],
                    }
                )
    settings = {
        "intensity": config.data.intensity_augmentation,
        "affine": config.data.affine_augmentation,
        "noise": config.data.noise_augmentation,
    }
    augmenter = ImageAugmentation()
    seeds = [seed + variant for variant in range(args.variants)]
    for variant_seed in seeds:
        sampled = sample_augmentation_parameters(
            config.data,
            len(entries),
            torch.device("cpu"),
            torch.Generator().manual_seed(variant_seed),
        )
        forced = AugmentationParameters(
            **{
                name: replace(
                    getattr(sampled, name),
                    applied=torch.full_like(
                        getattr(sampled, name).applied, setting.probability > 0
                    ),
                )
                for name, setting in settings.items()
            }
        )
        for index, (entry, stack) in enumerate(zip(entries, stacks, strict=True)):
            pixels = (
                (stack if stack.ndim == 4 else stack[:, None])
                .to(torch.float32)
                .div_(255.0)
            )
            series_index = torch.full((len(stack),), index, dtype=torch.int64)
            noise_seed = variant_seed * len(entries) + index
            record = {
                "intensity": {
                    "applied": bool(sampled.intensity.applied[index]),
                    "gamma": float(sampled.intensity.gamma[index]),
                    "contrast": float(sampled.intensity.contrast[index]),
                },
                "affine": {
                    "applied": bool(sampled.affine.applied[index]),
                    "angle_degrees": float(sampled.affine.angle_degrees[index]),
                    "translation_fraction": sampled.affine.translation_fraction[
                        index
                    ].tolist(),
                    "scale": float(sampled.affine.scale[index]),
                },
                "noise": {
                    "applied": bool(sampled.noise.applied[index]),
                    "std": float(sampled.noise.std[index]),
                    "seed": noise_seed,
                },
            }
            for prefix, parameters in (("", forced), ("sampled_", sampled)):
                augmented = augmenter(
                    pixels.clone(),
                    series_index,
                    parameters,
                    torch.Generator().manual_seed(noise_seed),
                )
                encoded = augmented.mul(255).round().to(torch.uint8)
                record[f"{prefix}images"] = [encode_image(image) for image in encoded]
                record[f"{prefix}channels"] = encode_channels(encoded)
            entry["variants"].append(record)
    data = {
        "image_size": config.data.image_size,
        "slices_per_input": config.data.slices_per_input,
        "position_encoding": config.data.position_encoding,
        "position_reference": config.data.position_reference,
        "spacing_mm": config.data.spacing_mm,
        "acquisition_sampling": config.data.acquisition_sampling,
        "slice_sampling": config.data.slice_sampling,
        "center_sampling_weight": config.data.center_sampling_weight,
        "sampling": args.sampling,
        "epoch": args.epoch,
        "settings": {
            name: setting.model_dump(mode="json") for name, setting in settings.items()
        },
        "types": [kind.model_dump(mode="json") for kind in config.data.series_types],
        "studies": dataset.studies["StudyInstanceUID"].gather(indices).to_list(),
        "seeds": seeds,
        "series": entries,
    }
    output.mkdir(parents=True)
    (output / "preview.html").write_text(
        PAGE.replace(
            "__DATA__",
            json.dumps(data, separators=(",", ":"), allow_nan=False).replace(
                "<", "\\u003c"
            ),
        )
    )
    overview(data, output / "overview.png")
    (output / "effective_config.json").write_text(
        config.model_dump_json(indent=2) + "\n"
    )
    sources = [
        Path(__file__).resolve(),
        *[
            Path("src/knee_abnormality") / name
            for name in (
                "augmentation.py",
                "config.py",
                "dataset.py",
                "inference.py",
            )
        ],
        Path("src/knee_abnormality/preprocessing"),
    ]
    project_root = Path.cwd().resolve()
    source_files = []
    for source in sources:
        source = source.resolve()
        source_files.extend(
            path for path in (source.rglob("*.py") if source.is_dir() else [source])
            if path.is_file()
        )
    (output / "source").mkdir()
    for source in source_files:
        relative = source.relative_to(project_root)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    manifest = {
        "html": str(output / "preview.html"),
        "overview": str(output / "overview.png"),
        "config": config.model_dump(mode="json"),
        "rng_device": "cpu",
        "sampling": args.sampling,
        "sampling_seed": seed,
        "sampling_epoch": args.epoch,
        "seeds": seeds,
        "display": "Fixed [0,1] before backbone normalization; triplet RGB composite or individual grayscale channels; enabled transforms forced by default, independent sampled training gates selectable",
        "versions": {
            name: version(name)
            for name in ("torch", "kornia", "kornia-rs", "pillow", "pydicom", "polars")
        },
        "source_sha256": {
            str(path.relative_to(project_root)): sha256_file(path)
            for path in source_files
        },
        "input_sha256": {
            name: sha256_file(config.raw_dir / name)
            for name in ("test.csv", "test_series.csv")
        },
        "series": [
            {
                **{
                    key: value
                    for key, value in entry.items()
                    if key not in {"original", "original_channels", "variants"}
                },
                "variants": [
                    {
                        key: value
                        for key, value in sample.items()
                        if key
                        not in {
                            "images",
                            "channels",
                            "sampled_images",
                            "sampled_channels",
                        }
                    }
                    for sample in entry["variants"]
                ],
            }
            for entry in entries
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "html": manifest["html"],
        "overview": manifest["overview"],
        "studies": len(indices),
        "series": len(entries),
        "inputs": sum(entry["inputs"] for entry in entries),
        "slices": sum(entry["slices"] for entry in entries),
        "types_present": len({entry["slot"] for entry in entries}),
        "seeds": seeds,
    }


def main() -> None:
    try:
        result = create_preview(PreviewCLI())
    except (
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        yaml.YAMLError,
        SettingsError,
    ) as error:
        print(f"Preview error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
