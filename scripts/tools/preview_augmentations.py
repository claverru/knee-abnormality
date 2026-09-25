import base64
import hashlib
import json
from dataclasses import replace
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from typing import Literal

import polars as pl
import torch
from PIL import Image, ImageDraw, ImageFont
from pydantic import Field
from pydantic_settings import CliPositionalArg

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import SLICES_PER_INPUT, load_config
from knee_abnormality.data.dataset import collate_studies
from knee_abnormality.data.labels import STUDY
from knee_abnormality.data.preprocessing.device import prepare_images
from knee_abnormality.identity import preprocessing_identity
from knee_abnormality.inference.engine import prediction_data
from knee_abnormality.modeling.augmentation import (
    AugmentationParameters,
    ImageAugmentation,
    affine_matrices,
    apply_reflection,
    augmented_to_prepared,
    reflection_channels,
    reflection_inverse,
    reflection_positions,
    sample_augmentation_parameters,
    sample_reflection_parameters,
)
from knee_abnormality.modeling.spatial_tokens import feature_geometry
from knee_abnormality.provenance import copy_sources, sha256_file

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
<label><input type="checkbox" id="reflection" checked>Reflection only</label>
<label><input type="checkbox" id="force" checked>Force enabled transforms</label>
</div>
<p id="explanation"></p><div id="cards"></div>
<script>
const data = __DATA__;
const study = document.getElementById('study');
const variant = document.getElementById('variant');
const force = document.getElementById('force');
const reflection = document.getElementById('reflection');
const display = document.getElementById('display');
const positions = new Map();
const acquisitions = new Map();
data.studies.forEach((uid,i) => study.add(new Option(`Example ${i+1} · …${uid.slice(-8)}`,uid)));
study.value = data.studies[0];
data.seeds.forEach((seed,i) => variant.add(new Option(`Seed ${seed}`,i)));
document.getElementById('intro').textContent = `${data.image_size} px · ${(data.image_size*data.spacing_mm).toFixed(1)} mm crop · ${data.sampling} slice sampling · three consecutive slices per input · Kornia · affine ${100*data.settings.affine.probability}% · gamma/contrast ${100*data.settings.intensity.probability}% · noise ${100*data.settings.noise.probability}%`;
function render() {
  const draw = Number(variant.value);
  const cards = document.getElementById('cards');
  cards.replaceChildren();
  document.getElementById('explanation').textContent = force.checked
    ? 'Enabled transforms are forced on for comparison. Uncheck the box to respect their independently sampled training probabilities. Images use a fixed display scale before backbone normalization.'
    : 'Training probabilities are active. Geometry, gamma, contrast and noise strength are shared within each acquisition; pixel noise varies between slices. Images use a fixed display scale before backbone normalization.';
  if (reflection.checked) document.getElementById('explanation').textContent = 'Pure forced study reflection, respecting eligibility. Axial/coronal columns reverse; sagittal channels reverse. Targets retain their identities. Source laterality is unchanged metadata. Unavailable categories: ' + data.coverage.unavailable_categories.join(', ');
  document.getElementById('explanation').textContent += ' The composite maps consecutive MRI slices to red, green and blue; select a channel to inspect its grayscale slice.';
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
    content.innerHTML = '<div class="status"></div><div class="pair"><figure><img class="original" alt="Prepared original MRI input"><figcaption>Prepared original</figcaption></figure><figure><img class="augmented" alt="Augmentation preview of the same MRI input"><figcaption class="caption"></figcaption></figure></div><label>Input<input type="range" min="0" step="1"></label><div class="position"></div><details><summary>Example study and preparation</summary><div class="identity"></div><pre class="paths" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre></details>';
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
    content.querySelector('.status').textContent = reflection.checked ? `Study reflection ${entry.study_reflection_eligible ? 'eligible' : 'rejected: '+entry.study_reflection_reasons.join(', ')} · Source laterality ${entry.source_lateralities.join(', ') || 'unavailable'} · Obliquity ${entry.obliquity_degrees.toFixed(1)}°` : effects.join(' · ') + ` · Reflection ${force.checked ? entry.study_reflection_eligible : sample.reflection_applied}`;
    content.querySelector('.caption').textContent = reflection.checked ? 'Forced eligible reflection' : (force.checked ? 'Forced enabled transforms' : 'Sampled training transforms');
    content.querySelector('.identity').textContent = `Study: ${entry.study_uid}. Acquisition: ${entry.series_uid}. Gaussian σ ${entry.sampling_metadata.sampling_sigma_mm != null ? entry.sampling_metadata.sampling_sigma_mm.toFixed(1)+' mm' : 'n/a'} (fraction ${data.sampling_sigma_fraction} of pool reach); epoch ${data.epoch}. ${entry.sampling_path} sampling. This acquisition supplies ${entry.inputs} triplets, ${entry.slices} channel appearances, ${entry.unique_slices} unique source slices; the type budget is ${type.max_images/data.slices_per_input} inputs across acquisitions. ${entry.failures.length} preparation conditions.`;
    function update() {
      const index = Number(slider.value);
      positions.set(key,index);
      const channel = display.value === 'composite' ? null : Number(display.value);
      const original = channel === null ? entry.original[index] : entry.original_channels[channel][index];
      const augmented = reflection.checked ? (channel === null ? entry.reflected[index] : entry.reflected_channels[channel][index]) : channel === null
        ? (force.checked ? sample.images[index] : sample.sampled_images[index])
        : (force.checked ? sample.channels[channel][index] : sample.sampled_channels[channel][index]);
      content.querySelector('.original').src = original;
      content.querySelector('.augmented').src = augmented;
      const geometry = reflection.checked ? entry.reflected_anchors : (force.checked ? sample.anchors : sample.sampled_anchors);
      const association = geometry.source_channels[index];
      content.querySelector('.paths').textContent = association.map((source, channel) => `Output channel ${channel+1} ← source channel ${source+1}: ${entry.selected_paths[index][source]}`).join(String.fromCharCode(10)) + '\\nCentered LPS anchors in mm (3×3 final image grid, unavailable = null):\\nBefore: ' + JSON.stringify(entry.original_anchors.centered_xyz_mm[index]) + '\\nAfter: ' + JSON.stringify(geometry.centered_xyz_mm[index]) + '\\nVisual validity: ' + JSON.stringify(geometry.visual_valid[index]) + '\\nGeometry validity: ' + JSON.stringify(geometry.geometry_valid[index]);
      content.querySelector('.position').textContent = `Triplet ${index+1} / ${entry.original.length} · middle slice: ${entry.positions[index].toFixed(2)} → ${geometry.positions[index].toFixed(2)} mm from acquisition center`;
    }
    slider.addEventListener('input',update); update(); card.append(content); cards.append(card);
  });
}
[study,variant,force,display,reflection].forEach(control => control.addEventListener('change',render));
render();
</script></body></html>
"""


class PreviewCLI(CLISettings):
    model_config = describe(
        "Render study-coherent reflection and other augmentations with source geometry "
        "for browsing, with optional CUDA profiling of the reflection"
    )

    config: CliPositionalArg[Path] = Field(
        default=Path("configs/training.yaml"), description="Experiment YAML file"
    )
    output_dir: Path = Field(
        default=Path("outputs/previews/augmentations"), description="Preview directory"
    )
    study_uid: str | None = Field(default=None, description="Preview only this study")
    seed: int | None = Field(
        default=None,
        ge=0,
        lt=2**32,
        description="Augmentation seed; defaults to the config's",
    )
    variants: int = Field(
        default=4, ge=1, le=12, description="Augmented variants per study"
    )
    sampling: Literal["validation", "training"] = Field(
        default="validation", description="Triplet sampling mode"
    )
    epoch: int = Field(default=0, ge=0, description="Sampling epoch")
    split: Literal["test", "validation"] = Field(
        default="test", description="Studies to preview"
    )
    max_studies: int = Field(default=8, ge=1, le=128, description="Studies to preview")
    offset: int = Field(default=0, ge=0, description="Skip this many studies")
    profile_cuda: bool = Field(
        default=False, description="Also time the reflection on CUDA"
    )
    profile_iterations: int = Field(
        default=30, ge=1, le=100, description="CUDA profiling iterations"
    )
    geometry_headers: Path = Field(
        default=Path("outputs/analysis/geometry-metadata-20260912/headers.parquet"),
        description="Header table used for source geometry when present",
    )


def encode_image(pixels: torch.Tensor) -> str:
    stream = BytesIO()
    if pixels.ndim == 3:
        pixels = pixels.permute(1, 2, 0)
    Image.fromarray(pixels.numpy()).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def encode_channels(images: torch.Tensor) -> list[list[str]]:
    return [
        [encode_image(image[channel]) for image in images]
        for channel in range(SLICES_PER_INPUT)
    ]


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
        "Prepared original / forced eligible reflection",
        fill="white",
        font=title_font,
    )
    draw.text(
        (16, 38),
        f"{data['split']} · {data['sampling']} sampling · seed {data['seeds'][0]}",
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
        for column, encoded in enumerate(
            (entry["original"][index], entry["reflected"][index])
        ):
            pixels = Image.open(BytesIO(base64.b64decode(encoded.split(",", 1)[1])))
            canvas.paste(pixels, (16 + column * (size + 16), top + 28))
        detail = f"Triplet {index + 1}/{len(entry['original'])} · eligible {entry['study_reflection_eligible']}"
        draw.text((16, top + size + 38), detail, fill="#8dc8ff", font=font)
        draw.text(
            (16, top + size + 58),
            f"Position {entry['positions'][index]:.2f} → {entry['reflected_anchors']['positions'][index]:.2f} mm",
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


def anchor_record(series, size, center, eligible, sagittal, inverse=None):
    count = series.triplet_count
    reflected = torch.full((count,), eligible, dtype=torch.bool)
    planes = torch.full((count,), sagittal, dtype=torch.bool)
    association = reflection_channels(reflected, planes)
    if inverse is None:
        inverse = torch.eye(3)[None].expand(count, -1, -1)
    inverse = reflection_inverse(inverse, reflected, planes, size)
    _, visual, _ = feature_geometry(
        series.valid_rects,
        inverse,
        (3, 3),
        (size, size),
        (0, 0, (size - 1) / 2, (size - 1) / 2),
        association,
    )
    steps = torch.arange(3, dtype=torch.float32) * (size - 1) / 2
    y, x = torch.meshgrid(steps, steps, indexing="ij")
    anchors = torch.stack((x.flatten(), y.flatten(), torch.ones(9)), -1)
    prepared = torch.einsum("nij,kj->nki", inverse.float(), anchors)
    middle = series.patient_maps.shape[1] // 2
    points = torch.einsum(
        "nij,nkj->nki", series.patient_maps[:, middle].float(), prepared
    ) * torch.where(
        reflected[:, None, None], torch.tensor([-1.0, 1.0, 1.0]), torch.ones(3)
    )
    geometry = series.geometry_valid[:, None] & torch.isfinite(points).all(-1)
    points = torch.where(geometry[..., None], points, 0.0)
    centered = geometry & (center is not None and center.point is not None)
    reference = (
        torch.tensor(center.point, dtype=torch.float32)
        * torch.tensor([-1.0 if eligible else 1.0, 1.0, 1.0])
        if centered.any()
        else torch.zeros(3)
    )
    return {
        "source_channels": association.tolist(),
        "positions": reflection_positions(series.positions, reflected, planes).tolist(),
        "centered_xyz_mm": [
            [
                point if valid else None
                for point, valid in zip(values, flags, strict=True)
            ]
            for values, flags in zip(
                (points - reference).tolist(), centered.tolist(), strict=True
            )
        ],
        "visual_valid": visual.tolist(),
        "geometry_valid": geometry.tolist(),
    }


def profile_reflection(config, stacks, entries, iterations, seed):
    if not torch.cuda.is_available():
        raise RuntimeError("--profile-cuda requires CUDA")
    device = torch.device("cuda")
    pixels = torch.cat(stacks).to(device=device, dtype=torch.float32).div_(255)
    series_index = torch.cat(
        [torch.full((len(stack),), index) for index, stack in enumerate(stacks)]
    ).to(device)
    eligible = torch.tensor(
        [entry["study_reflection_eligible"] for entry in entries], device=device
    )[series_index]
    sagittal = torch.tensor(
        [
            config.data.series_types[entry["slot"]].plane == "Sagittal"
            for entry in entries
        ],
        device=device,
    )[series_index]
    generator = torch.Generator(device=device).manual_seed(seed)
    parameters = sample_augmentation_parameters(
        config.data, len(entries), device, generator
    )
    matrices = affine_matrices(parameters.affine, *pixels.shape[-2:])
    inverse = augmented_to_prepared(matrices, *pixels.shape[-2:])[series_index]
    augmenter = ImageAugmentation().to(device)
    results = {}
    for name, enabled in (("probability_zero", False), ("forced_eligible", True)):
        reflected = eligible if enabled else torch.zeros_like(eligible)

        def execute(enabled=enabled, reflected=reflected):
            values = pixels.clone()
            if enabled:
                values = apply_reflection(values, reflected, sagittal)
            values = augmenter(values, series_index, parameters, generator, matrices)
            return (
                values,
                reflection_inverse(inverse, reflected, sagittal, pixels.shape[-1]),
                reflection_channels(reflected, sagittal),
            )

        for _ in range(5):
            execute()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        generator.manual_seed(seed + 1)
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(iterations):
            execute()
        end.record()
        torch.cuda.synchronize()
        results[name] = {
            "milliseconds_per_iteration": start.elapsed_time(end) / iterations,
            "incremental_peak_allocated_bytes": torch.cuda.max_memory_allocated()
            - allocated,
            "incremental_peak_reserved_bytes": torch.cuda.max_memory_reserved()
            - reserved,
        }
    return {
        "hardware": torch.cuda.get_device_name(),
        "shape": list(pixels.shape),
        "eligible_inputs": int(eligible.sum()),
        "iterations": iterations,
        "warmup": 5,
        "rng_device": str(device),
        "parameter_seed": seed,
        "noise_seed": seed + 1,
        "scope": "augmentation only; fixed shared affine parameters; includes reversal temporaries; no backbone or optimizer",
        "measurements": results,
    }


def create_preview(args: PreviewCLI) -> dict:
    config = load_config(args.config)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new --output-dir")
    torch.set_num_threads(1)
    dataset, _, _ = prediction_data(config, args.split, 0)
    seed = config.seed if args.seed is None else args.seed
    dataset.training_seed = seed if args.sampling == "training" else None
    dataset.set_epoch(args.epoch)
    indices = [
        index
        for index, uid in enumerate(dataset.studies[STUDY])
        if args.study_uid is None or uid == args.study_uid
    ]
    if not indices:
        raise ValueError("Requested study is absent from the selected split")
    indices = indices[args.offset : args.offset + args.max_studies]
    if not indices:
        raise ValueError("No studies in requested range")
    headers = (
        pl.read_parquet(args.geometry_headers)
        if args.geometry_headers.exists()
        else None
    )
    entries = []
    stacks = []
    prepared = []
    centers = []
    items = []
    for index in indices:
        item = dataset[index]
        items.append(item)
        batch = prepare_images(collate_studies([item]))
        study_eligible = bool(batch["reflection_eligible"][0])
        entry = 0
        for slot, acquisitions in enumerate(item["series"]):
            for series in acquisitions:
                images = batch["images"][entry, : series.triplet_count]
                entry += 1
                stacks.append(images)
                prepared.append(series)
                centers.append(item["study_reference"])
                reflected = torch.full((len(images),), study_eligible, dtype=torch.bool)
                sagittal = torch.full_like(
                    reflected, config.data.series_types[slot].plane == "Sagittal"
                )
                reflection = apply_reflection(images, reflected, sagittal)
                source_lateralities = []
                if headers is not None:
                    source_lateralities = (
                        headers.filter(pl.col("SeriesInstanceUID") == series.uid)[
                            "Laterality"
                        ]
                        .drop_nulls()
                        .unique()
                        .to_list()
                    )
                source_lateralities = [
                    value for value in source_lateralities if value in {"L", "R"}
                ]
                maps = series.patient_maps[:, 1, :, :2]
                directions = maps / maps.norm(dim=1, keepdim=True).clamp_min(1e-12)
                obliquity = float(
                    torch.rad2deg(
                        torch.acos(directions.abs().amax(dim=1).clamp(0, 1))
                    ).amax()
                )
                entries.append(
                    {
                        "study_uid": item["study_uid"],
                        "reflection_eligible": series.reflection_eligible,
                        "reflection_reason": series.reflection_reason,
                        "study_reflection_eligible": study_eligible,
                        "study_reflection_reasons": batch["reflection_reasons"][0],
                        "source_lateralities": source_lateralities,
                        "obliquity_degrees": obliquity,
                        "padded_crop": bool(
                            (series.valid_rects[..., 2:] < config.data.image_size).any()
                        ),
                        "center_reason": item["study_reference"].reason
                        if item["study_reference"]
                        else "missing",
                        "original_anchors": anchor_record(
                            series,
                            config.data.image_size,
                            item["study_reference"],
                            False,
                            bool(sagittal[0]),
                        ),
                        "reflected_anchors": anchor_record(
                            series,
                            config.data.image_size,
                            item["study_reference"],
                            study_eligible,
                            bool(sagittal[0]),
                        ),
                        "reflected": [encode_image(image) for image in reflection],
                        "reflected_channels": encode_channels(reflection),
                        "series_uid": series.uid,
                        "slot": slot,
                        "inputs": len(images),
                        "slices": series.slice_count,
                        "unique_slices": series.unique_slice_count,
                        "sampling_path": series.sampling_path,
                        "selected_paths": series.selected_paths,
                        "sampling_metadata": item["sampling"][slot],
                        "positions": series.positions.tolist(),
                        "prepared_sha256": hashlib.sha256(
                            images.numpy().tobytes()
                        ).hexdigest(),
                        "failures": list(series.failures),
                        "original": [encode_image(pixels) for pixels in images],
                        "original_channels": encode_channels(images),
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
    study_ids = [item["study_uid"] for item in items]
    eligibility = collate_studies(items)["reflection_eligible"]
    for variant_seed in seeds:
        generator = torch.Generator().manual_seed(variant_seed)
        reflection_draw = sample_reflection_parameters(
            config.data.reflection_augmentation.probability,
            eligibility,
            generator,
        )
        sampled = sample_augmentation_parameters(
            config.data,
            len(entries),
            torch.device("cpu"),
            generator,
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
            pixels = stack.to(torch.float32).div_(255.0)
            series_index = torch.full((len(stack),), index, dtype=torch.int64)
            noise_seed = variant_seed * len(entries) + index
            record = {
                "reflection_applied": bool(
                    reflection_draw.applied[study_ids.index(entry["study_uid"])]
                ),
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
                reflected = torch.full(
                    (len(stack),),
                    entry["study_reflection_eligible"]
                    if not prefix
                    else record["reflection_applied"],
                    dtype=torch.bool,
                )
                sagittal = torch.full_like(
                    reflected,
                    config.data.series_types[entry["slot"]].plane == "Sagittal",
                )
                record[f"{prefix}anchors"] = anchor_record(
                    prepared[index],
                    config.data.image_size,
                    centers[index],
                    bool(reflected[0]),
                    bool(sagittal[0]),
                    augmented_to_prepared(
                        affine_matrices(parameters.affine, *pixels.shape[-2:]),
                        *pixels.shape[-2:],
                    )[series_index],
                )
                augmented = augmenter(
                    apply_reflection(pixels.clone(), reflected, sagittal),
                    series_index,
                    parameters,
                    torch.Generator().manual_seed(noise_seed),
                )
                encoded = augmented.mul(255).round().to(torch.uint8)
                record[f"{prefix}images"] = [encode_image(image) for image in encoded]
                record[f"{prefix}channels"] = encode_channels(encoded)
            entry["variants"].append(record)
    coverage = {
        "source_lateralities": sorted(
            {value for entry in entries for value in entry["source_lateralities"]}
        ),
        "planes": sorted(
            {config.data.series_types[entry["slot"]].plane for entry in entries}
        ),
        "oblique_over_10_degrees": sum(
            entry["obliquity_degrees"] > 10 for entry in entries
        ),
        "padded_crops": sum(entry["padded_crop"] for entry in entries),
        "rejected_centers": sum(
            item["study_reference"] is None or item["study_reference"].point is None
            for item in items
        ),
        "missing_acquisitions": sum(
            any(not values for values in item["series"]) for item in items
        ),
        "ineligible_studies": int((~eligibility).sum()),
        "eligible_studies": int(eligibility.sum()),
        "unavailable_categories": [],
    }
    for category in (
        "oblique_over_10_degrees",
        "padded_crops",
        "rejected_centers",
        "missing_acquisitions",
        "ineligible_studies",
    ):
        if not coverage[category]:
            coverage["unavailable_categories"].append(category)
    if not {"L", "R"} <= set(coverage["source_lateralities"]):
        coverage["unavailable_categories"].append("both_source_lateralities")
    profile = (
        profile_reflection(config, stacks, entries, args.profile_iterations, seed)
        if args.profile_cuda
        else None
    )
    data = {
        "coverage": coverage,
        "split": args.split,
        "reflection_probability": config.data.reflection_augmentation.probability,
        "image_size": config.data.image_size,
        "slices_per_input": SLICES_PER_INPUT,
        "spacing_mm": config.data.spacing_mm,
        "sampling_sigma_fraction": config.data.sampling_sigma_fraction,
        "sampling": args.sampling,
        "epoch": args.epoch,
        "settings": {
            name: setting.model_dump(mode="json") for name, setting in settings.items()
        },
        "types": [kind.model_dump(mode="json") for kind in config.data.series_types],
        "studies": dataset.studies[STUDY].gather(indices).to_list(),
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
    source_sha256 = copy_sources(
        [
            Path(__file__),
            *(
                Path("src/knee_abnormality") / name
                for name in (
                    "config.py",
                    "identity.py",
                    "data/dataset.py",
                    "data/sources.py",
                    "inference/engine.py",
                    "modeling/augmentation.py",
                    "modeling/model.py",
                    "modeling/spatial_tokens.py",
                )
            ),
            Path("src/knee_abnormality/data/preprocessing"),
        ],
        output / "source",
    )
    manifest = {
        "preprocessing_identity": preprocessing_identity(config.model.backbone),
        "coverage": coverage,
        "profile": profile,
        "split": args.split,
        "geometry_headers_sha256": sha256_file(args.geometry_headers)
        if args.geometry_headers.exists()
        else None,
        "html": str(output / "preview.html"),
        "overview": str(output / "overview.png"),
        "config": config.model_dump(mode="json"),
        "rng_device": "cpu",
        "sampling": args.sampling,
        "sampling_seed": seed,
        "sampling_epoch": args.epoch,
        "seeds": seeds,
        "display": "Fixed [0,1] before backbone normalization; pure eligible study reflection by default; combined forced or sampled transforms selectable; source laterality metadata only",
        "versions": {
            name: version(name)
            for name in ("torch", "kornia", "kornia-rs", "pillow", "pydicom", "polars")
        },
        "source_sha256": source_sha256,
        "input_sha256": {
            name: sha256_file(config.raw_dir / name)
            for name in (
                ("test.csv", "test_series.csv")
                if args.split == "test"
                else ("train.csv", "train_series.csv")
            )
        },
        "series": [
            {
                **{
                    key: value
                    for key, value in entry.items()
                    if key
                    not in {
                        "original",
                        "original_channels",
                        "reflected",
                        "reflected_channels",
                        "variants",
                    }
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
        "coverage": coverage,
        "profile": profile,
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
    result = create_preview(PreviewCLI())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
