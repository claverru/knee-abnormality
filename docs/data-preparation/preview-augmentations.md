# Preview augmentations

Generate an offline comparison from the example-test DICOMs:

```bash
rtk proxy uv run --locked python -m scripts.tools.preview_augmentations configs/resnet50.yaml --output-dir outputs/previews/triplet-augmentations-new
```

Open the generated `preview.html` in a browser. It embeds its
images and needs no server, network or GPU. The accompanying
`overview.png` shows the middle selected input for all six plane–contrast types,
with the original on the left and a forced eligible reflection on the right.

- **Example study:** defaults to one synchronized study; optionally view all types across examples. The three example-test studies supply thirteen usable
  selected series and collectively cover all six types; individual studies
  have missing types, shown explicitly.
- **Input:** move each card's slider to compare the same triplet before
  and after augmentation. Its displayed position identifies the middle slice.
  Preparation uses the configured shortest usable
  acquisition, crop, per-series clipping and deterministic evaluation sampling.
- **Display:** inspect the three-channel composite or any of its three
  grayscale channels. The composite maps consecutive slices to red, green and
  blue; its colors show differences between adjacent slices. Sagittal reflection
  reverses source channel associations.
- **Augmentation draw:** choose seeded affine, gamma/contrast and Gaussian-noise
  parameters. Geometry and scalar strengths are shared across an acquisition;
  noise pixels differ by slice.
- **Reflection only:** checked by default for pure forced eligible reflection.
  Uncheck it to inspect combined transforms. **Force enabled transforms** then
  selects forced families; uncheck that to use sampled training gates.

The tool and model share `modeling/augmentation.py`. Images use the same fixed `[0, 1]`
display scale before checkpoint-specific normalization. Each displayed frame
is rounded to uint8 for PNG storage; training applies transforms in float32.
Test inference bypasses augmentation. The crop-selection era's validation crop and glimpse
overlays and its `--checkpoint` option were removed with the
[dropped investigation](../history/crop-selection-investigation.md). Preview seeds use the CPU
generator and
do not identify the exact CUDA random draws of a training run. The viewer does
not simulate whole-series dropout.

Regenerate with a fresh directory and optionally a different seed or number
of draws. `--study-uid` restricts the source to one UID from the selected split:

```bash
rtk proxy uv run --locked python -m scripts.tools.preview_augmentations configs/resnet50.yaml --seed 100 --variants 6 --output-dir outputs/previews/augmentations-seed100
```

`--split validation` previews fold-0 validation studies instead and needs the prepared fold
labels; `--offset` and `--max-studies` bound the range. Sampling is deterministic evaluation
sampling unless `--sampling training --epoch N` runs the seeded training sampler for that
epoch. Reflection decisions apply to every acquisition of a study, and all selected
acquisitions of a plane–contrast type stay separately selectable. Each variant draws reflection
and acquisition parameters from one advancing CPU generator; gates consume separate draws.
Cards also show centered LPS anchors from each prepared series' middle-slice patient maps
(the model does not use them), visual and geometry validity, and eligibility and source
laterality from the `--geometry-headers` audit when present. `--profile-cuda` adds bounded
augmentation-only CUDA timing and peak memory, without a backbone or optimizer; check GPU usage
first:

```bash
rtk proxy uv run --locked python -m scripts.tools.preview_augmentations configs/resnet50.yaml --variants 1 --profile-cuda --output-dir outputs/previews/reflection-profile-new
```

`manifest.json` records settings, CPU seeds, per-series parameters, sampled
application flags, study/type identities, selected positions, prepared-pixel
hashes, input/source hashes, dependency versions, unavailable coverage categories and the
preprocessing identity with the configured backbone's spatial mapping, without building the
backbone. The output also includes
the effective config and source snapshot. Existing directories are preserved.
Historical single-slice preview evidence remains under
`outputs/previews/augmentations/` and
`outputs/archive/maintenance/augmentation-preview-20260910T211117Z/`. The
adjacent-triplet preview generated on 2026-09-11 is
[available here](../../outputs/previews/triplet-augmentations/preview.html).
Its three example studies contain thirteen selected series, 69 complete
triplet inputs and 207 distinct selected slices. All six types are present,
with no reported preparation conditions. The manifest records input counts
and raw-slice counts separately. Its composite overview was visually inspected
and the viewer JavaScript passed a syntax check. Shared three-channel
augmentation also completed the bounded Lightning execution recorded in
[modeling](../modeling/triplet-inputs-and-sampling.md#adjacent-triplet-sampling).
