from pathlib import Path

import yaml

root = Path("autoresearch/campaigns/routing-and-shifts")
ideas = {}
for name in ("seed-ideas.yaml", "seed-ideas-2.yaml"):
    for idea in yaml.safe_load((root / name).read_text())["ideas"]:
        ideas[idea["id"]] = idea
RENAME = {
    "learned-shift-per-series-type": "shift-per-series-type-light-port",
    "learned-shift-per-study-one-point": "shift-per-study-one-point-light-port",
    "learned-shift-per-study-per-view": "shift-per-study-per-view-light-port",
    "learned-shift-per-study-from-the-spots": "shift-per-study-from-the-spots-light-port",
    "crops-and-frame-share-one-centre": "crops-and-frame-share-one-centre-light-port",
}
EVIDENCE = (
    " Keeper's requeue of 2026-10-07 on a lighter port. The first port (experiments "
    "learned-shift-per-series-type-20261006T220210Z-47b5085a and -20261006T221425Z-19a2bc48) worked "
    "(view_shift_grad_norm 0.024, epoch 1 at 0.8927 against 0.8921) and was invalid on cost: training "
    "projected at 2,021 s against the limit of 1,595 s, and the machine at 34 of 39 GB with swap full "
    "against 17 GB for the incumbent's pipeline (shared memory 20 GB against 4 GB, main process 15.7 GB "
    "against 4.5 GB). Its loader workers prepared float32 canvases of 304 px per frame and the "
    "normalisation sorted pixels after the cut."
)
LIGHT_MECHANISM = (
    "The light port, five steps; it replaces the first port's steps and nothing of that port's float "
    "canvases or post-cut normalisation is kept. (1) Canvas: each view window is prepared with a margin "
    "of M pixels on every side at the view's own pixel spacing, M = round(data.view_shift_limit_mm / "
    "pixel spacing) with data.view_shift_limit_mm = 15.0 (40 px at 84 mm, 36 px at 93.33 mm); 120 mm "
    "views keep M = 0 and a zero shift. The window's own position is computed exactly as today "
    "(view_position, with its clipping); the margin is not clipped, and canvas pixels outside the "
    "acquired image are zero and outside the valid rectangle. (2) Normalisation stays where the "
    "incumbent has it and uses the incumbent's numbers: the two quantile bounds of each series are "
    "computed over the valid pixels of its unshifted 224 px windows, exactly the incumbent's "
    "quantile_bounds call on the central region of the canvases, and the whole canvas is scaled with "
    "those bounds and rounded to uint8 as normalize_pixels does. The loader therefore still ships "
    "uint8 images, of 304 or 296 px in place of 224 px, and nothing float; the central 224 px of every "
    "canvas equals the incumbent's prepared image exactly. (3) Cut: in the model, before any "
    "augmentation, each image's 224 px crop is taken from its uint8 canvas converted to float with "
    "affine_grid and grid_sample (bilinear, align_corners False, zero padding) at the canvas centre "
    "plus the shift; with a zero shift the sampling points are pixel centres and the crop is the "
    "central region unchanged. The valid rectangle and the patient maps are translated by the same "
    "shift. There is no normalisation and no sorting after the cut. (4) The affine augmentation, the "
    "reflection and everything after run unchanged on the cut crop. "
)
SERIES_STEP = (
    "(5) One parameter view_shift[type, view, 2], zero at the start, gives the shift in millimetres "
    "along the image's rows and columns as limit * tanh(view_shift / limit) with limit = M * pixel "
    "spacing; it is trained by the target loss through the sampling grid at the head learning rate. "
    "Nothing else changes."
)
LIGHT_IMPLEMENTATION = (
    "Start from the first port's code (changes.diff and source/ of "
    "outputs/autoresearch/routing-and-shifts/experiments/learned-shift-per-series-type-20261006T221425Z-19a2bc48) "
    "and keep its geometry: the margin-aware window and valid rectangle, the cutter's grid, the "
    "translated patient maps, the deferred reflection, the shift parameter and the logging. Remove its "
    "float canvases, its byte transport of raw buffers for the model, and its quantile sorting after "
    "the cut: the loader normalises as the incumbent does (step 2) and ships uint8 canvases. Required "
    "checks, reported with their numbers, all on the CPU on one training batch of eight studies: (a) "
    "the central 224 px of every canvas equals the incumbent's prepared image, largest absolute "
    "difference 0; (b) with every shift at zero the crops entering the augmentation equal the "
    "incumbent's images, in training and in evaluation mode, largest absolute difference 0; (c) the "
    "total bytes of the tensors in one collated batch, for the incumbent and for the candidate: the "
    "ratio must be at most 2.0; a candidate above it is returned blocked. Then the guardrail check. "
)
out = []
series = dict(ideas["learned-shift-per-series-type"])
series["id"] = RENAME["learned-shift-per-series-type"]
series["opportunity"] = series["opportunity"].strip() + EVIDENCE
series["mechanism"] = LIGHT_MECHANISM + SERIES_STEP
series["implementation"] = (
    LIGHT_IMPLEMENTATION
    + "Files as in the first port: data/preprocessing, data/dataset.py, modeling, config.py and "
    "configs/training.yaml (the one key data.view_shift_limit_mm), inference/engine.py, "
    "training/lightning.py (view_shift_<type>_<view>_<axis>_mm per epoch). view_shift is created under "
    "torch.random.fork_rng."
)
series["budget"] = (
    "Same slices decoded. The loader resamples and ships uint8 canvases with 1.75 to 1.84 times the "
    "pixels of the incumbent's images, and one grid_sample per image is added on the GPU. Shared "
    "memory is expected near twice the incumbent's 4 GB. The run reports its training and prediction "
    "time."
)
series["priority_rationale"] = "Keeper's requeue: the first port was invalid on time and memory; every per-study shift seed builds on this one."
out.append(series)
for old in list(RENAME)[1:]:
    idea = dict(ideas[old])
    idea["id"] = RENAME[old]
    for field in ("title", "opportunity", "mechanism", "exploitation", "budget", "implementation", "novelty", "coverage"):
        text = idea[field]
        for before, after in RENAME.items():
            text = text.replace(before, after)
        idea[field] = text
    idea["mechanism"] = (
        "Port: steps (1) to (4) of shift-per-series-type-light-port exactly (uint8 canvases normalised "
        "in the loader with the incumbent's quantile bounds, the cut in the model, no normalisation "
        "after the cut); never the first port's float canvases. The locator's input is the uint8 "
        "canvas divided by 255, with no further scaling. "
        + idea["mechanism"]
        .replace(
            "scaled for the locator only between the same two\n    quantile levels of the canvas's own valid pixels (detached), ",
            "",
        )
        .replace("scaled for the locator only between the same two quantile levels of the canvas's own valid pixels (detached), ", "")
    )
    idea["implementation"] = (
        "Reuse the port from the latest shift-per-series-type-light-port candidate under "
        "outputs/autoresearch/routing-and-shifts/experiments/ (changes.diff and source/), and the locator "
        "from an earlier per-study candidate when one exists; do not reuse any candidate named "
        "learned-shift-*. The same check (c) applies: one collated batch at most 2.0 times the "
        "incumbent's bytes, or the candidate is returned blocked. " + idea["implementation"]
    )
    idea["priority_rationale"] = "Keeper's requeue on the light port. " + idea["priority_rationale"]
    out.append(idea)
(root / "light-port-shift-ideas.yaml").write_text(
    yaml.safe_dump({"ideas": out, "exhausted_directions": ""}, sort_keys=False, allow_unicode=True, width=96)
)
print([idea["id"] for idea in out])
for idea in out:
    print(idea["id"], {k: len(v) for k, v in idea.items() if isinstance(v, str) and len(v) > 1500})
