import torch


def sagittal_types(series_types) -> torch.Tensor:
    return torch.tensor([kind.plane == "Sagittal" for kind in series_types])


def reflect_batch(batch: dict, sagittal_lookup: torch.Tensor) -> dict:
    reflected = batch["reflection_eligible"][batch["series_study_index"]]
    sagittal = sagittal_lookup.to(reflected.device)[batch["series_type_index"]]
    mirror = reflected & ~sagittal
    reverse = reflected & sagittal
    images = batch["images"]
    rects = batch["valid_rects"]
    positions = batch["slice_positions"]
    if mirror.any() or reverse.any():
        images = images.clone()
        rects = rects.clone()
        positions = positions.clone()
    if mirror.any():
        images[mirror] = images[mirror].flip(-1)
        selected = rects[mirror]
        selected[..., 1] = images.shape[-1] - selected[..., 3] - selected[..., 1]
        rects[mirror] = selected
    if reverse.any():
        images[reverse] = images[reverse].flip(2)
        rects[reverse] = rects[reverse].flip(2)
        positions[reverse] = -positions[reverse]
    return {
        **batch,
        "images": images,
        "valid_rects": rects,
        "slice_positions": positions,
    }
