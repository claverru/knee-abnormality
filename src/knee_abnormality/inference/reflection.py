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
    patient_maps = batch["patient_maps"]
    if mirror.any() or reverse.any():
        images = images.clone()
        rects = rects.clone()
        positions = positions.clone()
        patient_maps = patient_maps.clone()
    if mirror.any():
        images[mirror] = images[mirror].flip(-1)
        selected = rects[mirror]
        selected[..., 1] = images.shape[-1] - selected[..., 3] - selected[..., 1]
        rects[mirror] = selected
        reversal = torch.eye(3, device=patient_maps.device, dtype=patient_maps.dtype)
        reversal[0, 0] = -1
        reversal[0, 2] = images.shape[-1] - 1
        patient_maps[mirror] = patient_maps[mirror] @ reversal
    if reverse.any():
        images[reverse] = images[reverse].flip(2)
        rects[reverse] = rects[reverse].flip(2)
        positions[reverse] = -positions[reverse]
        patient_maps[reverse] = patient_maps[reverse].flip(2)
    return {
        **batch,
        "images": images,
        "valid_rects": rects,
        "slice_positions": positions,
        "patient_maps": patient_maps,
    }
