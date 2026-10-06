import torch


def sagittal_types(series_types) -> torch.Tensor:
    return torch.tensor([kind.plane == "Sagittal" for kind in series_types])


def reflect_batch(
    batch: dict, sagittal_lookup: torch.Tensor, middle_only: bool
) -> dict:
    reflected = batch["reflection_eligible"][batch["series_study_index"]]
    sagittal = sagittal_lookup.to(reflected.device)[batch["series_type_index"]]
    mirror = reflected & ~sagittal
    reverse = reflected & sagittal
    if not (mirror.any() or reverse.any()):
        return batch
    slices = batch["images"].shape[2]
    kept = slice(1, 2) if middle_only else slice(None)
    images = batch["images"][:, :, kept]
    rects = batch["valid_rects"][:, :, kept]
    positions = batch["slice_positions"].clone()
    patient_maps = batch["patient_maps"][:, :, kept]
    view_to_wide = batch["view_to_wide"]
    if mirror.any() or not middle_only:
        images = images.clone()
        rects = rects.clone()
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
        view_to_wide = view_to_wide.clone()
        view_reversal = reversal.to(view_to_wide.dtype)
        view_to_wide[mirror] = view_reversal @ view_to_wide[mirror] @ view_reversal
    if reverse.any():
        positions[reverse] = -positions[reverse]
        if not middle_only:
            images[reverse] = images[reverse].flip(2)
            rects[reverse] = rects[reverse].flip(2)
            patient_maps[reverse] = patient_maps[reverse].flip(2)
    return {
        **batch,
        "images": images.expand(-1, -1, slices, -1, -1),
        "valid_rects": rects.expand(-1, -1, slices, -1),
        "slice_positions": positions,
        "patient_maps": patient_maps.expand(-1, -1, slices, -1, -1),
        "view_to_wide": view_to_wide,
    }
