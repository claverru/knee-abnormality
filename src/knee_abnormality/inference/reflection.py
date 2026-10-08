import torch


def sagittal_types(series_types) -> torch.Tensor:
    return torch.tensor([kind.plane == "Sagittal" for kind in series_types])


def reflect_batch(
    batch: dict, sagittal_lookup: torch.Tensor, middle_only: bool
) -> dict:
    return {**batch, "mirrored_study": batch["reflection_eligible"]}


def reflect_crops(batch: dict, sagittal_lookup: torch.Tensor) -> dict:
    reflected = batch["reflection_eligible"][batch["series_study_index"]]
    sagittal = sagittal_lookup.to(reflected.device)[batch["series_type_index"]]
    mirror = reflected & ~sagittal
    reverse = reflected & sagittal
    if not (mirror.any() or reverse.any()):
        return batch
    slices = batch["images"].shape[2]
    images = batch["images"]
    rects = batch["valid_rects"]
    positions = batch["slice_positions"].clone()
    patient_maps = batch["patient_maps"]
    crop_affine = batch["crop_affine"].clone()
    canvas_rects = batch["canvas_valid_rects"].clone()
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
        crop_affine[mirror] = crop_affine[mirror] @ reversal
    if reverse.any():
        positions[reverse] = -positions[reverse]
        images[reverse] = images[reverse].flip(2)
        rects[reverse] = rects[reverse].flip(2)
        patient_maps[reverse] = patient_maps[reverse].flip(2)
        canvas_rects[reverse] = canvas_rects[reverse].flip(2)
    return {
        **batch,
        "images": images.expand(-1, -1, slices, -1, -1),
        "valid_rects": rects.expand(-1, -1, slices, -1),
        "slice_positions": positions,
        "patient_maps": patient_maps.expand(-1, -1, slices, -1, -1),
        "crop_affine": crop_affine,
        "canvas_valid_rects": canvas_rects,
        "mirrored_study": batch["reflection_eligible"],
    }
