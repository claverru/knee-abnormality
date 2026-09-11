from io import BytesIO
import platform

import numpy as np
import polars as pl
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
from sklearn.metrics import roc_auc_score
import timm
import torch
from torchvision.ops import nms


def check_data_stack() -> None:
    table = pl.DataFrame({"study_id": ["synthetic_a", "synthetic_b"], "target": [0, 1]})
    buffer = BytesIO()
    table.write_parquet(buffer)
    buffer.seek(0)
    assert pl.read_parquet(buffer).equals(table), "Polars Parquet round-trip failed"

    pixels = np.arange(64, dtype="<u2").reshape(8, 8)
    dicom = Dataset()
    dicom.file_meta = FileMetaDataset()
    dicom.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dicom.Rows, dicom.Columns = pixels.shape
    dicom.SamplesPerPixel = 1
    dicom.PhotometricInterpretation = "MONOCHROME2"
    dicom.BitsAllocated = dicom.BitsStored = 16
    dicom.HighBit = 15
    dicom.PixelRepresentation = 0
    dicom.PixelData = pixels.tobytes()
    np.testing.assert_array_equal(dicom.pixel_array, pixels)
    assert roc_auc_score(table["target"].to_numpy(), [0.1, 0.9]) == 1.0
    print("PASS: Polars Parquet, uncompressed DICOM decoding, scikit-learn scoring")


def check_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the NVIDIA driver and PyTorch build")

    torch.manual_seed(0)
    model = timm.create_model("resnet18", pretrained=False, num_classes=3).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    images = torch.randn(2, 3, 64, 64, device="cuda")
    targets = torch.tensor([[1, 0, 1], [0, 1, 0]], device="cuda", dtype=torch.float32)
    parameter = next(model.parameters())
    before = parameter.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    use_bf16 = torch.cuda.is_bf16_supported()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        logits = model(images)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    assert logits.shape == targets.shape and torch.isfinite(logits).all()
    assert torch.isfinite(loss), "Non-finite loss"
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    assert parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
    optimizer.step()
    assert not torch.equal(before, parameter), "Optimizer did not update the weights"

    boxes = torch.tensor([[0, 0, 2, 2], [0, 0, 2, 2]], device="cuda", dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8], device="cuda")
    assert nms(boxes, scores, 0.5).tolist() == [0], "TorchVision CUDA operation failed"
    torch.cuda.synchronize()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    precision = "BF16" if use_bf16 else "FP32"
    print(f"PASS: {precision} forward/backward, AdamW update, TorchVision CUDA NMS; loss={loss.item():.6f}")


if __name__ == "__main__":
    print(f"Python: {platform.python_version()}; PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    check_data_stack()
    check_gpu()
