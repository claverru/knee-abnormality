import hashlib
from pathlib import Path

from knee_abnormality.preprocessing.geometry import RELIABILITY
from knee_abnormality.preprocessing.sampling import SAMPLING_IDENTITY
from knee_abnormality.spatial_tokens import SPATIAL_TOKEN_IDENTITY


def preprocessing_identity(package_root: Path | None = None) -> dict:
    root = Path(__file__).resolve().parent if package_root is None else package_root
    paths = [
        *(
            root / name
            for name in (
                "dataset.py",
                "augmentation.py",
                "model.py",
                "spatial_tokens.py",
            )
        ),
        *sorted((root / "preprocessing").rglob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return {
        "algorithm": "physical_gaussian_overlap_spatial_tokens_v1",
        "sampling": dict(SAMPLING_IDENTITY),
        "reliability": dict(RELIABILITY),
        "spatial_tokens": dict(SPATIAL_TOKEN_IDENTITY),
        "sources_sha256": digest.hexdigest(),
    }


def validate_preprocessing_identity(
    recorded: object, package_root: Path | None = None
) -> dict:
    current = preprocessing_identity(package_root)
    if recorded != current:
        raise ValueError(
            "Checkpoint preprocessing_identity is missing or differs from the executing sources. "
            "Use the checkpoint's archived source/bundle for historical inference, or start a fresh run "
            "with the current sampler. Do not rewrite checkpoint identities."
        )
    return current
