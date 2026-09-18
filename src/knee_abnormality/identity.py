import hashlib
from pathlib import Path

from knee_abnormality.data.preprocessing.filtering import FILTERING_IDENTITY
from knee_abnormality.data.preprocessing.geometry import (
    RELIABILITY,
)
from knee_abnormality.data.preprocessing.sampling import SAMPLING_IDENTITY
from knee_abnormality.modeling.augmentation import REFLECTION_IDENTITY
from knee_abnormality.modeling.spatial_tokens import (
    SPATIAL_TOKEN_IDENTITY,
    backbone_mapping_identity,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
PREPROCESSING_SOURCES = (
    "data/dataset.py",
    "data/preprocessing/dicom.py",
    "data/preprocessing/exclusions.py",
    "data/preprocessing/filtering.py",
    "data/preprocessing/geometry.py",
    "data/preprocessing/intensity.py",
    "data/preprocessing/preparation.py",
    "data/preprocessing/sampling.py",
    "data/preprocessing/series.py",
    "data/preprocessing/spatial.py",
    "modeling/augmentation.py",
    "modeling/model.py",
    "modeling/spatial_tokens.py",
)


def preprocessing_identity(
    backbone: str,
    package_root: Path = PACKAGE_ROOT,
) -> dict:
    digest = hashlib.sha256()
    for name in PREPROCESSING_SOURCES:
        relative = name.encode()
        payload = (package_root / name).read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    identity = {
        "algorithm": "ranked_filtered_physical_gaussian_overlap_spatial_tokens_reflection_v10",
        "reflection": dict(REFLECTION_IDENTITY),
        "sampling": dict(SAMPLING_IDENTITY),
        "filtering": dict(FILTERING_IDENTITY),
        "reliability": dict(RELIABILITY),
        "spatial_tokens": {
            **SPATIAL_TOKEN_IDENTITY,
            "feature_mapping": backbone_mapping_identity(backbone),
        },
        "sources_sha256": digest.hexdigest(),
    }
    return identity


def validate_preprocessing_identity(
    recorded: object,
    backbone: str,
    package_root: Path = PACKAGE_ROOT,
) -> dict:
    current = preprocessing_identity(backbone, package_root)
    if recorded != current:
        raise ValueError(
            "Checkpoint preprocessing_identity is missing or differs from the executing sources. "
            "Use the checkpoint's archived source/bundle for historical inference, or start a fresh run "
            "with the current sampler. Do not rewrite checkpoint identities."
        )
    return current
