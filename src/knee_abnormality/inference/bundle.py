import json
from pathlib import Path

from knee_abnormality.config import validate_attention_temperature
from knee_abnormality.provenance import sha256_file

BUNDLE_SCHEMA_VERSION = 1


def bundle_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }


def verify_bundle(root: Path) -> dict:
    root = root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported bundle schema version: {manifest.get('schema_version')}"
        )
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Bundle manifest has no file inventory")
    actual = bundle_hashes(root)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            name
            for name in set(actual) & set(expected)
            if actual[name] != expected[name]
        )
        raise ValueError(
            f"Bundle file verification failed: missing={missing}, extra={extra}, changed={changed}"
        )
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    checkpoint = manifest.get("checkpoint", {})
    if (
        config.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or config.get("targets") != checkpoint.get("targets")
        or config.get("fold") != checkpoint.get("fold")
        or not isinstance(config.get("experiment"), dict)
    ):
        raise ValueError("Bundle config target order or fold differs from the manifest")
    multiplier = config["inference"]["image_multiplier"]
    if (
        type(multiplier) is not int
        or multiplier < 1
        or multiplier != manifest["runtime"]["image_multiplier"]
    ):
        raise ValueError("Bundle inference image multiplier differs from the manifest")
    temperature = validate_attention_temperature(
        config["inference"]["attention_temperature"]
    )
    if temperature != validate_attention_temperature(
        manifest["runtime"]["attention_temperature"]
    ):
        raise ValueError(
            "Bundle inference attention temperature differs from the manifest"
        )
    return manifest


def discover_kaggle_data_root(base: Path = Path("/kaggle/input")) -> Path:
    base = base.expanduser().resolve()
    candidates = [base]
    candidates.extend(path for path in base.glob("*") if path.is_dir())
    candidates.extend(path for path in base.glob("*/*") if path.is_dir())
    matches = []
    for candidate in candidates:
        if (
            (candidate / "sample_submission.csv").is_file()
            and (candidate / "test_series.csv").is_file()
            and (candidate / "test_series").is_dir()
        ):
            matches.append(candidate)
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one Kaggle competition data root under {base}, found {matches}"
        )
    return matches[0]
