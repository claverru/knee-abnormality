import json
import re
import sys
import tempfile
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
from kagglesdk.kernels.types.kernels_enums import KernelExecutionType
from pydantic import Field
from pydantic_settings import CliImplicitFlag, SettingsError
from requests import HTTPError

from knee_abnormality.cli import CLISettings
from knee_abnormality.inference.bundle import (
    bundle_checkpoints,
    submission_description,
)

LABELS = {
    1: "baseline3 resnet50d",
    2: "baseline4 rexnet200 x1",
    3: "baseline4 rexnet200 x2",
    4: "baseline4 rexnet200 x3",
    5: "overnight 02-longer",
    6: "overnight 04-higher-lr",
    7: "overnight 05-faster-head",
    8: "crop140mm control",
    9: "regional spatial fusion 27 queries",
    10: "spatial tokens five embeddings",
    11: "tonylin soft labels v3",
    12: "3D physical sampling",
    13: "study reflection sigma 80",
    14: "1B shared meniscus and OA heads",
    15: "sigma 40 central sampling",
    16: "1C compartment attention",
    17: "1C compartment attention retry",
    18: "1A shared meniscus head",
    19: "reflection sigma40 champion",
    20: "overnight 00-baseline no 3D",
    21: "champion with unknown_3d inference",
    22: "champion fast runtime",
    23: "EMA 0.999",
    24: "control seed 42",
    25: "control seed 43",
    26: "control seed 44",
    27: "ranked-filtered seed42",
    28: "ranked-filtered seed42 + tau0.75",
    29: "ranked-filtered seed42 + tau0.75 (dup)",
    30: "overnight 07-series-share-1.0",
    31: "ranked-filtered seed42 + tau0.75",
    32: "robust-center-20 topup-26",
    33: "b0-fold0",
    34: "rexnet-fold0",
    35: "ens-5xb0",
    36: "ens-5xb0 + mirror TTA",
    37: "ens-10x mixed + mirror TTA",
}

FOLDED_TEMPERATURES = {28: 0.75, 29: 0.75, 31: 0.75}


class RelabelCLI(CLISettings):
    competition: str = "rsna-knee-abnormality-detection"
    records: Path = Path("outputs/submission/records")
    exports: Path = Path("outputs/submission/exports")
    ledger: Path = Path("outputs/submission/kaggle-ledger.json")
    apply: CliImplicitFlag[bool] = False
    pause_seconds: float = Field(default=4.0, ge=0.0)
    retry_attempts: int = Field(default=6, gt=0)


def _jsonable(value: object) -> dict:
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    return {key: item for key, item in vars(value).items() if not key.startswith("_")}


def _submissions(api: KaggleApi, competition: str) -> list[dict]:
    rows: dict[int, dict] = {}
    page = 1
    while True:
        batch = (
            api.competition_submissions(competition, page_number=page, page_size=100)
            or []
        )
        for row in batch:
            values = _jsonable(row)
            rows[values["ref"]] = values
        if len(batch) < 100:
            break
        page += 1
    return sorted(rows.values(), key=lambda value: str(value["date"]))


def _bundle_path(recorded: str, exports: Path) -> Path | None:
    candidate = Path(recorded)
    if candidate.is_dir():
        return candidate
    moved = exports / candidate.name
    return moved if moved.is_dir() else None


def _local_records(records: Path, exports: Path) -> dict[object, dict]:
    found: dict[object, dict] = {}
    for path in sorted(records.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        notebook = value.get("notebook")
        if not notebook or not value.get("dataset"):
            continue
        entry = {
            "dataset": value["dataset"],
            "bundle": _bundle_path(value.get("bundle") or "", exports),
            "bundle_manifest_sha256": value.get("bundle_manifest_sha256") or "",
        }
        found[notebook] = entry
        found[entry["dataset"]] = entry
        for source in (
            value.get("submission") or {},
            value.get("submission_response") or {},
            value.get("response") or {},
        ):
            reference = source.get("ref") if isinstance(source, dict) else None
            if isinstance(reference, int):
                found[reference] = entry
    return found


def _settings(manifest: dict, folded: float) -> str:
    runtime = manifest.get("runtime") or {}
    baked = bundle_checkpoints(manifest)[0].get("attention_temperature") or 1.0
    temperature = (
        float(runtime.get("attention_temperature") or 1.0) * float(baked) * folded
    )
    parts = []
    if runtime.get("image_multiplier"):
        parts.append(f"x{runtime['image_multiplier']}")
    parts.append(f"tau{f'{temperature:.4f}'.rstrip('0').rstrip('.')}")
    if runtime.get("batch_size"):
        parts.append(f"bs{runtime['batch_size']}")
    if runtime.get("reflection_tta"):
        parts.append("+mirrorTTA")
    return " ".join(parts)


def _validation(manifest: dict, bundle: Path) -> str:
    scores = []
    for checkpoint in bundle_checkpoints(manifest):
        record = ((checkpoint.get("results") or {}).get("epoch_metrics") or {}).get(
            "record"
        ) or {}
        value = record.get("macro_auc")
        if value is None:
            for score in (checkpoint.get("results") or {}).get(
                "checkpoint_callback_scores"
            ) or []:
                if score.get("name") == "best_model_score":
                    value = score.get("value")
        if value is not None:
            scores.append(float(value))
    if not scores:
        return "val0 unrecorded"
    config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    multiplier = (
        (config.get("experiment") or {})
        .get("data", {})
        .get("validation_image_multiplier")
    )
    name = "val0" if multiplier is None else f"val0(x{multiplier})"
    mean = sum(scores) / len(scores)
    if len(scores) > 1:
        return f"{name} {len(scores)}x mean {mean:.6f}"
    return f"{name} {mean:.6f}"


def _notebook_dataset(api: KaggleApi, notebook: str, args: "RelabelCLI") -> str:
    with tempfile.TemporaryDirectory(prefix="knee-sources-") as name:
        folder = Path(name)
        _throttled(
            partial(api.kernels_pull, notebook, str(folder), True),
            args.pause_seconds,
            args.retry_attempts,
        )
        meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    sources = meta.get("dataset_sources") or []
    return sources[0] if sources else ""


def _fallback_label(slug: str) -> str:
    stripped = re.sub(r"^s\d\d-|^knee-|-inference$", "", slug)
    return stripped.replace("-", " ")[:45]


def build_ledger(args: RelabelCLI, api: KaggleApi) -> list[dict]:
    local = _local_records(args.records, args.exports)
    entries = []
    for sequence, submission in enumerate(_submissions(api, args.competition), start=1):
        match = re.match(
            r"/code/([^/?]+)/([^/?]+)\?scriptVersionId=(\d+)",
            submission.get("url") or "",
        )
        slug = match.group(2) if match else ""
        notebook = f"{match.group(1)}/{slug}" if match else ""
        record = local.get(submission["ref"]) or local.get(notebook) or {}
        if not record and notebook:
            record = local.get(_notebook_dataset(api, notebook, args)) or {}
        bundle = record.get("bundle")
        label = LABELS.get(sequence) or _fallback_label(slug)
        settings = "settings unrecorded"
        validation = "val0 unrecorded"
        if bundle is not None:
            manifest = json.loads(
                (bundle / "manifest.json").read_text(encoding="utf-8")
            )
            settings = _settings(manifest, FOLDED_TEMPERATURES.get(sequence, 1.0))
            validation = _validation(manifest, bundle)
        entries.append(
            {
                "sequence": sequence,
                "reference": submission["ref"],
                "date": str(submission["date"]),
                "label": label,
                "title": f"S{sequence:02d} {label}"[:50],
                "description": submission_description(
                    sequence,
                    label,
                    settings,
                    validation,
                    record.get("bundle_manifest_sha256") or "",
                ),
                "previous_description": submission.get("description"),
                "public_score": submission.get("publicScore"),
                "status": str(submission.get("status")),
                "notebook": notebook,
                "notebook_version": match.group(3) if match else "",
                "dataset": record.get("dataset"),
                "bundle": str(bundle) if bundle else None,
                "bundle_manifest_sha256": record.get("bundle_manifest_sha256") or "",
            }
        )
    return entries


def _throttled(call: Callable[[], object], pause: float, attempts: int) -> object:
    delay = max(pause, 1.0)
    for attempt in range(attempts):
        try:
            return call()
        except HTTPError as error:
            status = getattr(error.response, "status_code", None)
            if status != 429 or attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("Unreachable retry state")


def relabel(api: KaggleApi, entries: list[dict], args: RelabelCLI) -> list[dict]:
    primary: dict[str, dict] = {}
    extra: dict[str, list[dict]] = {}
    for entry in entries:
        dataset = entry["dataset"]
        if not dataset:
            continue
        if dataset in primary:
            extra.setdefault(dataset, []).append(entry)
        else:
            primary[dataset] = entry
    results = []
    for dataset, entry in primary.items():
        scores = [
            f"S{value['sequence']:02d} {value['public_score'] or 'unscored'}"
            for value in [entry, *extra.get(dataset, [])]
        ]
        subtitle = f"Submission {scores[0][:3]}, public LB {', '.join(scores)[4:]}"[:80]
        description = (
            f"{entry['description']}\n"
            f"Private inference bundle {entry['bundle_manifest_sha256']}"
        )
        action = {
            "dataset": dataset,
            "title": entry["title"],
            "subtitle": subtitle,
            "description": entry["description"],
            "state": "planned",
        }
        if args.apply:
            with tempfile.TemporaryDirectory(prefix="knee-relabel-") as name:
                directory = Path(name)
                path = Path(
                    _throttled(
                        partial(api.dataset_metadata, dataset, str(directory)),
                        args.pause_seconds,
                        args.retry_attempts,
                    )
                )
                metadata = json.loads(path.read_text(encoding="utf-8"))
                info = metadata.get("info") or metadata
                if not info.get("isPrivate"):
                    raise RuntimeError(f"Dataset {dataset} is not private")
                current = (
                    info.get("title"),
                    info.get("subtitle"),
                    info.get("description"),
                )
                if current == (entry["title"], subtitle, description):
                    action["state"] = "unchanged"
                else:
                    info["title"] = entry["title"]
                    info["subtitle"] = subtitle
                    info["description"] = description
                    metadata["info"] = info
                    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                    _throttled(
                        partial(api.dataset_metadata_update, dataset, str(directory)),
                        args.pause_seconds,
                        args.retry_attempts,
                    )
                    action["state"] = "updated"
                    time.sleep(args.pause_seconds)
        results.append(action)
    return results


def _notebook_source(folder: Path, meta: dict) -> str:
    source = json.loads((folder / meta["code_file"]).read_text(encoding="utf-8"))
    for cell in source.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
        if isinstance(cell.get("source"), list):
            cell["source"] = "".join(cell["source"])
    return json.dumps(source)


def _quick_save(api: KaggleApi, meta: dict, text: str, title: str) -> object:
    with api.build_kaggle_client() as client:
        request = ApiSaveKernelRequest()
        request.id = meta["id_no"]
        request.slug = meta["id"]
        request.new_title = title
        request.text = text
        request.language = meta["language"]
        request.kernel_type = meta["kernel_type"]
        request.is_private = meta["is_private"]
        request.enable_gpu = meta["enable_gpu"]
        request.enable_tpu = meta["enable_tpu"]
        request.enable_internet = meta["enable_internet"]
        request.dataset_data_sources = meta["dataset_sources"]
        request.competition_data_sources = meta["competition_sources"]
        request.kernel_data_sources = meta["kernel_sources"]
        request.model_data_sources = meta["model_sources"]
        request.category_ids = meta["keywords"]
        request.docker_image = meta["docker_image"]
        request.machine_shape = meta["machine_shape"]
        request.kernel_execution_type = KernelExecutionType.QUICK_SAVE
        return client.kernels.kernels_api_client.save_kernel(request)


def rename_notebooks(
    api: KaggleApi, entries: list[dict], args: RelabelCLI
) -> list[dict]:
    primary: dict[str, dict] = {}
    for entry in entries:
        if entry["notebook"] and entry["notebook"] not in primary:
            primary[entry["notebook"]] = entry
    results = []
    for notebook, entry in primary.items():
        action = {
            "notebook": notebook,
            "title": entry["title"],
            "state": "planned",
        }
        if args.apply:
            status = str(api.kernels_status(notebook).status).lower()
            if not any(
                value in status for value in ("complete", "error", "fail", "cancel")
            ):
                action["state"] = f"skipped, notebook status {status}"
                results.append(action)
                continue
            with tempfile.TemporaryDirectory(prefix="knee-rename-") as name:
                folder = Path(name)
                _throttled(
                    partial(api.kernels_pull, notebook, str(folder), True),
                    args.pause_seconds,
                    args.retry_attempts,
                )
                meta = json.loads(
                    (folder / "kernel-metadata.json").read_text(encoding="utf-8")
                )
                if meta["title"] == entry["title"]:
                    action["state"] = "unchanged"
                else:
                    response = _throttled(
                        partial(
                            _quick_save,
                            api,
                            meta,
                            _notebook_source(folder, meta),
                            entry["title"],
                        ),
                        args.pause_seconds,
                        args.retry_attempts,
                    )
                    error = getattr(response, "error", None)
                    if error:
                        raise RuntimeError(f"Quick save failed for {notebook}: {error}")
                    action["state"] = "renamed"
                    action["version"] = getattr(response, "version_number", None)
                    time.sleep(args.pause_seconds)
        results.append(action)
    return results


def main() -> None:
    try:
        args = RelabelCLI()
        api = KaggleApi()
        api.authenticate()
        entries = build_ledger(args, api)
        args.ledger.parent.mkdir(parents=True, exist_ok=True)
        args.ledger.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
        datasets = relabel(api, entries, args)
        notebooks = rename_notebooks(api, entries, args)
        print(
            json.dumps(
                {
                    "entries": len(entries),
                    "datasets": datasets,
                    "notebooks": notebooks,
                },
                indent=2,
            )
        )
    except (OSError, ValueError, TypeError, KeyError, SettingsError) as error:
        print(f"Relabel error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
