import csv
import json
import math
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from kaggle.api.kaggle_api_extended import KaggleApi
from pydantic import Field, StringConstraints
from pydantic_settings import SettingsError
from requests import HTTPError

from knee_abnormality.cli import CLISettings
from knee_abnormality.inference.bundle import (
    bundle_validation_multiplier,
    submission_description,
    submission_settings,
    submission_validation,
    verify_bundle,
)
from knee_abnormality.provenance import sha256_file

KaggleId = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
]


class SubmitCLI(CLISettings):
    bundle: Path
    dataset: KaggleId
    notebook: KaggleId
    record: Path
    competition: str = "rsna-knee-abnormality-detection"
    label: str
    sequence: int | None = Field(default=None, gt=0)
    dataset_license: str = "other"
    accelerator: str = "NvidiaTeslaT4"
    notebook_timeout_seconds: int = Field(default=32400, gt=0, le=32400)
    poll_seconds: int = Field(default=30, gt=0)
    max_wait_seconds: int = Field(default=36000, gt=0)


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dict__"):
        return {
            key: _jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def _status(value: object) -> str:
    text = str(value).lower()
    return text.rsplit(".", 1)[-1]


def _write_record(path: Path, record: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _update(path: Path, record: dict, **values: object) -> None:
    previous_state = record.get("state")
    record.update(values)
    record["updated_at"] = datetime.now(UTC).isoformat()
    if record.get("state") != previous_state:
        event = {"state": record["state"], "at": record["updated_at"]}
        record.setdefault("history", []).append(event)
        print(json.dumps(event), flush=True)
    _write_record(path, record)


def _poll_dataset(
    api: KaggleApi, dataset: str, minimum_version: int, args: SubmitCLI
) -> str:
    deadline = time.monotonic() + args.max_wait_seconds
    while True:
        try:
            identity = _dataset_identity(api, dataset)
        except RuntimeError:
            identity = None
        if (
            identity is not None
            and identity["version"] >= minimum_version
            and identity["total_bytes"] > 0
        ):
            return "ready"
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for dataset {dataset}")
        time.sleep(args.poll_seconds)


def _dataset_identity(api: KaggleApi, dataset: str) -> dict:
    owner, slug = dataset.split("/", 1)
    matches = [
        value
        for value in (api.dataset_list(search=slug, user=owner) or [])
        if value is not None and value.ref == dataset
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Could not resolve one uploaded dataset record for {dataset}"
        )
    value = matches[0]
    if not value.is_private:
        raise RuntimeError(f"Dataset {dataset} is not private")
    if value.current_version_number < 1:
        raise RuntimeError(f"Dataset {dataset} has no resolved version number")
    return {
        "ref": value.ref,
        "version": value.current_version_number,
        "url": value.url,
        "is_private": value.is_private,
        "total_bytes": value.total_bytes,
        "last_updated": str(value.last_updated),
    }


def _dataset_exists(api: KaggleApi, dataset: str) -> bool:
    owner, slug = dataset.split("/", 1)
    return any(
        value is not None and value.ref == dataset
        for value in (api.dataset_list(search=slug, user=owner) or [])
    )


def _kernel_status(api: KaggleApi, notebook: str) -> tuple[str, object]:
    response = api.kernels_status(notebook)
    return _status(response.status), response


def _kernel_exists(api: KaggleApi, notebook: str) -> bool:
    owner, slug = notebook.split("/", 1)
    matches = [
        value
        for value in (api.kernels_list(search=slug, user=owner) or [])
        if value is not None and value.ref == notebook
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Resolved multiple notebook records for {notebook}")
    return bool(matches)


def _poll_kernel(api: KaggleApi, notebook: str, args: SubmitCLI) -> object:
    deadline = time.monotonic() + args.max_wait_seconds
    while True:
        status, response = _kernel_status(api, notebook)
        if "complete" in status:
            return response
        if any(value in status for value in ("error", "fail", "cancel")):
            message = getattr(response, "failure_message", None)
            raise RuntimeError(
                f"Notebook {notebook} reached status {status}: {message}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for notebook {notebook}: {status}")
        time.sleep(args.poll_seconds)


def _validate_submission(path: Path, targets: list[str]) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    expected = ["StudyInstanceUID", *targets]
    if not rows or rows[0] != expected or len(rows) < 2:
        raise ValueError("Notebook output has the wrong submission schema or no rows")
    ids = [row[0] for row in rows[1:]]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("Notebook output has missing or duplicate study IDs")
    for row in rows[1:]:
        if len(row) != len(expected):
            raise ValueError(f"Notebook output row has the wrong width: {row[0]}")
        values = [float(value) for value in row[1:]]
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
            raise ValueError(f"Notebook output has invalid probabilities: {row[0]}")
    return {"path": str(path), "sha256": sha256_file(path), "rows": len(rows) - 1}


def _notebook_reference(response: object, requested: str) -> str:
    reference = str(getattr(response, "ref", "") or "")
    slug = reference.rsplit("/code/", 1)[-1].strip("/")
    return slug if slug.count("/") == 1 else requested


def _submission_sequence(api: KaggleApi, competition: str) -> int:
    references = set()
    page = 1
    while True:
        rows = (
            api.competition_submissions(competition, page_number=page, page_size=100)
            or []
        )
        for row in rows:
            values = _jsonable(row)
            if isinstance(values, dict):
                references.add(values.get("ref") or values.get("id"))
        if len(rows) < 100:
            return len(references) + 1
        page += 1


def _submission_match(
    api: KaggleApi, args: SubmitCLI, response: object, description: str
) -> object | None:
    submissions = api.competition_submissions(args.competition, page_size=50) or []
    response_values = _jsonable(response)
    response_id = None
    if isinstance(response_values, dict):
        response_id = response_values.get("ref") or response_values.get("id")
    for submission in submissions:
        values = _jsonable(submission)
        if not isinstance(values, dict):
            continue
        candidate_id = values.get("ref") or values.get("id")
        if response_id is not None and candidate_id == response_id:
            return submission
        candidate = values.get("description") or values.get("submissionDescription")
        if response_id is None and candidate == description:
            return submission
    return None


def _poll_submission(
    api: KaggleApi, args: SubmitCLI, response: object, description: str
) -> object | None:
    deadline = time.monotonic() + args.max_wait_seconds
    last = None
    while True:
        last = _submission_match(api, args, response, description)
        if last is not None:
            values = _jsonable(last)
            status = (
                _status(values.get("status", "")) if isinstance(values, dict) else ""
            )
            if status in {"complete", "finished"}:
                return last
            if status in {"error", "failed", "cancelled"}:
                return last
        if time.monotonic() >= deadline:
            return last
        time.sleep(args.poll_seconds)


def submit(args: SubmitCLI) -> dict:
    bundle = args.bundle.expanduser().resolve()
    record_path = args.record.expanduser().resolve()
    if record_path.exists():
        raise FileExistsError(f"Submission record already exists: {record_path}")
    label = args.label.strip()
    if not 2 <= len(label) <= 45:
        raise ValueError("Submission label must contain 2 to 45 characters")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = verify_bundle(bundle)
    if not manifest.get("ready"):
        raise ValueError(
            f"Bundle has unmet prerequisites: {manifest.get('unmet_prerequisites')}"
        )
    record = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "bundle": str(bundle),
        "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "dataset": args.dataset,
        "notebook": args.notebook,
        "competition": args.competition,
        "state": "starting",
    }
    record["history"] = [{"state": "starting", "at": record["created_at"]}]
    _write_record(record_path, record)
    api = KaggleApi()
    try:
        api.authenticate()
        sequence = args.sequence or _submission_sequence(api, args.competition)
        title = f"S{sequence:02d} {label}"
        description = submission_description(
            sequence,
            label,
            submission_settings(manifest),
            submission_validation(manifest, bundle_validation_multiplier(bundle)),
            record["bundle_manifest_sha256"],
        )
        _update(
            record_path,
            record,
            sequence=sequence,
            title=title,
            description=description,
        )
        if _kernel_exists(api, args.notebook):
            current_status, current = _kernel_status(api, args.notebook)
            if not any(
                value in current_status
                for value in ("complete", "error", "fail", "cancel")
            ):
                raise RuntimeError(
                    f"Notebook {args.notebook} already has an active run: {current_status}"
                )
            record["previous_notebook_status"] = _jsonable(current)
        with tempfile.TemporaryDirectory(prefix="knee-submit-") as temporary_name:
            temporary = Path(temporary_name)
            dataset_dir = temporary / "dataset"
            dataset_dir.mkdir()
            shutil.copytree(bundle, dataset_dir / "bundle")
            metadata = {
                "id": args.dataset,
                "title": title,
                "licenses": [{"name": args.dataset_license}],
                "description": (
                    f"{description}\n"
                    f"Private inference bundle {record['bundle_manifest_sha256']}"
                ),
            }
            (dataset_dir / "dataset-metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            if _dataset_exists(api, args.dataset):
                existing_identity = _dataset_identity(api, args.dataset)
                existing_metadata_dir = temporary / "existing-dataset-metadata"
                existing_metadata_path = Path(
                    api.dataset_metadata(args.dataset, str(existing_metadata_dir))
                )
                existing_metadata = json.loads(
                    existing_metadata_path.read_text(encoding="utf-8")
                )
                description = existing_metadata.get("info", {}).get("description")
                if description == metadata["description"]:
                    dataset_response = {"reused_version": True}
                    dataset_action = "reuse"
                    expected_dataset_version = existing_identity["version"]
                else:
                    dataset_response = api.dataset_create_version(
                        str(dataset_dir),
                        f"Bundle {record['bundle_manifest_sha256']}",
                        quiet=False,
                        dir_mode="zip",
                    )
                    dataset_action = "version"
                    expected_dataset_version = existing_identity["version"] + 1
            else:
                dataset_response = api.dataset_create_new(
                    str(dataset_dir), public=False, quiet=False, dir_mode="zip"
                )
                dataset_action = "create"
                expected_dataset_version = 1
            if dataset_response is None or getattr(dataset_response, "error", None):
                raise RuntimeError(
                    f"Dataset upload failed: {_jsonable(dataset_response)}"
                )
            _update(
                record_path,
                record,
                state="dataset_uploaded",
                dataset_action=dataset_action,
                dataset_response=_jsonable(dataset_response),
            )
            dataset_status = _poll_dataset(
                api, args.dataset, expected_dataset_version, args
            )
            dataset_identity = _dataset_identity(api, args.dataset)
            _update(
                record_path,
                record,
                state="dataset_ready",
                dataset_status=dataset_status,
                dataset_version=dataset_identity,
            )
            dataset_metadata_dir = temporary / "dataset-metadata"
            dataset_metadata_path = Path(
                api.dataset_metadata(args.dataset, str(dataset_metadata_dir))
            )
            _update(
                record_path,
                record,
                dataset_remote_metadata=json.loads(
                    dataset_metadata_path.read_text(encoding="utf-8")
                ),
            )
            notebook_dir = temporary / "notebook"
            notebook_dir.mkdir()
            shutil.copy2(bundle / "inference.ipynb", notebook_dir / "inference.ipynb")
            notebook_metadata = {
                "id": args.notebook,
                "title": title,
                "code_file": "inference.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": False,
                "competition_sources": [args.competition],
                "dataset_sources": [args.dataset],
            }
            (notebook_dir / "kernel-metadata.json").write_text(
                json.dumps(notebook_metadata, indent=2) + "\n", encoding="utf-8"
            )
            kernel_response = api.kernels_push(
                str(notebook_dir),
                str(args.notebook_timeout_seconds),
                args.accelerator,
            )
            version_number = getattr(kernel_response, "versionNumber", None)
            if (
                kernel_response is None
                or getattr(kernel_response, "error", None)
                or not version_number
            ):
                raise RuntimeError(
                    f"Notebook push failed: {_jsonable(kernel_response)}"
                )
            notebook = _notebook_reference(kernel_response, args.notebook)
            _update(
                record_path,
                record,
                state="notebook_running",
                notebook=notebook,
                requested_notebook=args.notebook,
                notebook_version=version_number,
                notebook_push_response=_jsonable(kernel_response),
            )
            kernel_status = _poll_kernel(api, notebook, args)
            log_path = record_path.parent / f"{record_path.stem}-notebook.log"
            if log_path.exists():
                raise FileExistsError(f"Notebook log already exists: {log_path}")
            log_path.write_text(api.kernels_logs(notebook), encoding="utf-8")
            _update(
                record_path,
                record,
                state="notebook_complete",
                notebook_status=_jsonable(kernel_status),
                notebook_log={
                    "path": str(log_path),
                    "sha256": sha256_file(log_path),
                },
            )
            output_dir = record_path.parent / f"{record_path.stem}-notebook-output"
            if output_dir.exists():
                raise FileExistsError(f"Notebook output directory exists: {output_dir}")
            output_dir.mkdir()
            api.kernels_output(notebook, path=str(output_dir), force=False, quiet=False)
            matches = sorted(output_dir.rglob("submission.csv"))
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one notebook submission.csv, found {matches}"
                )
            output_record = _validate_submission(
                matches[0],
                manifest["checkpoint"]["targets"]
                if "checkpoint" in manifest
                else manifest["checkpoints"][0]["targets"],
            )
            _update(
                record_path,
                record,
                state="output_verified",
                notebook_output=output_record,
            )
            submission_response = api.competition_submit_code(
                "submission.csv",
                description,
                args.competition,
                notebook,
                int(version_number),
                False,
            )
            if submission_response is None or getattr(
                submission_response, "error", None
            ):
                raise RuntimeError(
                    f"Competition submission failed: {_jsonable(submission_response)}"
                )
            _update(
                record_path,
                record,
                state="submitted",
                submission_response=_jsonable(submission_response),
            )
            submission = _poll_submission(api, args, submission_response, description)
            submission_values = _jsonable(submission)
            submission_status = (
                _status(submission_values.get("status", ""))
                if isinstance(submission_values, dict)
                else "pending"
            )
            final_state = (
                "submission_complete"
                if submission_status in {"complete", "finished"}
                else "submission_pending"
            )
            _update(
                record_path,
                record,
                state=final_state,
                submission=submission_values,
            )
            if submission_status in {"error", "failed", "cancelled"}:
                raise RuntimeError(
                    f"Competition submission reached status {submission_status}"
                )
            return record
    except BaseException as error:
        _update(
            record_path,
            record,
            state="failed",
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise


def main() -> None:
    try:
        args = SubmitCLI()
        record = submit(args)
        print(json.dumps(record))
    except (
        HTTPError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        SettingsError,
    ) as error:
        print(f"Submission error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
