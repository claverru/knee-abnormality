import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel, Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.provenance import sha256_file

GIB = 1024**3
ARRAY_BYTES = 50 * 1024**2
ARRAY_SUFFIXES = {".npy", ".npz", ".pt", ".pkl"}
ARRAY_AREAS = ("analysis", "profiling", "previews")
BUILD_MARKERS = ("CACHEDIR.TAG", ".rustc_info.json", "pyvenv.cfg")
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".json", ".sh", ".md", ".txt"}
SKIPPED_FOLDERS = {".venv", ".git", "__pycache__"}
FOLD = re.compile(r"^fold_\d+$")
RUN_STAMP = re.compile(r"-\d{8}T\d{6}Z-[0-9a-f]{8}(-\d+)?$")
RUN_PREFIX = re.compile(r"^\d{8}-\d{6}-")
RUN_DATE = re.compile(r"-\d{8}$")
SESSION = re.compile(r"^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$")

LIVE = "Used by a live process"
UNFINISHED = "Run without a selection record"
ACTIVE = "Campaign is not stopped"
SUBMITTED = "Source of a submitted bundle in the ledger"
ONGOING = "Named in docs/ongoing or the final solution"
PRETRAINING = "Last checkpoint of a pretraining, the start weights of later stages"
UNRECORDED = "Result not recorded in the docs or a campaign state"
NO_RULE = "No rule covers this location"
ATTENTION = (UNFINISHED, UNRECORDED, NO_RULE)


class Record(BaseModel):
    path: str
    action: Literal["delete", "keep"]
    rule: str
    reason: str
    size_bytes: int
    allocated_bytes: int
    files: int
    mtime_ns: int
    device: int | None = None
    inode: int | None = None
    links: int | None = None
    error: str | None = None


class Selection(BaseModel):
    path: Path


class CampaignState(BaseModel):
    status: str
    revision: int


class CampaignIdentity(BaseModel):
    sha256: str
    revision: int


class Session(BaseModel):
    pid: int
    session_id: str = Field(alias="sessionId")


class LedgerEntry(BaseModel):
    bundle: Path | None = None


class ExportedCheckpoint(BaseModel):
    source_path: Path | None = None


class BundleManifest(BaseModel):
    checkpoint: ExportedCheckpoint | None = None
    checkpoints: list[ExportedCheckpoint] = []


class Manifest(BaseModel):
    version: int = 2
    created_at: str
    status: Literal["planned", "complete"]
    scope: str
    roots: list[str]
    records: list[Record]
    campaign_states: dict[str, CampaignIdentity]
    filesystem_before: dict[str, int]
    completed_at: str | None = None
    summary: dict[str, dict[str, int]] = {}
    filesystem_after: dict[str, int] = {}


class FreeDiskCLI(CLISettings):
    model_config = describe(
        "Free disk space by fixed rules: delete obsolete checkpoints, export weights "
        "of unsubmitted models, regenerable analysis arrays, download chunks and tool "
        "caches across every worktree and the home caches, and write a manifest of "
        "every deleted and kept path"
    )

    dry_run: bool = Field(
        default=False,
        description="Print the plan without deleting anything or writing a manifest",
    )


def named(names: Iterable[str], text: str) -> bool:
    return any(
        re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text) for name in names
    )


def measure(path: Path) -> dict[str, int]:
    status = path.lstat()
    if not stat.S_ISDIR(status.st_mode):
        return {
            "size_bytes": status.st_size,
            "allocated_bytes": status.st_blocks * 512,
            "files": 1,
            "mtime_ns": status.st_mtime_ns,
            "device": status.st_dev,
            "inode": status.st_ino,
            "links": status.st_nlink,
        }
    size = allocated = files = 0
    newest = status.st_mtime_ns
    for folder, _, names in os.walk(path):
        for name in names:
            try:
                entry = os.lstat(os.path.join(folder, name))
            except FileNotFoundError:
                continue
            size += entry.st_size
            allocated += entry.st_blocks * 512
            files += 1
            newest = max(newest, entry.st_mtime_ns)
    return {
        "size_bytes": size,
        "allocated_bytes": allocated,
        "files": files,
        "mtime_ns": newest,
    }


def read_text(folders: Iterable[Path]) -> str:
    parts = []
    for folder in folders:
        if folder.is_file():
            parts.append(folder.read_text(errors="ignore"))
        for current, names, files in os.walk(folder):
            names[:] = [name for name in names if name not in SKIPPED_FOLDERS]
            for name in files:
                path = Path(current, name)
                if path.suffix in TEXT_SUFFIXES and path.stat().st_size < 1_000_000:
                    parts.append(path.read_text(errors="ignore"))
    return "\n".join(parts)


def worktrees() -> list[Path]:
    listing = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    roots = [
        Path(line.removeprefix("worktree "))
        for line in listing.splitlines()
        if line.startswith("worktree ")
    ]
    return [root for root in roots if root.is_dir()]


def processes() -> tuple[str, list[str]]:
    commands = []
    paths = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            commands.append(
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="ignore")
            )
            paths.append(os.readlink(entry / "cwd"))
            paths.extend(os.readlink(link) for link in (entry / "fd").iterdir())
        except OSError:
            continue
    return "\n".join(commands), paths


def run_names(run: Path) -> set[str]:
    names = {run.name}
    short = RUN_DATE.sub("", RUN_PREFIX.sub("", RUN_STAMP.sub("", run.name)))
    if "-" in short:
        names.add(short)
    if run.parent.name == "runs":
        names.add(run.parent.parent.name)
    return names


def fold_run(fold: Path) -> Path:
    return fold.parent.parent if fold.parent.name == "training" else fold.parent


def bundle_sources(bundle: Path) -> list[Path]:
    manifest = bundle / "manifest.json"
    if not manifest.exists():
        return []
    data = BundleManifest.model_validate_json(manifest.read_text())
    entries = data.checkpoints or ([data.checkpoint] if data.checkpoint else [])
    return [entry.source_path for entry in entries if entry.source_path]


@dataclass
class Scan:
    roots: list[Path]
    ongoing: str
    docs: str
    code: str
    commands: str
    open_paths: list[str]
    entries: list[tuple[Path, Record]] = field(default_factory=list)
    campaigns: dict[str, CampaignIdentity] = field(default_factory=dict)

    def label(self, path: Path) -> str:
        main = self.roots[0]
        return str(path.relative_to(main) if path.is_relative_to(main) else path)

    def live(self, path: Path, names: Iterable[str] = ()) -> bool:
        target = str(path)
        return (
            target in self.commands
            or named(names, self.commands)
            or any(
                opened == target or opened.startswith(f"{target}/")
                for opened in self.open_paths
            )
        )

    def hold(self, path: Path, names: Iterable[str] = ()) -> str | None:
        return LIVE if self.live(path, names) else None

    def add(
        self,
        path: Path,
        rule: str,
        keep: str | None,
        delete: str,
        stats: dict[str, int] | None = None,
    ) -> None:
        try:
            stats = stats or measure(path)
        except FileNotFoundError:
            return
        self.entries.append(
            (
                path,
                Record(
                    path=self.label(path),
                    action="keep" if keep else "delete",
                    rule=rule,
                    reason=keep or delete,
                    **stats,
                ),
            )
        )


def scan_folds(
    scan: Scan, outputs: Path, folds: dict[Path, list[Path]], sources: set[Path]
) -> None:
    states: dict[Path, tuple[CampaignState | None, str]] = {}
    for fold, files in sorted(folds.items()):
        run = fold_run(fold)
        names = run_names(run)
        parts = fold.relative_to(outputs).parts
        recorded = scan.docs
        active = False
        if parts[0] == "autoresearch":
            campaign = outputs / "autoresearch" / parts[1]
            if campaign not in states:
                path = campaign / "state.json"
                text = path.read_text() if path.exists() else ""
                state = CampaignState.model_validate_json(text) if text else None
                states[campaign] = (state, text)
                if state:
                    scan.campaigns[scan.label(path)] = CampaignIdentity(
                        sha256=sha256_file(path), revision=state.revision
                    )
            state, text = states[campaign]
            recorded = f"{recorded}\n{text}"
            active = state is None or state.status != "stopped"
        selection = fold / "selected_checkpoint.json"
        selected = None
        if active:
            hold = ACTIVE
        elif not selection.exists():
            hold = UNFINISHED
        else:
            selected = Selection.model_validate_json(selection.read_text()).path.name
            hold = scan.hold(run, names)
        for file in sorted(files):
            if file.resolve() in sources:
                scan.add(file, "checkpoints", SUBMITTED, "")
            elif hold:
                scan.add(file, "checkpoints", hold, "")
            elif file.name == "last.ckpt" and any(
                name.endswith("-pretrain") for name in names
            ):
                scan.add(file, "checkpoints", PRETRAINING, "")
            elif file.name == "last.ckpt":
                scan.add(
                    file, "checkpoints", None, "Resume checkpoint of a finished run"
                )
            elif file.name != selected:
                scan.add(
                    file,
                    "checkpoints",
                    None,
                    "Non-selected checkpoint of a finished run",
                )
            elif named(names, scan.ongoing):
                scan.add(file, "checkpoints", ONGOING, "")
            elif not named(names, recorded):
                scan.add(file, "checkpoints", UNRECORDED, "")
            else:
                scan.add(
                    file,
                    "checkpoints",
                    None,
                    "Recorded run outside docs/ongoing, the final solution and the ledger",
                )


def scan_exports(scan: Scan, exports: Path, submitted: set[Path]) -> None:
    for bundle in sorted(exports.iterdir()):
        if not bundle.is_dir() or bundle.is_symlink():
            continue
        weights = [
            path
            for path in (
                bundle / "model.ckpt",
                *sorted((bundle / "models").glob("*.ckpt")),
                bundle / "wheels",
            )
            if path.exists() and not path.is_symlink()
        ]
        if not weights:
            continue
        names = {bundle.name}
        for source in bundle_sources(bundle):
            names |= run_names(fold_run(source.parent))
        if bundle.resolve() in submitted:
            keep = "Submitted bundle in the ledger"
        elif named(names, scan.ongoing):
            keep = ONGOING
        else:
            keep = scan.hold(bundle, {bundle.name})
        unrecorded = not named(names, scan.docs)
        for path in weights:
            scan.add(
                path,
                "exports",
                keep or (UNRECORDED if unrecorded and path.is_file() else None),
                "Export weights of a model outside docs/ongoing, the final solution and the ledger",
            )


def scan_arrays(scan: Scan, outputs: Path) -> None:
    for area in ARRAY_AREAS:
        if not (outputs / area).is_dir():
            continue
        for investigation in sorted((outputs / area).iterdir()):
            if not investigation.is_dir() or investigation.is_symlink():
                continue
            arrays = []
            for folder, _, files in os.walk(investigation):
                for name in files:
                    path = Path(folder, name)
                    try:
                        status = path.lstat()
                    except FileNotFoundError:
                        continue
                    if (
                        stat.S_ISREG(status.st_mode)
                        and path.suffix in ARRAY_SUFFIXES
                        and status.st_size >= ARRAY_BYTES
                    ):
                        arrays.append(path)
            if not arrays:
                continue
            names = {investigation.name}
            if named(names, scan.ongoing):
                keep = ONGOING
            elif named(names, scan.code):
                keep = "Path named in scripts, src or configs"
            else:
                keep = scan.hold(investigation, names)
            for path in sorted(arrays):
                scan.add(
                    path,
                    "arrays",
                    keep,
                    "Regenerable array of an investigation outside docs/ongoing and the final solution",
                )


def existing(roots: list[Path], *parts: str) -> list[Path]:
    return sorted(
        {
            root.joinpath(*parts).resolve()
            for root in roots
            if root.joinpath(*parts).is_dir()
        }
    )


def submitted_bundles(outputs: list[Path]) -> set[Path]:
    submitted = set()
    for folder in outputs:
        ledger = folder / "submission" / "kaggle-ledger.json"
        if not ledger.exists():
            continue
        for row in json.loads(ledger.read_text()):
            entry = LedgerEntry.model_validate(row)
            if entry.bundle:
                submitted.add((folder.parent / entry.bundle).resolve())
    return submitted


def scan_outputs(
    scan: Scan, outputs: Path, submitted: set[Path], sources: set[Path]
) -> None:
    exports = outputs / "submission" / "exports"
    folds = defaultdict(list)
    for folder, _, files in os.walk(outputs):
        here = Path(folder)
        if here.is_relative_to(exports):
            continue
        for name in files:
            path = here / name
            if not name.endswith(".ckpt") or path.is_symlink():
                continue
            if FOLD.match(here.name):
                folds[here].append(path)
            else:
                scan.add(path, "checkpoints", NO_RULE, "")
    scan_folds(scan, outputs, folds, sources)
    if exports.is_dir():
        scan_exports(scan, exports, submitted)
    scan_arrays(scan, outputs)


def scan_download(scan: Scan, state: Path) -> None:
    chunks = state / "chunks"
    if not chunks.is_dir() or not any(chunks.iterdir()):
        return
    keep = (
        scan.hold(state)
        if (state / "complete.json").exists()
        else "Download is not complete"
    )
    scan.add(chunks, "download", keep, "Chunk cache of a completed download")


def scan_huggingface(scan: Scan, hub: Path) -> None:
    for model in sorted(hub.glob("models--*")):
        repository = model.name.removeprefix("models--").replace("--", "/")
        if named({repository, repository.split("/", 1)[-1]}, scan.code):
            keep = "Model named in scripts, src, configs or z_labels"
        else:
            keep = scan.hold(model)
        scan.add(model, "huggingface", keep, "Cached model no code or config names")


def build_folders(folder: Path) -> Iterator[Path]:
    for current, names, files in os.walk(folder):
        here = Path(current)
        if here.name == "__pycache__" or any(
            marker in files for marker in BUILD_MARKERS
        ):
            names.clear()
            yield here


def scan_codex(scan: Scan, codex: Path) -> None:
    packages = codex / "packages"
    for package in sorted(packages.iterdir()) if packages.is_dir() else []:
        current = package / "current"
        releases = package / "releases"
        if not current.is_symlink() or not releases.is_dir():
            continue
        target = current.resolve()
        for release in sorted(releases.iterdir()):
            keep = (
                "Current release" if release.resolve() == target else scan.hold(release)
            )
            scan.add(release, "codex-releases", keep, "Superseded release")
    for area in ("adversarial-reviews", "debates"):
        for build in sorted(build_folders(codex / area)):
            owner = codex / area / build.relative_to(codex / area).parts[0]
            scan.add(
                build,
                "codex-builds",
                scan.hold(owner),
                "Build output or environment inside a review or debate",
            )


def running_sessions(registry: Path) -> set[str]:
    sessions = (
        Session.model_validate_json(path.read_text())
        for path in registry.glob("*.json")
    )
    return {
        session.session_id
        for session in sessions
        if Path("/proc", str(session.pid)).exists()
    }


def scan_scratchpads(scan: Scan, base: Path, running: set[str]) -> None:
    for project in sorted(base.iterdir()) if base.is_dir() else []:
        if not project.is_dir() or project.is_symlink():
            continue
        for session in sorted(project.iterdir()):
            if not session.is_dir() or not SESSION.match(session.name):
                continue
            keep = (
                "Session is running"
                if session.name in running
                else scan.hold(session, {session.name})
            )
            scan.add(session, "scratchpads", keep, "Scratchpad of a finished session")


def summarize(records: list[Record]) -> pl.DataFrame:
    return (
        pl.DataFrame([record.model_dump() for record in records])
        .group_by("action", "rule", "reason")
        .agg(
            pl.len().alias("entries"),
            pl.col("files").sum(),
            (pl.col("allocated_bytes").sum() / GIB).round(3).alias("gib"),
        )
        .sort("action", "rule", "gib", descending=[False, False, True])
    )


def report(records: list[Record]) -> None:
    with pl.Config(
        tbl_rows=-1,
        tbl_width_chars=200,
        fmt_str_lengths=80,
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
    ):
        print(summarize(records))
    attention = [record for record in records if record.reason in ATTENTION]
    if attention:
        print("Kept, needs a decision:")
        for record in attention:
            print(f"  {record.reason}: {record.path}")


def remove(path: Path, record: Record, bases: list[Path]) -> None:
    if not any(path.is_relative_to(base) for base in bases):
        raise RuntimeError(f"{path} is outside the cleanup locations")
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError as error:
        record.error = str(error)


def main() -> None:
    args = FreeDiskCLI()
    roots = worktrees()
    home = Path.home()
    hub = home / ".cache" / "huggingface" / "hub"
    codex = home / ".codex"
    scratch = Path(tempfile.gettempdir()) / f"claude-{os.getuid()}"
    commands, open_paths = processes()
    scan = Scan(
        roots=roots,
        ongoing=read_text(
            path
            for root in roots
            for path in (root / "docs" / "ongoing", root / "docs" / "final-solution.md")
        ),
        docs=read_text(root / "docs" for root in roots),
        code=read_text(
            root / name
            for root in roots
            for name in ("scripts", "src", "configs", "z_labels")
        ),
        commands=commands,
        open_paths=open_paths,
    )
    outputs = existing(roots, "outputs")
    downloads = existing(roots, "data", "raw", ".download")
    submitted = submitted_bundles(outputs)
    sources = {
        source.resolve() for bundle in submitted for source in bundle_sources(bundle)
    }
    for folder in outputs:
        scan_outputs(scan, folder, submitted, sources)
    for state in downloads:
        scan_download(scan, state)
    scan_huggingface(scan, hub)
    scan_codex(scan, codex)
    scan_scratchpads(scan, scratch, running_sessions(home / ".claude" / "sessions"))
    records = [record for _, record in scan.entries]
    if not records:
        return
    report(records)
    if args.dry_run:
        print("Would delete:")
        for record in records:
            if record.action == "delete":
                print(f"  {record.allocated_bytes / GIB:7.3f} GiB  {record.path}")
        return
    main_root = roots[0]
    stamp = datetime.now(UTC)
    path = (
        main_root
        / "outputs"
        / "archive"
        / "maintenance"
        / f"disk-cleanup-{stamp:%Y%m%dT%H%M%SZ}"
        / "manifest.json"
    )
    manifest = Manifest(
        created_at=stamp.isoformat(),
        status="planned",
        scope=(
            "Checkpoints, export weights, analysis arrays, download chunks, cached "
            "models, Codex releases and builds, session scratchpads"
        ),
        roots=[str(root) for root in roots],
        records=records,
        campaign_states=scan.campaigns,
        filesystem_before={"available_bytes": shutil.disk_usage(main_root).free},
    )
    path.parent.mkdir(parents=True)
    path.write_text(manifest.model_dump_json(indent=2))
    bases = [
        *outputs,
        *(state / "chunks" for state in downloads),
        hub,
        codex / "packages",
        codex / "adversarial-reviews",
        codex / "debates",
        scratch,
    ]
    for target, record in scan.entries:
        if record.action == "delete":
            remove(target, record, bases)
    manifest.status = "complete"
    manifest.completed_at = datetime.now(UTC).isoformat()
    manifest.summary = {
        action: {
            "entries": sum(record.action == action for record in records),
            "files": sum(record.files for record in records if record.action == action),
            "bytes": sum(
                record.allocated_bytes for record in records if record.action == action
            ),
        }
        for action in ("delete", "keep")
    }
    manifest.filesystem_after = {"available_bytes": shutil.disk_usage(main_root).free}
    path.write_text(manifest.model_dump_json(indent=2))
    freed = (
        manifest.filesystem_after["available_bytes"]
        - manifest.filesystem_before["available_bytes"]
    )
    failed = [record for record in records if record.error]
    print(f"Freed {freed / GIB:.2f} GiB; manifest {scan.label(path)}")
    for record in failed:
        print(f"  not deleted: {record.path}: {record.error}")


if __name__ == "__main__":
    main()
