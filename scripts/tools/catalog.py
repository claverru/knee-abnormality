import json
import sys
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from pydantic_settings import CliPositionalArg, SettingsError

from knee_abnormality.cli import NonExitingCLISettings

ROOT = Path(__file__).resolve().parents[2]
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9-]*$")]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Role = Literal["pipeline", "tools", "adhoc"]
Status = Literal["active", "historical", "superseded"]


class ScriptEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Identifier
    path: Annotated[
        str,
        StringConstraints(
            pattern=r"^scripts/(pipeline|tools|adhoc)/([a-z][a-z0-9_]*/)*[a-z][a-z0-9_]*\.py$"
        ),
    ]
    summary: Text
    tags: Annotated[list[Identifier], Field(min_length=1)]
    status: Status = "active"
    commands: list[Text]
    inputs: list[Text]
    outputs: list[Text]
    notes: list[Text] = Field(default_factory=list)
    docs: list[Text] = Field(default_factory=list)
    superseded_by: Identifier | None = None

    @model_validator(mode="after")
    def validate_status(self):
        if self.status == "active" and not self.commands:
            raise ValueError("Active scripts need at least one usage command")
        if (self.status == "superseded") != (self.superseded_by is not None):
            raise ValueError("Only superseded scripts must name a replacement")
        if len(self.tags) != len(set(self.tags)):
            raise ValueError("Tags must be unique")
        return self


class Catalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    scripts: list[ScriptEntry]


class CatalogCLI(NonExitingCLISettings):
    action: CliPositionalArg[Literal["search", "show", "check"]] = Field(
        description="search, show, or check"
    )
    query: str = Field(default="", description="Search terms; every term must match")
    id: str | None = Field(default=None, description="Exact script ID for show")
    role: Role | None = Field(default=None, description="Filter search by folder role")
    status: Status | None = Field(
        default=None, description="Filter search by lifecycle status"
    )
    limit: int = Field(default=10, ge=1, le=100, description="Maximum search results")
    offset: int = Field(default=0, ge=0, description="Skip this many search results")


def load_entries() -> tuple[dict[str, tuple[ScriptEntry, Path]], list[Path]]:
    files = sorted((ROOT / "scripts").rglob("catalog.yaml"))
    if not files:
        raise ValueError("No script catalogs found under scripts/")
    entries = {}
    paths = set()
    for file in files:
        try:
            catalog = Catalog.model_validate(
                yaml.safe_load(file.read_text(encoding="utf-8"))
            )
        except (ValueError, yaml.YAMLError) as error:
            raise ValueError(f"{file.relative_to(ROOT)}: {error}") from error
        for entry in catalog.scripts:
            if entry.id in entries:
                raise ValueError(f"Duplicate script ID: {entry.id}")
            if entry.path in paths:
                raise ValueError(f"Duplicate script path: {entry.path}")
            if not (ROOT / entry.path).is_relative_to(file.parent):
                raise ValueError(
                    f"{entry.id}: entry belongs in a catalog above its script"
                )
            entries[entry.id] = entry, file
            paths.add(entry.path)
    return entries, files


def search(entries: dict[str, tuple[ScriptEntry, Path]], args: CatalogCLI) -> dict:
    terms = args.query.casefold().split()
    matches = []
    for entry, file in entries.values():
        role = PurePosixPath(entry.path).parts[1]
        if args.role is not None and role != args.role:
            continue
        if args.status is not None and entry.status != args.status:
            continue
        searchable = " ".join(
            [
                entry.id,
                entry.path,
                entry.summary,
                *entry.tags,
                *entry.inputs,
                *entry.outputs,
                *entry.notes,
            ]
        ).casefold()
        if all(term in searchable for term in terms):
            score = sum(
                4 * (term in entry.id)
                + 2 * (term in entry.tags)
                + (term in entry.summary.casefold())
                for term in terms
            )
            matches.append((score, entry, file, role))
    matches.sort(key=lambda item: (-item[0], item[1].id))
    end = args.offset + args.limit
    return {
        "total": len(matches),
        "offset": args.offset,
        "next_offset": end if end < len(matches) else None,
        "results": [
            {
                "id": entry.id,
                "path": entry.path,
                "role": role,
                "status": entry.status,
                "summary": entry.summary,
                "catalog": str(file.relative_to(ROOT)),
            }
            for _, entry, file, role in matches[args.offset : end]
        ],
    }


def check(entries: dict[str, tuple[ScriptEntry, Path]], files: list[Path]) -> dict:
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "scripts").rglob("*.py")
        if path.name != "__init__.py" and "__pycache__" not in path.parts
    }
    indexed = {entry.path for entry, _ in entries.values()}
    errors = [f"Unindexed script: {path}" for path in sorted(actual - indexed)]
    errors.extend(f"Missing script: {path}" for path in sorted(indexed - actual))
    for entry, _ in entries.values():
        if not (ROOT / entry.path).resolve().is_relative_to(ROOT):
            errors.append(f"{entry.id}: script resolves outside the repository")
        for link in entry.docs:
            target = (ROOT / link.split("#", 1)[0]).resolve()
            if not target.is_relative_to(ROOT) or not target.is_file():
                errors.append(f"{entry.id}: missing or non-local documentation: {link}")
        if entry.superseded_by is not None:
            replacement = entries.get(entry.superseded_by)
            if replacement is None or replacement[0].status != "active":
                errors.append(
                    f"{entry.id}: replacement must name an active indexed script"
                )
    return {"catalogs": len(files), "scripts": len(entries), "errors": errors}


def main() -> None:
    try:
        args = CatalogCLI()
        entries, files = load_entries()
        if args.action == "search":
            result = search(entries, args)
        elif args.action == "show":
            if args.id not in entries:
                raise ValueError("show requires --id with an existing script ID")
            entry, file = entries[args.id]
            result = {**entry.model_dump(), "catalog": str(file.relative_to(ROOT))}
        else:
            result = check(entries, files)
        print(json.dumps(result, indent=2))
        if args.action == "check" and result["errors"]:
            raise SystemExit(1)
    except (OSError, ValueError, yaml.YAMLError, SettingsError) as error:
        print(f"Catalog error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
