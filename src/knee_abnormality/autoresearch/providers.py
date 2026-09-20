import json
import os
from pathlib import Path

from knee_abnormality.autoresearch.campaigns import CampaignPolicy
from knee_abnormality.autoresearch.isolation import SOURCE_FILES, sandbox
from knee_abnormality.autoresearch.models import REPORTS
from knee_abnormality.autoresearch.storage import read_json, write_json
from knee_abnormality.autoresearch.usage import capture_usage, events


def invocation(
    provider, role, profile, source: Path, job: Path, policy: CampaignPolicy
):
    schema = REPORTS[role].model_json_schema()
    write_json(job / "schema.json", schema)
    scratch = job / "scratch"
    scratch.mkdir()
    for directory in (scratch / "data", scratch / "cache"):
        directory.mkdir()
    writable = [scratch]
    home = Path.home()
    if provider == "codex":
        state = home / ".codex"
        writable.append(state)
        command = [
            "codex",
            "exec",
            "--ignore-user-config",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "danger-full-access",
            "-c",
            'approval_policy="never"',
            "-c",
            "features.multi_agent=false",
            "-m",
            profile["model"],
            "-c",
            f'model_reasoning_effort="{profile["effort"]}"',
            "--json",
            "--output-schema",
            str(job / "schema.json"),
            "--output-last-message",
            str(scratch / "result.json"),
            "-",
        ]
    else:
        writable.append(home / ".claude")
        command = [
            "env",
            f"CLAUDE_CONFIG_DIR={home / '.claude'}",
            "claude",
            "-p",
            "--safe-mode",
            "--no-session-persistence",
            "--model",
            profile["model"],
            *(["--effort", profile["effort"]] if profile["effort"] is not None else []),
            "--output-format",
            "stream-json",
            "--verbose",
            "--json-schema",
            json.dumps(schema),
            "--permission-mode",
            "acceptEdits",
            "--permission-prompts",
            "none",
            "--tools",
            "Read,Glob,Grep,Bash,Edit,Write,WebSearch,WebFetch",
            "--allowedTools",
            "Read,Glob,Grep,Bash,Edit,Write,WebSearch,WebFetch",
            "--strict-mcp-config",
        ]
    readonly = []
    if role == "developer":
        for name in policy.editable_directories:
            (source / name).mkdir(parents=True, exist_ok=True)
        writable.append(source)
        readonly = [
            source / path
            for path in (
                "docs",
                "autoresearch",
                "scripts",
                "labels",
                "src/knee_abnormality/autoresearch",
                *SOURCE_FILES,
            )
        ]
    command = [
        "env",
        f"XDG_DATA_HOME={scratch / 'data'}",
        f"XDG_CACHE_HOME={scratch / 'cache'}",
        *command,
    ]
    return sandbox(command, source, writable, readonly=readonly, network=True)


def result(provider, role, job: Path):
    capture_usage(provider, job)
    if provider == "codex":
        value = read_json(job / "scratch/result.json")
    else:
        results = [
            event
            for event in events(job / "stdout.log")
            if event.get("type") == "result"
        ]
        if not results:
            raise ValueError(
                "Claude did not return a final result; inspect stdout.log and usage.json"
            )
        envelope = results[-1]
        if envelope.get("is_error"):
            raise ValueError(str(envelope.get("result", envelope.get("errors"))))
        value = envelope["structured_output"]
    report = REPORTS[role].model_validate(value).model_dump(mode="json")
    write_json(job / "report.json", report)
    capture_usage(provider, job)
    return report


def auth_environment():
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("OPENAI_", "ANTHROPIC_", "CODEX_", "CLAUDE_"))
        and key != "CLAUDECODE"
    }
