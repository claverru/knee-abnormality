import json
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from knee_abnormality.autoresearch.runtime.isolation import (
    SOURCE_FILES,
    sandbox,
    scrubbed_environment,
)
from knee_abnormality.autoresearch.runtime.storage import read_json, tail, write_json
from knee_abnormality.autoresearch.runtime.usage import capture_usage, events

LIMIT = re.compile(
    r"session limit|usage limit|weekly limit|5-hour limit|hit your limit|rate limit"
    r"|too many requests|\b429\b",
    re.IGNORECASE,
)
AUTHENTICATION = re.compile(
    r"authenticat|unauthori[sz]ed|oauth|/login|\blog ?in\b|\bsign ?in\b|refresh token"
    r"|access token|token (?:has )?expired|credential|api key|forbidden|\b40[13]\b",
    re.IGNORECASE,
)
TRANSIENT = re.compile(
    r"overloaded|high demand|at capacity|server error|service unavailable|bad gateway"
    r"|gateway time-?out|\b(?:api error|status):?\s*5\d\d\b|connection|network"
    r"|timed out|timeout|econn\w*|etimedout|enotfound|eai_again|socket hang up"
    r"|fetch failed|stream disconnected|temporarily unavailable",
    re.IGNORECASE,
)
RESET_AT = re.compile(
    r"\b(?:resets?|try again)\s+(?:at\s+|on\s+)?"
    r"(?:(?P<weekday>mon|tue|wed|thu|fri|sat|sun)[a-z]*,?\s+)?"
    r"(?:(?P<month>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?,?\s+(?:(?P<year>\d{4}),?\s+)?(?:at\s+)?)?"
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<meridiem>[ap]\.?m\b)?",
    re.IGNORECASE,
)
RESET_IN = re.compile(
    r"\b(?:resets?|try again)\s+in\s+(?P<span>(?:\d+\s*[a-z]+[\s,]*(?:and\s+)?)+)",
    re.IGNORECASE,
)
ZONE = re.compile(r"\((?P<zone>[A-Za-z_]+(?:/[A-Za-z0-9_+-]+)+|UTC|GMT)\)")
EPOCH = re.compile(r"\|(?P<epoch>\d{10})\b")
MONTHS = (
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
)
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
UNITS = {"d": 86400, "h": 3600, "m": 60, "s": 1}
LIMIT_FALLBACK_SECONDS = 3600
CODEX_SLUG = re.compile(r"gpt-(?P<version>\d+(?:\.\d+)*)-(?P<family>[a-z]+)")


def codex_model(family):
    return max(
        (tuple(int(part) for part in match["version"].split(".")), model["slug"])
        for model in read_json(Path.home() / ".codex/models_cache.json")["models"]
        if model["visibility"] == "list"
        and (match := CODEX_SLUG.fullmatch(model["slug"]))
        and match["family"] == family
    )[1]


def resolved(provider, profile):
    if provider == "codex":
        return {**profile, "model": codex_model(profile["model"])}
    return dict(profile)


def codex_command(profile, schema: Path, output: Path, sandbox_mode, *options):
    return [
        "codex",
        "exec",
        "--ignore-user-config",
        "--ephemeral",
        *options,
        "--sandbox",
        sandbox_mode,
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
        str(schema),
        "--output-last-message",
        str(output),
        "-",
    ]


def invocation(provider, model, profile, source: Path, job: Path, editable=None):
    schema = model.model_json_schema()
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
        command = codex_command(
            profile,
            job / "schema.json",
            scratch / "result.json",
            "danger-full-access",
            "--skip-git-repo-check",
        )
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
    if editable is not None:
        for name in editable:
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


def result(provider, model, job: Path):
    capture_usage(provider, job)
    if provider == "codex":
        value = read_json(job / "scratch/result.json")
    else:
        value = [
            event
            for event in events(job / "stdout.log")
            if event.get("type") == "result"
        ][-1]["structured_output"]
    report = model.model_validate(value).model_dump(mode="json")
    write_json(job / "report.json", report)
    capture_usage(provider, job)
    return report


def reset_time(text, now):
    if match := EPOCH.search(text):
        return max(float(match["epoch"]), now) + 60
    if match := RESET_IN.search(text):
        seconds = sum(
            int(count) * UNITS.get(unit[0].lower(), 0)
            for count, unit in re.findall(
                r"(\d+)\s*([a-z]+)", match["span"], re.IGNORECASE
            )
        )
        if seconds:
            return now + seconds + 60
    match = RESET_AT.search(text)
    if not match or not (match["minute"] or match["meridiem"]):
        return None
    try:
        zone = ZONE.search(text)
        timezone = (
            ZoneInfo(zone["zone"]) if zone else datetime.now().astimezone().tzinfo
        )
        current = datetime.fromtimestamp(now, timezone)
        hour = int(match["hour"])
        if match["meridiem"]:
            hour = hour % 12 + 12 * (match["meridiem"][0].lower() == "p")
        minute = int(match["minute"] or 0)
        if match["month"]:
            reset = datetime(
                int(match["year"] or current.year),
                MONTHS.index(match["month"][:3].lower()) + 1,
                int(match["day"]),
                hour,
                minute,
                tzinfo=timezone,
            )
            if not match["year"] and reset < current - timedelta(days=1):
                reset = reset.replace(year=reset.year + 1)
        else:
            reset = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if match["weekday"]:
                reset += timedelta(
                    days=(
                        WEEKDAYS.index(match["weekday"][:3].lower()) - current.weekday()
                    )
                    % 7
                )
            if reset < current - timedelta(minutes=5):
                reset += timedelta(days=7 if match["weekday"] else 1)
    except (KeyError, ValueError, OSError):
        return None
    return max(reset.timestamp(), now) + 60


def classify(message, status=None):
    now = time.time()
    code = int(status) if str(status).isdigit() else None
    if code == 429 or LIMIT.search(message):
        return {
            "kind": "limit",
            "message": message,
            "retry_after": reset_time(message, now) or now + LIMIT_FALLBACK_SECONDS,
        }
    if code in {401, 403} or AUTHENTICATION.search(message):
        kind = "authentication"
    elif (code or 0) >= 500 or TRANSIENT.search(message):
        kind = "transient"
    else:
        kind = "other"
    return {"kind": kind, "message": message}


def failure(provider, job: Path, receipt):
    if receipt.get("error"):
        return {"kind": "other", "message": str(receipt["error"])}
    failed = receipt["exit_code"] != 0
    status = None
    messages = []
    if provider == "codex":
        if not failed:
            return None
        messages = [
            str(event.get("message") or event.get("error"))
            for event in events(job / "stdout.log")
            if event.get("type") in {"error", "turn.failed"}
        ][-1:]
    else:
        results = [
            event
            for event in events(job / "stdout.log")
            if event.get("type") == "result"
        ]
        envelope = results[-1] if results else {}
        if envelope and not failed and not envelope.get("is_error"):
            return None
        if envelope.get("is_error"):
            status = envelope.get("api_error_status")
            messages = [
                str(value)
                for value in (envelope.get("result"), envelope.get("errors"))
                if value
            ]
    message = (
        " ".join(messages).strip()
        or tail(job / "stderr.log", 2000).strip()
        or f"Exit code {receipt['exit_code']} without provider output"
    )
    return classify(message, status)


def preflight(provider: str, root: Path):
    for executable in (provider, "bwrap", "nvidia-smi", "rtk"):
        if shutil.which(executable) is None:
            raise ValueError(f"Required command is unavailable: {executable}")
    command = (
        ["codex", "login", "status"]
        if provider == "codex"
        else ["claude", "auth", "status"]
    )
    authentication_environment = scrubbed_environment()
    if provider == "claude":
        authentication_environment["CLAUDE_CONFIG_DIR"] = str(Path.home() / ".claude")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30,
        env=authentication_environment,
        check=False,
    )
    if result.returncode:
        raise ValueError(
            f"{provider} subscription login is unavailable; log in with its CLI first"
        )
    if provider == "codex":
        if "using ChatGPT" not in result.stdout + result.stderr:
            raise ValueError("Codex must use ChatGPT subscription authentication")
    else:
        info = json.loads(result.stdout)
        if (
            not info.get("loggedIn")
            or info.get("authMethod") != "claude.ai"
            or info.get("apiProvider") != "firstParty"
        ):
            raise ValueError("Claude must use its first-party subscription login")
    result = subprocess.run(
        sandbox(["true"], root, []),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode:
        raise ValueError(
            f"Required Linux isolation is unavailable: {result.stderr.strip()}"
        )
