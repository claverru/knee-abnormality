import html
import json
import re
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address, ip_network
from pathlib import Path
from urllib.parse import parse_qs, quote

import plotly
import plotly.graph_objects as go
import yaml
from plotly.offline import get_plotlyjs
from pydantic import BaseModel, Field, ValidationError

from knee_abnormality.autoresearch.campaign.rules import (
    CALIBRATION_EPOCHS,
    SEED_TERMINAL,
)
from knee_abnormality.autoresearch.runtime.layout import (
    PREDICTIONS,
    campaign_fold,
    fold_dir,
    job_experiment,
)
from knee_abnormality.autoresearch.runtime.storage import alive
from knee_abnormality.cli import CLISettings, describe

CONTROL = "control"
WATCHDOG = "watchdog"
MAXIMUM_MESSAGE_BYTES = 20000
MINIMUM_AGE_SECONDS = 5
STATUS_TIMEOUT_SECONDS = 60
PAGE_ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError)
TAILNET = (ip_network("100.64.0.0/10"), ip_network("fd7a:115c:a1e0::/48"))
OUTPUTS = Path("outputs/autoresearch")
REGISTRY = OUTPUTS / "monitor/campaigns"
PLOTLY_JS = f"plotly-{plotly.__version__}.min.js"
GAINS: dict[tuple, list[float]] = {}
STATES: dict[Path, tuple[int, dict]] = {}
REGENERATION = threading.Lock()
STATUS = threading.Lock()
last_regeneration: dict[str, float] = {}
SUPERVISED = {"RUNNING", "CALIBRATING", "STOPPING"}
STATE_COLOR = {
    "RUNNING": "--good",
    "CALIBRATING": "--s1",
    "STOPPING": "--warn",
    "PAUSED": "--bad",
}
STATE_DETAIL = {
    "RUNNING": "working normally",
    "CALIBRATING": "measuring the budget on CPU",
    "STOPPED": "stopped; nothing runs until resumed",
    "STOPPING": "finishing the current run before stopping",
    "PAUSED": "paused; needs a repair",
}
STATE_ORDER = {"PAUSED": 0, "RUNNING": 1, "CALIBRATING": 1, "STOPPING": 2}
ROLES = {
    "training": "training",
    "developer": "being implemented",
    "orchestrator": "under review",
    "completion_reporter": "recording its result",
    "comparison": "scoring",
    "calibration": "calibrating",
    "researcher": "proposing ideas",
    "config": "checking its configuration",
}
VERDICTS = {
    "provisional_accept": "accepted",
    "reject": "rejected",
    "invalid": "invalid",
    "failed": "failed",
    "blocked": "blocked",
}
IDEA_FIELDS = (
    "opportunity",
    "mechanism",
    "precedents",
    "novelty",
    "coverage",
    "exploitation",
    "budget",
    "implementation",
    "open_choices",
    "priority_rationale",
)
LINE = {
    "control": {"width": 2},
    "incumbent": {"width": 1.5},
    "running": {"width": 1.5, "dash": "dash"},
    "latest": {"width": 1.5},
}
COLOR = {
    "incumbent": "--s1",
    "running": "--s2",
    "latest": "--s2",
    "other": "--other",
}
LIGHT = {
    "--ink": "#0b0b0b",
    "--ink2": "#52514e",
    "--card": "#f0efec",
    "--grid": "#e4e3df",
    "--other": "#a8a79f",
    "--s1": "#2a78d6",
    "--s2": "#eb6834",
    "--good": "#0ca30c",
    "--warn": "#eda100",
    "--bad": "#e34948",
}
PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}
STYLE = """
:root { color-scheme: light; --bg: #fcfcfb; --card: #f0efec; --ink: #0b0b0b; --ink2: #52514e; --grid: #e4e3df;
  --other: #a8a79f; --s1: #2a78d6; --s2: #eb6834; --good: #0ca30c; --warn: #eda100; --bad: #e34948;
  --up: #087a08; --down: #c4302f; --amber: #8a5d00; --tint: #e1ebf7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --bg: #1a1a19;
  --card: #262624; --ink: #ffffff; --ink2: #c3c2b7; --grid: #383835; --other: #75746e; --s1: #3987e5;
  --s2: #d95926; --good: #1baf7a; --warn: #c98500; --bad: #e66767; --up: #3cc68f; --down: #f07a7a;
  --amber: #e5ad3a; --tint: #23324a; } }
:root[data-theme="dark"] { color-scheme: dark; --bg: #1a1a19; --card: #262624; --ink: #ffffff; --ink2: #c3c2b7;
  --grid: #383835; --other: #75746e; --s1: #3987e5; --s2: #d95926; --good: #1baf7a; --warn: #c98500; --bad: #e66767;
  --up: #3cc68f; --down: #f07a7a; --amber: #e5ad3a; --tint: #23324a; }
* { box-sizing: border-box; }
body { margin: 0 auto; max-width: 1440px; padding: 12px 16px 40px; background: var(--bg); color: var(--ink);
  font: 14px/1.4 system-ui, sans-serif; -webkit-text-size-adjust: 100%; }
main > * + * { margin-top: 12px; }
a { color: var(--s1); }
:focus-visible { outline: 2px solid var(--s1); outline-offset: 2px; }
button { font: inherit; color: var(--ink); background: var(--card); border: 1px solid var(--grid); border-radius: 8px;
  padding: 8px 14px; min-height: 40px; cursor: pointer; }
button.primary { background: var(--s1); border-color: var(--s1); color: #fff; font-weight: 600; }
button:disabled { opacity: .6; cursor: default; }
nav { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; }
nav .strip { display: flex; gap: 6px; flex: 1; min-width: 0; overflow-x: auto; scrollbar-width: none; }
nav .strip::-webkit-scrollbar { display: none; }
nav a { flex: none; display: inline-flex; align-items: center; gap: 6px; min-height: 40px;
  padding: 6px 12px; border-radius: 8px; border: 0; background: var(--card); color: var(--ink);
  text-decoration: none; white-space: nowrap; }
nav a.home { font-weight: 600; }
nav a[aria-current] { box-shadow: inset 0 0 0 2px var(--s1); font-weight: 600; }
.dot { display: inline-block; flex: none; width: 9px; height: 9px; border-radius: 50%; background: var(--c); }
h1 { font-size: 20px; margin: 0 0 8px; overflow-wrap: anywhere; }
h2 { font-size: 13px; color: var(--ink2); text-transform: uppercase; letter-spacing: .04em; margin: 26px 0 8px; }
h2 .sub { text-transform: none; letter-spacing: 0; font-weight: 400; }
.sub { color: var(--ink2); font-size: 13px; }
.card { background: var(--card); border-radius: 8px; padding: 10px 12px; min-width: 0; }
.banner { background: var(--card); border-left: 6px solid var(--c); border-radius: 8px; padding: 10px 12px; }
.banner .state { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 8px; font-size: 15px; }
.banner .detail { margin-top: 2px; color: var(--ink2); }
.banner.alert .detail { color: var(--down); font-weight: 600; }
#age { font-size: 12px; margin-top: 6px; color: var(--ink2); }
#age.stale { color: var(--down); font-weight: 600; }
.up { color: var(--up); font-weight: 600; } .down { color: var(--down); font-weight: 600; }
.waiting { color: var(--amber); font-weight: 600; }
.split { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr); gap: 12px; }
.tiles { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.tile .name { font-size: 15px; font-weight: 600; overflow-wrap: anywhere; }
.fact { margin-top: 3px; font-size: 13px; overflow-wrap: anywhere; font-variant-numeric: tabular-nums; }
.fact .key { color: var(--ink2); }
.label { color: var(--ink2); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; margin-bottom: 4px; }
.job { padding: 8px 0; } .job + .job { border-top: 1px solid var(--grid); }
.job .head { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 2px 12px; }
.job progress { display: block; width: 100%; height: 8px; margin: 6px 0 4px; accent-color: var(--s2); }
.job .numbers { font-variant-numeric: tabular-nums; }
ol, ul { margin: 4px 0 2px 20px; padding: 0; } li { margin: 4px 0; }
.table { overflow-x: auto; padding: 2px 0; }
table { border-collapse: separate; border-spacing: 0; width: 100%; font-size: 13px; }
th, td { text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--grid); white-space: nowrap;
  font-variant-numeric: tabular-nums; }
th { color: var(--ink2); font-weight: 600; vertical-align: bottom; }
th.text, td.text { text-align: left; }
th:first-child, td:first-child { padding-left: 12px; } th:last-child, td:last-child { padding-right: 12px; }
th.name, td.name { text-align: left; position: sticky; left: 0; z-index: 1; background: var(--card); }
tr:last-child td { border-bottom: 0; }
tr.incumbent td { background: var(--tint); }
tr.incumbent td.name::before { content: "★ "; color: var(--s1); }
button.idea { font: inherit; color: var(--s1); background: none; border: 0; border-radius: 0; padding: 2px 0;
  min-height: 0; cursor: pointer; text-align: left; text-decoration: underline dotted; text-underline-offset: 3px; }
.thread { max-height: 320px; overflow-y: auto; }
.message { margin: 6px 0; padding-left: 8px; border-left: 3px solid var(--s1); white-space: pre-wrap;
  overflow-wrap: anywhere; }
form.send { display: flex; gap: 8px; margin-top: 8px; align-items: flex-end; }
form.send textarea { flex: 1; min-width: 0; min-height: 64px; font: inherit; font-size: 16px; color: var(--ink);
  background: var(--bg); border: 1px solid var(--grid); border-radius: 8px; padding: 8px; resize: vertical; }
.charts { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.pair { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; }
.chart { overflow: hidden; }
.pair .chart { display: flex; flex-direction: column; }
.pair .chart > div:last-child { margin-top: auto; }
.chart .caption { font-weight: 600; font-size: 13px; }
.plotly-graph-div { width: 100% !important; }
a.campaign { display: block; color: var(--ink); text-decoration: none; }
a.campaign .head { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 10px; }
a.campaign .name { font-size: 16px; font-weight: 600; overflow-wrap: anywhere; }
.badge { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600; }
details.status { margin-top: 26px; }
footer { margin-top: 26px; }
details.status summary { cursor: pointer; padding: 8px 0; color: var(--ink2); }
pre { font-size: 11px; line-height: 1.35; white-space: pre-wrap; overflow-wrap: anywhere; background: var(--card);
  padding: 8px; border-radius: 8px; margin: 0; }
dialog.idea { width: min(760px, calc(100vw - 32px)); max-height: calc(100dvh - 48px); background: var(--bg);
  color: var(--ink); border: 1px solid var(--grid); border-radius: 8px; padding: 0 16px 14px; overflow-x: hidden;
  overflow-y: auto; overflow-wrap: anywhere; }
dialog.idea::backdrop { background: rgba(0, 0, 0, .45); }
dialog.idea .head { position: sticky; top: 0; display: flex; gap: 12px; align-items: center;
  justify-content: space-between; background: var(--bg); padding: 12px 0 8px; border-bottom: 1px solid var(--grid); }
dialog.idea h3 { font-size: 15px; margin: 0; }
dialog.idea dl { margin: 0; }
dialog.idea dt { color: var(--ink2); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; margin-top: 12px; }
dialog.idea dd { margin: 2px 0 0; white-space: pre-wrap; line-height: 1.45; }
dialog.idea form { flex: none; margin: 0; }
@media (max-width: 1100px) { .charts, .tiles { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); } }
@media (max-width: 900px) { .split, .pair { grid-template-columns: minmax(0, 1fr); }
  .chart.wide { overflow-x: auto; } .chart.wide .plotly-graph-div { min-width: 720px; } }
@media (max-width: 720px) { .charts { grid-template-columns: minmax(0, 1fr); }
  table.cards, table.cards tbody { display: block; }
  table.cards thead { display: none; }
  table.cards tr { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); grid-auto-flow: row dense;
    gap: 4px 8px; padding: 8px 12px; border-bottom: 1px solid var(--grid); font-size: 12px; }
  table.cards tr:last-child { border-bottom: 0; }
  table.cards td { display: block; min-width: 0; padding: 0; border: 0; text-align: left; white-space: normal;
    position: static; background: none; }
  table.cards td.none { display: none; }
  table.cards td.name { grid-column: span 4; font-size: 13px; font-weight: 600; overflow-wrap: anywhere; }
  table.cards td.wide { grid-column: span 2; }
  table.cards td.verdict { text-align: right; }
  table.cards td:not(.name):not(.verdict)::before { content: attr(data-label); display: block; font-size: 10.5px;
    color: var(--ink2); white-space: nowrap; }
  table.cards tr.incumbent { background: var(--tint); } }
@media (max-width: 560px) { .tiles { grid-template-columns: minmax(0, 1fr); } }
"""
THEME_BOOT = """
try { const theme = localStorage.getItem("theme"); if (theme) document.documentElement.dataset.theme = theme; } catch {}
"""
SCRIPT = """
const root = document.documentElement;
const css = name => getComputedStyle(root).getPropertyValue(name).trim();
const age = document.getElementById("age");
const interval = Number(age.dataset.interval) * 1000;
const updated = Date.now();
const draft = document.getElementById("draft");
const themeButton = document.getElementById("theme");
const themes = ["auto", "light", "dark"];
let lastTouch = 0;
let unreachable = false;
const touched = () => { lastTouch = Date.now(); };
function busy() {
  return Date.now() - lastTouch < interval
    || Boolean(document.querySelector("dialog[open], details[open]"))
    || Boolean(draft && draft.value.trim())
    || Boolean(document.activeElement && document.activeElement.matches("textarea, input, select"))
    || Boolean(window.visualViewport && visualViewport.scale > 1.05)
    || String(getSelection()).length > 0;
}
function describeAge() {
  const elapsed = Date.now() - updated;
  const minutes = Math.floor(elapsed / 60000);
  const when = minutes < 1 ? "updated just now" : "updated " + minutes + " min ago";
  const every = " · refreshes every " + Math.round(interval / 60000) + " min";
  const note = unreachable ? " · monitor unreachable, retrying"
    : elapsed > interval && busy() ? " · refresh paused while you zoom, type, select or read" : every;
  age.textContent = when + note;
  age.className = unreachable ? "stale" : "";
}
async function refresh() {
  describeAge();
  if (document.hidden || Date.now() - updated < interval || busy()) return;
  try {
    const response = await fetch(location.href, {method: "HEAD", cache: "no-store"});
    if (response.ok) return location.reload();
  } catch {}
  unreachable = true;
  describeAge();
}
function showTheme() { themeButton.textContent = "Theme: " + (root.dataset.theme || "auto"); }
themeButton.addEventListener("click", () => {
  const next = themes[(themes.indexOf(root.dataset.theme || "auto") + 1) % themes.length];
  if (next === "auto") delete root.dataset.theme; else root.dataset.theme = next;
  try { if (next === "auto") localStorage.removeItem("theme"); else localStorage.setItem("theme", next); } catch {}
  showTheme();
  window.dispatchEvent(new Event("themechange"));
});
["pointerdown", "keydown", "wheel"].forEach(name => document.addEventListener(name, touched, {passive: true}));
document.addEventListener("visibilitychange", refresh);
setInterval(refresh, 10000);
showTheme();
describeAge();
"""
PLOT_SCRIPT = """
const plots = () => [...document.querySelectorAll(".js-plotly-plot")];
const narrow = matchMedia("(max-width: 640px)");
const stacked = matchMedia("(max-width: 900px)");
const coarse = matchMedia("(pointer: coarse)");
function paint() {
  return Promise.all(plots().map(gd => {
    const lines = [], colors = [], heat = [];
    gd.data.forEach((trace, index) => {
      if (trace.type === "heatmap") heat.push(index);
      else if (trace.meta) { lines.push(index); colors.push(css(trace.meta)); }
    });
    const jobs = [];
    if (lines.length) jobs.push(Plotly.restyle(gd, {"line.color": colors, "marker.color": colors, "error_x.color": colors}, lines));
    if (heat.length) jobs.push(Plotly.restyle(gd, {
      colorscale: [[[0, css("--bad")], [0.5, css("--card")], [1, css("--s1")]]], "textfont.color": css("--ink"),
    }, heat));
    jobs.push(Plotly.relayout(gd, {
      "font.color": css("--ink2"), "xaxis.gridcolor": css("--grid"), "yaxis.gridcolor": css("--grid"),
      "legend.font.color": css("--ink2"), "hoverlabel.bgcolor": css("--card"), "hoverlabel.font.color": css("--ink"),
    }));
    return Promise.all(jobs);
  }));
}
function fit() {
  return Promise.all(plots().map(gd => {
    const update = {dragmode: coarse.matches ? false : "zoom"};
    const names = gd.layout.meta;
    if (names) {
      update["yaxis.ticktext"] = narrow.matches ? names.map(name => name.length > 24 ? name.slice(0, 23) + "…" : name) : names;
      if (gd.data[0].type === "heatmap") update["yaxis.showticklabels"] = stacked.matches;
    }
    return Plotly.relayout(gd, update);
  }));
}
function listen() {
  plots().forEach(gd => {
    gd.on("plotly_relayout", touched);
    gd.on("plotly_legendclick", touched);
    gd.on("plotly_click", touched);
  });
}
window.addEventListener("load", () => {
  plots().forEach(gd => Plotly.Plots.resize(gd));
  Promise.all([paint(), fit()]).then(listen);
  const strip = document.querySelector("nav .strip");
  const current = strip.querySelector("[aria-current]").getBoundingClientRect();
  const frame = strip.getBoundingClientRect();
  strip.scrollLeft += current.left - frame.left - (frame.width - current.width) / 2;
  const thread = document.getElementById("thread");
  thread.scrollTop = thread.scrollHeight;
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", paint);
window.addEventListener("themechange", paint);
[narrow, stacked, coarse].forEach(query => query.addEventListener("change", fit));
document.addEventListener("click", event => {
  const link = event.target.closest("button.idea");
  if (link) return document.getElementById("idea-" + link.dataset.idea).showModal();
  if (event.target.matches("dialog")) {
    const box = event.target.getBoundingClientRect();
    const inside = event.clientX >= box.left && event.clientX <= box.right && event.clientY >= box.top && event.clientY <= box.bottom;
    if (!inside) event.target.close();
  }
});
const form = document.querySelector("form.send");
form.addEventListener("submit", () => { form.querySelector("button").disabled = true; });
draft.addEventListener("keydown", event => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && draft.value.trim()) form.requestSubmit();
});
const statusBox = document.querySelector("details.status");
statusBox.addEventListener("toggle", async () => {
  if (!statusBox.open) return;
  const pre = statusBox.querySelector("pre");
  pre.textContent = "loading…";
  try {
    const response = await fetch("status", {cache: "no-store"});
    pre.textContent = await response.text();
  } catch { pre.textContent = "could not load the status"; }
});
"""


class MonitorCLI(CLISettings):
    model_config = describe(
        "Serve one auto-refreshing site for every registered autoresearch campaign: a front page "
        "listing the live ones whatever their status, and per campaign its state, the active jobs with the "
        "training's progress against the incumbent, the decided runs against the control, per-epoch "
        "validation AUC and loss curves of the control, the incumbent and the current run, "
        "per-label deltas and its own message box, regenerated on each page request"
    )

    hosts: list[str] = Field(
        description="Addresses to bind, e.g. the Tailscale IPv4 and IPv6 and 127.0.0.1"
    )
    port: int = Field(default=8765, gt=0, lt=65536, description="Port")
    directory: Path = Field(
        default=Path("outputs/analysis/campaign-monitor"),
        description="Directory the page is served from",
    )
    interval: int = Field(
        default=120, ge=30, description="Seconds between automatic page reloads"
    )


class Message(BaseModel):
    time: datetime
    text: str


class Registration(BaseModel):
    campaign: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    root: Path


class Overview(BaseModel):
    status: str
    supervisor: bool
    provider: str
    detail: str
    color: str
    alert: bool


class Job(BaseModel):
    kind: str
    idea: str | None
    name: str
    started: float | None
    note: str | None = None
    epoch: int | None = None
    epochs: int | None = None
    auc: float | None = None
    delta: float | None = None
    preflight: bool = False


def campaign_dir(campaign: str) -> Path:
    return OUTPUTS / campaign


def registrations() -> dict[str, Registration]:
    entries = {}
    for path in sorted(REGISTRY.glob("*.json")):
        try:
            entry = Registration.model_validate_json(path.read_text())
        except (OSError, ValidationError):
            continue
        if (campaign_dir(entry.campaign) / "state.json").is_file() and (
            entry.root / "autoresearch/campaigns" / entry.campaign
        ).is_dir():
            entries[entry.campaign] = entry
    return entries


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def load_state(campaign: str) -> dict:
    path = campaign_dir(campaign) / "state.json"
    stamp = path.stat().st_mtime_ns
    cached = STATES.get(path)
    if cached is None or cached[0] != stamp:
        cached = stamp, json.loads(path.read_text())
        STATES[path] = cached
    return cached[1]


def messages_path(campaign: str) -> Path:
    return campaign_dir(campaign) / "messages.jsonl"


def read_messages(campaign: str) -> list[Message]:
    path = messages_path(campaign)
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text().splitlines():
        try:
            entries.append(Message.model_validate_json(line))
        except ValidationError:
            continue
    return entries


def delivered(campaign: str) -> int:
    seen = messages_path(campaign).with_suffix(".seen")
    try:
        return int(seen.read_text())
    except (OSError, ValueError):
        return 0


def append_message(campaign: str, text: str) -> None:
    message = Message(time=datetime.now().astimezone(), text=text)
    with messages_path(campaign).open("a") as file:
        file.write(message.model_dump_json() + "\n")


def label(name: str) -> str:
    return re.sub(r"-\d{8}T\d{6}Z-[0-9a-f]{8}$", "", name)


def overview(campaign: str, state: dict) -> Overview:
    status = str(state.get("status", "unknown")).upper()
    supervisor = alive(read_json(campaign_dir(campaign) / "supervisor.json"))
    stranded = status in SUPERVISED and not supervisor
    notes = [state.get("reason") or STATE_DETAIL.get(status, "")]
    if stranded:
        notes.insert(0, "supervisor is not running, so nothing progresses")
    if (campaign_dir(campaign) / "stop.json").exists() and status != "STOPPED":
        notes.append("stop requested")
    retry = state.get("retry_after") or 0
    if status == "RUNNING" and retry > time.time():
        notes.append(
            f"agents wait for the provider until {datetime.fromtimestamp(retry).astimezone():%H:%M}"
        )
    alert = stranded or status == "PAUSED"
    return Overview(
        status=status,
        supervisor=supervisor,
        provider=state.get("provider") or "no provider",
        detail=" · ".join(note for note in notes if note),
        color="--bad" if alert else STATE_COLOR.get(status, "--other"),
        alert=alert,
    )


def badge(view: Overview) -> str:
    return (
        f"<span class='badge' style='--c: var({view.color})'><span class='dot'></span>"
        f"{html.escape(view.status)} · supervisor {'alive' if view.supervisor else 'not running'}</span>"
    )


def epoch_metrics(fold: Path) -> dict[int, dict]:
    metrics = {}
    for path in sorted((fold / "epochs").glob("metrics_epoch_*.json")):
        values = read_json(path)
        if "macro_auc" in values:
            metrics[int(path.stem.rsplit("_", 1)[1])] = values
    return metrics


def x4(run: Path, fold: int) -> float | None:
    return read_json(fold_dir(run, fold) / "selected_checkpoint.json").get(
        "prediction_macro_auc"
    )


def run_record(name: str, run: Path, fold: int) -> dict:
    directory = fold_dir(run, fold)
    selected = read_json(directory / "selected_checkpoint.json")
    epoch = re.search(r"best-epoch=(\d+)", selected.get("path") or "")
    return {
        "name": name,
        "epochs": epoch_metrics(directory),
        "x4": selected.get("prediction_macro_auc"),
        "best_epoch": int(epoch[1]) if epoch else None,
        "seconds": read_json(directory / "timing.json").get("training_seconds"),
    }


def verdict_of(key: str, exp: dict, running: str | None, retained: set) -> tuple:
    verdict = exp.get("decision") or exp.get("eligible_decision")
    if not verdict and exp.get("stage") == "repair_requested":
        verdict = "failed"
    withdrawn = (
        bool(exp.get("decision"))
        and "provisional_accept" in (verdict, exp.get("eligible_decision"))
        and key not in retained
    )
    if withdrawn:
        verdict += (
            " (not retained)"
            if verdict == "provisional_accept"
            else " (acceptance withdrawn)"
        )
    return verdict or ("training" if key == running else "pending"), withdrawn


def collect(state: dict) -> tuple[dict, list[dict]]:
    fold = campaign_fold(state)
    control = run_record(CONTROL, Path(state["reference"]), fold)
    control.update(
        role=CONTROL, verdict="–", color="--ink", run=Path(state["reference"]).resolve()
    )
    incumbent, running = state["incumbent"]["id"], state.get("running")
    retained, link = set(), incumbent
    while link in state["experiments"] and link not in retained:
        retained.add(link)
        link = (state["experiments"][link].get("reference") or {}).get("id")
    runs, seen = [], {}
    for key, exp in state["experiments"].items():
        training = Path(exp["directory"]) / "training"
        if not training.is_dir():
            continue
        name = label(exp["idea"])
        seen[name] = seen.get(name, 0) + 1
        record = run_record(
            name if seen[name] == 1 else f"{name} #{seen[name]}", training, fold
        )
        verdict, withdrawn = verdict_of(key, exp, running, retained)
        evidence = exp.get("evidence") or {}
        scores = evidence.get("macro_auc")
        reference = exp.get("reference")
        record.update(
            idea=exp["idea"],
            run=training.resolve(),
            withdrawn=withdrawn,
            comparison=evidence.get("comparison"),
            level=scores["reference"] - control["x4"]
            if scores and control["x4"] is not None
            else None,
            verdict=verdict,
            role="incumbent"
            if key == incumbent
            else "running"
            if key == running
            else "other",
            delta=scores["candidate"] - scores["reference"] if scores else None,
            interval=(evidence.get("adoption_bootstrap") or {}).get(
                "delta_percentile_80"
            ),
            soft=(evidence.get("soft_macro_auc") or {}).get("delta"),
            per_label=evidence.get("per_label"),
            reference=Path(reference["run"]).resolve() if reference else None,
            reference_name=label(reference["id"]) if reference else None,
        )
        runs.append(record)
    trained = [record for record in runs if record["epochs"]]
    if trained and trained[-1]["role"] == "other":
        trained[-1]["role"] = "latest"
    for record in runs:
        record["color"] = COLOR[record["role"]]
    return control, runs


def active_jobs(campaign: str, state: dict) -> list[Job]:
    fold = campaign_fold(state)
    jobs = []
    if state.get("status") == "calibrating":
        ledger = read_json(campaign_dir(campaign) / "baseline/triplet_accounting.json")
        jobs.append(
            Job(
                kind="calibration",
                idea=None,
                name="control",
                started=None,
                epoch=len(ledger.get("train") or []),
                epochs=min(CALIBRATION_EPOCHS, state["limits"]["epochs"]),
            )
        )
    incumbent = epoch_metrics(fold_dir(Path(state["incumbent"]["run"]), fold))
    for job in state["jobs"].values():
        if job.get("processed"):
            continue
        directory = Path(job["path"])
        exp = state["experiments"].get(job_experiment(job)) or {}
        note = (
            "finishing"
            if (directory / "receipt.json").exists()
            else None
            if alive(read_json(directory / "owner.json"))
            else "starting or awaiting recovery"
        )
        entry = Job(
            kind=job["kind"],
            idea=exp.get("idea"),
            name=label(exp.get("idea") or job["kind"]),
            started=job.get("created_at"),
            note=note,
        )
        if exp and job["kind"] == "training":
            preflight = Path(exp["directory"]) / "preflight"
            report = preflight / "preflight.json"
            if report.exists():
                entry.started = max(entry.started or 0.0, report.stat().st_mtime)
            elif preflight.is_dir():
                entry.preflight = True
            run = fold_dir(Path(exp["directory"]) / "training", fold)
            ledger = read_json(run / "triplet_accounting.json")
            metrics = epoch_metrics(run)
            entry.epoch = len(ledger.get("train") or []) if ledger else None
            entry.epochs = (
                (exp.get("config") or {}).get("trainer", {}).get("max_epochs")
            )
            if metrics:
                last = max(metrics)
                entry.auc = metrics[last]["macro_auc"]
                if last in incumbent:
                    entry.delta = entry.auc - incumbent[last]["macro_auc"]
        jobs.append(entry)
    return jobs


def tally(state: dict) -> tuple[int, int]:
    decisions = [exp.get("decision") for exp in state["experiments"].values()]
    return sum(map(bool, decisions)), decisions.count("provisional_accept")


def outcome(exp: dict) -> str | None:
    if exp.get("decision"):
        return exp["decision"]
    if exp.get("stage") == "repair_requested":
        return "failed"
    return "blocked" if exp.get("stage") == "blocked" else None


def clock(timestamp: float) -> str:
    moment = datetime.fromtimestamp(timestamp).astimezone()
    today = datetime.now().astimezone().date()
    return f"{moment:%H:%M}" if moment.date() == today else f"{moment:%d %b %H:%M}"


def budget_text(state: dict) -> str:
    limits = state["limits"]
    if state["budget"]["mode"] == "sealed":
        return (
            f"≤ {minutes(limits['max_run_seconds'])} per run, {limits['epochs']} epochs"
        )
    return (
        f"{minutes(limits['min_training_seconds'])} – "
        f"{minutes(limits['max_training_seconds'])} training"
    )


def research_text(state: dict) -> str:
    seeds = state.get("seed_ideas") or []
    done = sum(
        (state["ideas"].get(idea) or {}).get("status") in SEED_TERMINAL
        for idea in seeds
    )
    refill = f"research refills below {state['refill_below']} queued"
    if not seeds:
        return refill
    if done < len(seeds):
        return f"forced seeds {done}/{len(seeds)} done · research waits for them"
    return f"forced seeds {done}/{len(seeds)} done · {refill}"


def best_x1(record: dict) -> float | None:
    return (
        max(m["macro_auc"] for m in record["epochs"].values())
        if record["epochs"]
        else None
    )


def signed(value) -> str:
    return "–" if value is None else f"{value:+.4f}"


def number(value, digits: int = 4) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def tone(value, text: str | None = None) -> str:
    text = signed(value) if text is None else text
    if not value:
        return text
    return f"<span class='{'up' if value > 0 else 'down'}'>{text}</span>"


def minutes(seconds: float) -> str:
    if seconds >= 86400:
        return f"{seconds // 86400:.0f} d {seconds % 86400 // 3600:.0f} h"
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    return f"{seconds // 3600:.0f} h {seconds % 3600 / 60:02.0f} min"


def kilo(value: int) -> str:
    return f"{value / 1e6:.1f}M" if value >= 1e6 else f"{value / 1e3:.0f}k"


def interval_text(interval) -> str:
    return "–" if not interval else f"[{interval[0]:+.4f}, {interval[1]:+.4f}]"


def layout(**extra) -> dict:
    axis = {
        "showgrid": True,
        "gridcolor": LIGHT["--grid"],
        "gridwidth": 1,
        "zeroline": False,
        "showline": False,
        "ticks": "",
        "automargin": True,
    }
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "family": "system-ui, sans-serif",
            "size": 11,
            "color": LIGHT["--ink2"],
        },
        "margin": {"l": 8, "r": 8, "t": 8, "b": 8},
        "height": 300,
        "hovermode": "closest",
        "hoverlabel": {"font": {"size": 12}},
        "legend": {"orientation": "h", "y": -0.22, "font": {"size": 11}},
        "xaxis": axis,
        "yaxis": {**axis, "tickformat": ".4f"},
        **extra,
    }


def figure_html(figure: go.Figure, caption: str, note: str = "", css: str = "") -> str:
    return (
        f"<div class='card chart {css}'><div class='caption'>{caption}</div>"
        + (f"<div class='sub'>{note}</div>" if note else "")
        + figure.to_html(full_html=False, include_plotlyjs=False, config=PLOT_CONFIG)
        + "</div>"
    )


def curve(title: str, series: list[tuple[dict, dict[int, float]]]) -> str:
    figure = go.Figure()
    for record, values in series:
        if not values:
            continue
        epochs = sorted(values)
        figure.add_trace(
            go.Scatter(
                x=epochs,
                y=[values[epoch] for epoch in epochs],
                name=record["name"],
                meta=record["color"],
                mode="lines+markers",
                line={**LINE[record["role"]], "color": LIGHT[record["color"]]},
                marker={"size": 5, "color": LIGHT[record["color"]]},
                hovertemplate=f"<b>{html.escape(record['name'])}</b><br>epoch %{{x}}<br>%{{y:.4f}}<extra></extra>",
            )
        )
    base = layout()
    figure.update_layout(
        base, xaxis={**base["xaxis"], "title": {"text": "epoch"}, "dtick": 1}
    )
    return figure_html(figure, title)


def per_label(run: dict) -> dict[str, dict]:
    entries = read_json(Path(run["per_label"])) if run["per_label"] else {}
    return {entry["target"]: entry for entry in entries.get("per_target") or []}


def local_input(path: str) -> str:
    if Path(path).is_file() or "/data/" not in path:
        return path
    return str(Path("data") / path.rsplit("/data/", 1)[1])


def control_gain(control: dict, run: Path, fold: int, runs: list[dict]) -> list | None:
    report = next((r["comparison"] for r in runs if r["comparison"]), None)
    files = [fold_dir(path, fold) / PREDICTIONS for path in (control["run"], run)]
    key = (*files, report)
    if run == control["run"] or not report or not all(f.is_file() for f in files):
        return None
    if key not in GAINS:
        import polars as pl

        try:
            gain = bootstrap_gain(Path(report), files, fold)
        except (*PAGE_ERRORS, pl.exceptions.PolarsError) as error:
            print(f"control gain for {run} failed: {error!r}", flush=True)
            return None
        if gain is None:
            return None
        GAINS[key] = gain
    return GAINS[key]


def bootstrap_gain(report: Path, files: list[Path], fold: int) -> list | None:
    import numpy as np
    import polars as pl

    from knee_abnormality.data import splits
    from knee_abnormality.evaluation import metrics

    report = read_json(report)
    inputs = {
        Path(item["path"]).name: local_input(item["path"])
        for item in report.get("inputs") or []
    }
    if "studies.csv" not in inputs:
        return None
    frame = (
        pl.read_csv(inputs["studies.csv"])
        .filter(pl.col("fold") == fold)
        .join(pl.read_csv(inputs["targets.csv"]), on=splits.STUDY, how="inner")
        .sort(splits.STUDY)
    )
    targets = report["targets"]
    hard = (frame.select(targets).to_numpy() >= splits.TARGET_THRESHOLD).astype(int)
    scores = [pl.read_csv(file).sort(splits.STUDY) for file in files]
    if frame.height != report["studies"] or any(
        score[splits.STUDY].to_list() != frame[splits.STUDY].to_list()
        for score in scores
    ):
        return None
    bootstrap = report["paired_report_group_bootstrap"]
    groups, inverse = np.unique(frame["report_group"].to_numpy(), return_inverse=True)
    counts = np.random.default_rng(bootstrap["seed"]).multinomial(
        len(groups), np.full(len(groups), 1 / len(groups)), size=bootstrap["draws"]
    )[:, inverse]
    positive = counts @ hard
    weights = counts[
        ((positive > 0) & (positive < counts.sum(axis=1, keepdims=True))).all(axis=1)
    ]
    if not len(weights):
        return None
    reference, candidate = (
        metrics.bootstrap_auc(
            hard, score.select(targets).to_numpy().astype(float), weights
        )
        for score in scores
    )
    return np.percentile(candidate - reference, [10, 90]).tolist()


def segments(
    levels: list[float], drawn: set[int], offset: float = 0, joined: bool = True
) -> dict:
    xs, ys = [], []
    for row, level in enumerate(levels):
        if row in drawn:
            xs += [level + offset, level + offset, None]
            ys += [row - 0.5, row + 0.5, None]
        if joined and row + 1 < len(levels):
            xs += [level + offset, levels[row + 1] + offset, None]
            ys += [row + 0.5, row + 0.5, None]
    return {"x": xs, "y": ys}


def rows_layout(names: list[str], **extra) -> dict:
    base = layout()
    return layout(
        height=max(240, 120 + 26 * len(names)),
        margin={"l": 8, "r": 8, "t": 96, "b": 16},
        meta=names,
        yaxis={
            **base["yaxis"],
            "tickformat": None,
            "showgrid": False,
            "tickmode": "array",
            "tickvals": list(range(len(names))),
            "ticktext": names,
            "range": [len(names) - 0.5, -0.5],
        },
        **extra,
    )


def comparison(control: dict, runs: list[dict], fold: int) -> str:
    scored = [
        run
        for run in runs
        if run["delta"] is not None and run["interval"] and run["level"] is not None
    ]
    if not scored:
        return "<div class='card sub'>No scored run yet.</div>"
    names = [
        ("★ " if run["role"] == "incumbent" else "") + run["name"] for run in scored
    ]
    cells = [per_label(run) for run in scored]
    baseline = next(
        (
            {target: entry["reference"] for target, entry in row.items()}
            for run, row in zip(scored, cells)
            if row and run["reference"] == control["run"]
        ),
        {},
    )
    targets = list(baseline)
    base = layout()
    campaign = {control["run"], *(run["run"] for run in runs)}
    levels = [run["level"] for run in scored]
    manual = {row for row, run in enumerate(scored) if run["reference"] not in campaign}
    everything = set(range(len(scored)))
    gains = {
        reference: control_gain(control, reference, fold, runs)
        for reference in {run["reference"] for run in scored}
    }
    intervals = go.Figure()
    for spans, style in (
        (segments(levels, everything - manual), {"width": 1}),
        (segments(levels, manual, joined=False), {"width": 1, "dash": "dash"}),
    ):
        intervals.add_trace(
            go.Scatter(
                **spans,
                meta="--ink2",
                mode="lines",
                line={**style, "color": LIGHT["--ink2"]},
                showlegend=False,
                hoverinfo="skip",
            )
        )
    for row, run in enumerate(scored):
        low, high = run["interval"]
        position = run["level"] + run["delta"]
        gain = gains[run["reference"]]
        reference = (
            CONTROL
            if run["reference"] == control["run"]
            else f"{run['reference_name']} {run['level']:+.4f} vs control"
            + (f", 80% [{gain[0]:+.4f}, {gain[1]:+.4f}]" if gain else "")
        )
        intervals.add_trace(
            go.Scatter(
                x=[position],
                y=[row],
                meta=run["color"],
                mode="markers",
                marker={
                    "size": 10,
                    "symbol": "diamond-open" if run["withdrawn"] else "diamond",
                    "color": LIGHT[run["color"]],
                },
                error_x={
                    "color": LIGHT[run["color"]],
                    "type": "data",
                    "symmetric": False,
                    "array": [high - run["delta"]],
                    "arrayminus": [run["delta"] - low],
                    "thickness": 2,
                    "width": 0,
                },
                showlegend=False,
                hovertemplate=(
                    f"<b>{html.escape(run['name'])}</b><br>{html.escape(run['verdict'])}<br>"
                    f"Δ vs control {position:+.4f}<br>"
                    f"Δ vs reference {run['delta']:+.4f}, 80% [{low:+.4f}, {high:+.4f}]<br>"
                    f"reference: {html.escape(reference)}<extra></extra>"
                ),
            )
        )
    top = {"side": "top", "automargin": False, "showgrid": True}
    intervals.update_layout(
        rows_layout(names, xaxis={**base["xaxis"], **top, "tickformat": "+.3f"})
    )
    boxes = figure_html(
        intervals,
        "×4 macro Δ vs control, 80% interval vs each run's reference",
        "Line: the incumbent each run was judged against, dashed if set by hand. "
        "A run is adopted when its interval lies above the line.",
    )
    if targets:
        heatmap = go.Figure(
            go.Heatmap(
                x=targets,
                y=list(range(len(scored))),
                z=[
                    [
                        (row[t]["candidate"] - baseline[t]) * 1000 if t in row else None
                        for t in targets
                    ]
                    for row in cells
                ],
                customdata=[
                    [
                        f"{html.escape(run['name'])}<br>{t}: {baseline[t]:.4f} → {row[t]['candidate']:.4f}"
                        if t in row
                        else ""
                        for t in targets
                    ]
                    for run, row in zip(scored, cells)
                ],
                texttemplate="%{z:+.1f}",
                textfont={"size": 10, "color": LIGHT["--ink"]},
                colorscale=[
                    [0, LIGHT["--bad"]],
                    [0.5, LIGHT["--card"]],
                    [1, LIGHT["--s1"]],
                ],
                zmin=-10,
                zmax=10,
                showscale=False,
                xgap=2,
                ygap=2,
                hovertemplate="<b>%{customdata}</b><br>Δ ×1000 %{z:+.1f}<extra></extra>",
            )
        )
        heatmap.update_layout(
            rows_layout(
                names,
                xaxis={**base["xaxis"], **top, "showgrid": False, "tickangle": -40},
            )
        )
        boxes += figure_html(
            heatmap,
            "Per-label ×4 Δ vs control ×1000",
            "Blue up, red down, full colour at ±10. Rows follow the interval chart.",
            "wide",
        )
    return f"<div class='pair'>{boxes}</div>"


def metric(runs: list[dict], key: str) -> list[tuple[dict, dict[int, float]]]:
    return [
        (run, {e: m[key] for e, m in run["epochs"].items() if key in m}) for run in runs
    ]


def curves(control: dict, runs: list[dict]) -> str:
    shown = [control, *(run for run in runs if run["role"] != "other")]
    return (
        "<div class='sub'>Thick line in the text colour: control. Blue: the incumbent. Orange: the run training now (dashed) or, "
        "until the next one logs an epoch, the last run that finished. "
        "Tap a legend entry to hide a run, double-tap to isolate it.</div>"
        "<div class='charts'>"
        + curve("×1 validation macro AUC", metric(shown, "macro_auc"))
        + curve(
            "×1 validation disease loss (soft-target BCE, lower is better)",
            metric(shown, "soft_target_bce"),
        )
        + curve("×1 soft macro AUC", metric(shown, "soft_macro_auc"))
        + "</div>"
    )


def fact(key: str, value: str) -> str:
    return f"<div class='fact'><span class='key'>{key}</span> {value}</div>"


def leader(state: dict, record: dict | None, value: float | None, base) -> str:
    if record is None or value is None:
        return "none yet"
    gain = f" ({tone(value - base)})" if base is not None else ""
    return f"{idea_link(state, record.get('idea'), record['name'])} {value:.4f}{gain}"


def tiles(control: dict, runs: list[dict], fold: int, state: dict) -> str:
    incumbent = next((run for run in runs if run["role"] == "incumbent"), None)
    if incumbent is None:
        adopted = fact("×4", f"{number(control['x4'])} · no run adopted yet")
    else:
        bounds = control_gain(control, incumbent["run"], fold, runs)
        gain = (
            incumbent["x4"] - control["x4"]
            if None not in (incumbent["x4"], control["x4"])
            else None
        )
        adopted = fact("×4", f"{number(incumbent['x4'])} · {tone(gain)} vs control") + (
            fact("80%", interval_text(bounds)) if bounds else ""
        )
    scored = [run for run in runs if run["x4"] is not None]
    trained = [run for run in runs if run["epochs"]]
    top4 = max(scored, key=lambda run: run["x4"], default=None)
    top1 = max(trained, key=best_x1, default=None)
    counts: dict[str, int] = {}
    last = None
    for exp in state["experiments"].values():
        if verdict := outcome(exp):
            counts[verdict] = counts.get(verdict, 0) + 1
            last = exp, verdict
    tally_text = " · ".join(
        f"{counts.get(key, 0)} {name}"
        for key, name in VERDICTS.items()
        if counts.get(key) or key == "provisional_accept"
    )
    timed = [run for run in runs if run["seconds"] is not None]
    starts = [
        job["created_at"] for job in state["jobs"].values() if job.get("created_at")
    ]
    items = [
        (
            "Incumbent",
            "<div class='name'>"
            + (
                idea_link(state, incumbent.get("idea"), incumbent["name"])
                if incumbent
                else CONTROL
            )
            + "</div>"
            + adopted,
        ),
        (
            "Best so far",
            fact("×4", leader(state, top4, top4 and top4["x4"], control["x4"]))
            + fact(
                "×1",
                leader(state, top1, top1 and best_x1(top1), best_x1(control)),
            ),
        ),
        (
            "Verdicts",
            fact("", tally_text)
            + (
                fact(
                    "last",
                    f"{idea_link(state, last[0]['idea'], label(last[0]['idea']))} "
                    f"<span class='{'up' if last[1] == 'provisional_accept' else 'down'}'>"
                    f"{html.escape(VERDICTS.get(last[1], last[1]))}</span>",
                )
                if last
                else ""
            ),
        ),
        (
            "Campaign",
            (
                fact(
                    "started",
                    f"{clock(min(starts))} · {minutes(time.time() - min(starts))} ago",
                )
                if starts
                else ""
            )
            + fact("budget", f"{html.escape(budget_text(state))} · fold {fold}")
            + (
                fact(
                    "last run",
                    f"{html.escape(timed[-1]['name'])} trained {minutes(timed[-1]['seconds'])}",
                )
                if timed
                else ""
            ),
        ),
    ]
    return (
        "<div class='tiles'>"
        + "".join(
            f"<div class='card tile'><div class='label'>{name}</div>{value}</div>"
            for name, value in items
        )
        + "</div>"
    )


def idea_link(state: dict, idea: str | None, text: str) -> str:
    if idea not in state["ideas"]:
        return html.escape(text)
    return f"<button class='idea' data-idea='{html.escape(idea)}'>{html.escape(text)}</button>"


def idea_field(value) -> str:
    if isinstance(value, list):
        return "<br>".join(f"• {html.escape(str(item))}" for item in value)
    return html.escape(str(value))


def idea_dialogs(state: dict, ideas: list[str]) -> str:
    dialogs = []
    for idea in dict.fromkeys(ideas):
        record = state["ideas"].get(idea)
        if not record:
            continue
        fields = "".join(
            f"<dt>{field.replace('_', ' ')}</dt><dd>{idea_field(record[field])}</dd>"
            for field in IDEA_FIELDS
            if record.get(field) not in (None, "", [])
        )
        dialogs.append(
            f"<dialog class='idea' id='idea-{html.escape(idea)}' aria-labelledby='title-{html.escape(idea)}'>"
            f"<div class='head'><div><h3 id='title-{html.escape(idea)}'>{html.escape(idea)}</h3>"
            f"<div class='sub'>{html.escape(record.get('status') or '')}"
            f"{' · forced seed' if record.get('seeded') else ''}</div></div>"
            "<form method='dialog'><button>Close</button></form></div>"
            f"<dl><dt>title</dt><dd>{html.escape(record.get('title') or '')}</dd>{fields}</dl>"
            "</dialog>"
        )
    return "".join(dialogs)


def table(
    columns: list[tuple[str, str, str]], rows: list[tuple[str, list[str]]]
) -> str:
    head = "".join(f"<th class='{css}'>{name}</th>" for name, _, css in columns)
    body = "".join(
        f"<tr class='{row}'>"
        + "".join(
            f"<td class='{css}{' none' if cell == '–' else ''}' data-label='{short}'>{cell}</td>"
            for (_, short, css), cell in zip(columns, cells)
        )
        + "</tr>"
        for row, cells in rows
    )
    return (
        f"<div class='card table'><table class='cards'><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


RESULT_COLUMNS = [
    ("run", "", "name"),
    ("verdict", "", "text verdict"),
    ("×4 macro", "×4", ""),
    ("Δ vs its reference", "Δ ref", ""),
    ("80% interval", "80% interval", "wide"),
    ("soft Δ", "soft Δ", ""),
    ("Δ ×4 vs control", "Δ control", ""),
    ("best ×1", "best ×1", ""),
    ("best epoch", "epoch", ""),
    ("train min", "train min", ""),
]


def results(control: dict, runs: list[dict], state: dict) -> str:
    rows = []
    for run in [control, *runs]:
        verdict = run["verdict"]
        css = (
            "down"
            if verdict.split()[0] in {"reject", "invalid", "failed"}
            else "up"
            if verdict.startswith("provisional_accept") and not run.get("withdrawn")
            else ""
        )
        versus = (
            run["x4"] - control["x4"]
            if run is not control
            and run["x4"] is not None
            and control["x4"] is not None
            else None
        )
        starred = run["role"] == "incumbent" or (
            run is control and not any(r["role"] == "incumbent" for r in runs)
        )
        rows.append(
            (
                "incumbent" if starred else "",
                [
                    idea_link(state, run.get("idea"), run["name"]),
                    f"<span class='{css}'>{html.escape(verdict.replace('_', ' '))}</span>"
                    if run is not control
                    else "–",
                    number(run["x4"]),
                    tone(run.get("delta")),
                    interval_text(run.get("interval")),
                    tone(run.get("soft")),
                    tone(versus),
                    number(best_x1(run)),
                    "–" if run["best_epoch"] is None else str(run["best_epoch"]),
                    "–" if run["seconds"] is None else f"{run['seconds'] / 60:.1f}",
                ],
            )
        )
    return table(RESULT_COLUMNS, rows)


def job_progress(job: Job) -> str:
    parts = []
    if job.epoch is not None and job.epochs:
        parts.append(f"epoch {job.epoch}/{job.epochs}")
        if 0 < job.epoch < job.epochs and job.started and job.kind == "training":
            left = (time.time() - job.started) / job.epoch * (job.epochs - job.epoch)
            parts.append(f"ETA {clock(time.time() + left)} (≈{minutes(left)})")
    if job.auc is not None:
        parts.append(f"×1 {job.auc:.4f}")
    if job.delta is not None:
        parts.append(f"{tone(job.delta)} vs incumbent at this epoch")
    return " · ".join(parts)


def job_stage(job: Job) -> str:
    stage = ROLES.get(job.kind, job.kind.replace("_", " "))
    if job.preflight:
        stage = "preflight (memory and time check before training)"
    elif job.kind == "training" and job.epoch is None and job.auc is None:
        stage += " (starting)"
    return (
        html.escape(stage)
        + (
            f" · {minutes(time.time() - job.started)} since {clock(job.started)}"
            if job.started
            else ""
        )
        + (
            f" · <span class='waiting'>{html.escape(job.note)}</span>"
            if job.note
            else ""
        )
    )


def job_block(job: Job, state: dict) -> str:
    bar = (
        f"<progress max='{job.epochs}' value='{min(job.epoch, job.epochs)}' "
        f"aria-label='epoch {job.epoch} of {job.epochs}'></progress>"
        if job.epoch is not None and job.epochs
        else ""
    )
    progress = job_progress(job)
    return (
        "<div class='job'><div class='head'>"
        f"<b>{idea_link(state, job.idea, job.name)}</b>"
        f"<span class='sub'>{job_stage(job)}</span>"
        f"</div>{bar}"
        + (f"<div class='sub numbers'>{progress}</div>" if progress else "")
        + "</div>"
    )


def agents(state: dict, root: Path) -> str:
    configured = (
        yaml.safe_load((root / "autoresearch/agents.yaml").read_text()) or {}
    ).get(state["provider"], {})
    jobs = campaign_dir(state["campaign"]) / "jobs"
    used = {path.parent.name.rsplit("-", 3)[0] for path in jobs.glob("*/usage.json")}
    rows = []
    for role in [*configured, *sorted(used - set(configured))]:
        if role == WATCHDOG:
            continue
        done = failed = tokens_in = tokens_out = 0
        model = (configured.get(role) or {}).get("model")
        effort = (configured.get(role) or {}).get("effort")
        working, last_end = None, None
        for path in sorted(jobs.glob(f"{role}-*")):
            usage, started, receipt = (
                read_json(path / name)
                for name in ("usage.json", "started.json", "receipt.json")
            )
            model, effort = usage.get("model") or model, usage.get("effort") or effort
            tokens = usage.get("tokens") or {}
            tokens_in += (tokens.get("input") or 0) + (tokens.get("cache_read") or 0)
            tokens_out += tokens.get("output") or 0
            begun = started.get("time")
            if receipt:
                done += 1
                failed += bool(
                    receipt.get("exit_code")
                    or receipt.get("error")
                    or usage.get("provider_error")
                )
                if begun:
                    last_end = begun + (receipt.get("seconds") or 0)
            elif begun and alive(read_json(path / "owner.json")):
                target = label(
                    read_json(path / "assignment.json").get("experiment")
                    or usage.get("experiment")
                    or ""
                )
                working = f"working {minutes(time.time() - begun)}" + (
                    f" · {target}" if target else ""
                )
        state_text = working or (
            f"idle · last {datetime.fromtimestamp(last_end).astimezone():%d %b %H:%M}"
            if last_end
            else "idle"
        )
        rows.append(
            (
                "",
                [
                    f"<b>{html.escape(role.replace('_', ' '))}</b>",
                    html.escape(str(model or "–"))
                    + (f" · {html.escape(str(effort))}" if effort else ""),
                    f"<span class=up>{html.escape(state_text)}</span>"
                    if working
                    else html.escape(state_text),
                    f"{done}"
                    + (f" <span class=down>({failed} failed)</span>" if failed else ""),
                    f"{kilo(tokens_in)} / {kilo(tokens_out)}",
                ],
            )
        )
    columns = [
        ("agent", "", "name"),
        ("model", "model", "text wide"),
        ("state", "state", "text wide"),
        ("jobs", "jobs", ""),
        ("tokens in / out", "tokens in / out", "wide"),
    ]
    return (
        f"<h2>Agents <span class='sub'>· {html.escape(state['provider'])}</span></h2>"
        + table(columns, rows)
    )


def activity(campaign: str, state: dict) -> str:
    jobs = "".join(job_block(job, state) for job in active_jobs(campaign, state))
    queue = "".join(
        f"<li>{idea_link(state, idea, idea)} <span class='sub'>{html.escape(state['ideas'][idea].get('status') or '')}</span></li>"
        for idea in state["queue"]
    )
    return (
        "<div class='split'>"
        f"<div class='card'><div class='label'>Now</div>{jobs or '<div class=sub>no active jobs</div>'}</div>"
        f"<div class='card'><div class='label'>Up next</div>"
        f"<div class='sub'>{html.escape(research_text(state))}</div>"
        f"{f'<ol>{queue}</ol>' if queue else '<div class=sub>queue empty</div>'}</div>"
        "</div>"
    )


def messages(campaign: str) -> str:
    entries = read_messages(campaign)
    count = delivered(campaign)
    waiting = len(entries) - count
    rows = "".join(
        "<div class='message'><div class='sub'>"
        f"{entry.time.astimezone():%d %b %H:%M} · "
        + (
            "delivered to the keeper"
            if index < count
            else "<span class='waiting'>waiting for the keeper</span>"
        )
        + f"</div>{html.escape(entry.text)}</div>"
        for index, entry in enumerate(entries)
    )
    return (
        "<h2>Messages to the keeper"
        + (
            f" <span class='sub waiting'>· {waiting} waiting</span>"
            if waiting > 0
            else ""
        )
        + "</h2><div class='card'>"
        f"<div class='thread' id='thread'>{rows or "<div class='sub'>none yet</div>"}</div>"
        "<form class='send' method='post' action='messages'>"
        f"<textarea id='draft' name='text' maxlength='{MAXIMUM_MESSAGE_BYTES // 4}' required "
        "aria-label='Message to the keeper' placeholder='Message to the keeper'></textarea>"
        "<button class='primary'>Send</button></form></div>"
    )


def icon(color: str) -> str:
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><circle cx='8' cy='8' r='7' fill='{LIGHT[color]}'/></svg>"
    return "data:image/svg+xml," + quote(svg)


def page(
    title: str,
    color: str,
    navigation: str,
    body: str,
    plots: bool = False,
) -> str:
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{title}</title><link rel='icon' href='{icon(color)}'>"
        f"<script>{THEME_BOOT}</script>"
        + (f"<script src='/{PLOTLY_JS}'></script>" if plots else "")
        + f"<style>{STYLE}</style></head><body>"
        + (f"<nav>{navigation}</nav>" if navigation else "")
        + f"<main>{body}</main>"
        "<footer><button id='theme' type='button'>Theme</button></footer>"
        f"<script>{SCRIPT}{PLOT_SCRIPT if plots else ''}</script>"
        "</body></html>"
    )


def age_line(settings: MonitorCLI) -> str:
    return f"<div id='age' data-interval='{settings.interval}'></div>"


def navigation(campaign: str) -> str:
    links = []
    for name in registrations():
        try:
            color = overview(name, load_state(name)).color
        except PAGE_ERRORS:
            color = "--bad"
        current = " aria-current='page'" if name == campaign else ""
        links.append(
            f"<a href='/{html.escape(name)}/'{current}>"
            f"<span class='dot' style='--c: var({color})'></span>{html.escape(name)}</a>"
        )
    return f"<a class='home' href='/'>← Campaigns</a><div class='strip'>{''.join(links)}</div>"


def banner(view: Overview) -> str:
    return (
        f"<div class='banner{' alert' if view.alert else ''}' style='--c: var({view.color})'>"
        f"<div class='state'><span class='dot'></span><b>{html.escape(view.status)}</b>"
        f"<span>supervisor {'alive' if view.supervisor else 'not running'}</span>"
        f"<span class='sub'>agents on {html.escape(view.provider)}</span></div>"
        f"<div class='detail'>{html.escape(view.detail)}</div></div>"
    )


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def regenerate(settings: MonitorCLI, entry: Registration) -> None:
    campaign = entry.campaign
    if not (settings.directory / PLOTLY_JS).exists():
        write(settings.directory / PLOTLY_JS, get_plotlyjs())
    state = load_state(campaign)
    view = overview(campaign, state)
    control, runs = collect(state)
    fold = campaign_fold(state)
    body = (
        f"<h1>{html.escape(campaign)}</h1>"
        + banner(view)
        + age_line(settings)
        + activity(campaign, state)
        + tiles(control, runs, fold, state)
        + messages(campaign)
        + "<h2>Results</h2>"
        + results(control, runs, state)
        + comparison(control, runs, fold)
        + "<h2>Curves</h2>"
        + curves(control, runs)
        + agents(state, entry.root)
        + "<details class='status'><summary>Full status (runs the status command)</summary><pre></pre></details>"
        + idea_dialogs(state, [*state["queue"], *(run["idea"] for run in runs)])
    )
    write(
        settings.directory / campaign / "index.html",
        page(
            f"{html.escape(view.status)} · {html.escape(campaign)}",
            view.color,
            navigation(campaign),
            body,
            plots=True,
        ),
    )
    print(
        f"{datetime.now().astimezone():%H:%M} {campaign} {view.status} "
        f"supervisor {'alive' if view.supervisor else 'not running'}",
        flush=True,
    )


def failure(settings: MonitorCLI, campaign: str, error: Exception) -> None:
    body = (
        f"<h1>{html.escape(campaign)}</h1>"
        "<div class='banner alert' style='--c: var(--bad)'><div class='state'><span class='dot'></span>"
        "<b>PAGE FAILED</b></div><div class='detail'>The monitor could not read this campaign; "
        "it retries on the next refresh.</div></div>"
        + age_line(settings)
        + f"<pre style='margin-top: 12px'>{html.escape(f'{type(error).__name__}: {error}')}</pre>"
    )
    write(
        settings.directory / campaign / "index.html",
        page(f"FAILED · {html.escape(campaign)}", "--bad", navigation(campaign), body),
    )


def fresh(settings: MonitorCLI, entry: Registration) -> None:
    with REGENERATION:
        age = time.monotonic() - last_regeneration.get(entry.campaign, float("-inf"))
        if age < MINIMUM_AGE_SECONDS:
            return
        try:
            regenerate(settings, entry)
        except PAGE_ERRORS as error:
            print(f"regeneration of {entry.campaign} failed: {error!r}", flush=True)
            failure(settings, entry.campaign, error)
        last_regeneration[entry.campaign] = time.monotonic()


def job_line(job: Job) -> str:
    progress = job_progress(job)
    return (
        f"<li><b>{html.escape(job.name)}</b> <span class='sub'>{job_stage(job)}</span>"
        + (f"<div class='sub'>{progress}</div>" if progress else "")
        + "</li>"
    )


def summary_card(name: str) -> tuple[tuple, str]:
    state = load_state(name)
    view = overview(name, state)
    fold = campaign_fold(state)
    decided, accepted = tally(state)
    control = x4(Path(state["reference"]), fold)
    incumbent = x4(Path(state["incumbent"]["run"]), fold)
    leader = (
        CONTROL
        if Path(state["incumbent"]["run"]).resolve()
        == Path(state["reference"]).resolve()
        else label(state["incumbent"]["id"])
    )
    gain = (
        incumbent - control
        if leader != CONTROL and None not in (incumbent, control)
        else None
    )
    waiting = len(read_messages(name)) - delivered(name)
    jobs = "".join(job_line(job) for job in active_jobs(name, state))
    card = (
        f"<a class='campaign card' href='/{html.escape(name)}/'>"
        f"<div class='head'><span class='name'>{html.escape(name)}</span>{badge(view)}</div>"
        f"<div class='sub'>{html.escape(view.detail)}</div>"
        + (f"<ul>{jobs}</ul>" if jobs else "")
        + f"<div class='sub'>incumbent <b>{html.escape(leader)}</b>"
        + (f" · ×4 {tone(gain)} vs control" if gain is not None else "")
        + f" · {decided} decided · {accepted} accepted · {len(state['queue'])} queued</div>"
        + (
            f"<div class='waiting'>{waiting} message{'s' if waiting > 1 else ''} waiting for the keeper</div>"
            if waiting > 0
            else ""
        )
        + "</a>"
    )
    return (not view.alert, STATE_ORDER.get(view.status, 3), name), card


def front_page(settings: MonitorCLI) -> None:
    cards = []
    for name in registrations():
        try:
            cards.append(summary_card(name))
        except PAGE_ERRORS as error:
            cards.append(
                (
                    (False, 0, name),
                    (
                        f"<a class='campaign card' href='/{html.escape(name)}/'>"
                        f"<div class='head'><span class='name'>{html.escape(name)}</span>"
                        "<span class='badge down'>unreadable</span></div>"
                        f"<div class='sub'>{html.escape(f'{type(error).__name__}: {error}')}</div></a>"
                    ),
                )
            )
    alert = any(not key[0] for key, _ in cards)
    body = (
        "<h1>Campaigns</h1>"
        + age_line(settings)
        + (
            "".join(card for _, card in sorted(cards))
            or "<div class='sub'>No live campaign.</div>"
        )
    )
    write(
        settings.directory / "index.html",
        page("Campaigns", "--bad" if alert else "--s1", "", body),
    )


def fresh_front(settings: MonitorCLI) -> None:
    with REGENERATION:
        front_page(settings)


def full_status(entry: Registration) -> str:
    python = entry.root / ".venv/bin/python"
    with STATUS:
        try:
            result = subprocess.run(
                [
                    str(python if python.exists() else sys.executable),
                    "-m",
                    "scripts.tools.autoresearch",
                    "status",
                    entry.campaign,
                ],
                cwd=entry.root,
                capture_output=True,
                text=True,
                check=False,
                timeout=STATUS_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return f"status timed out after {STATUS_TIMEOUT_SECONDS} s"
    if result.returncode:
        return f"status failed with exit code {result.returncode}\n\n" + "\n".join(
            result.stderr.strip().splitlines()[-12:]
        )
    return result.stdout


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, settings: MonitorCLI, **kwargs):
        self.settings = settings
        super().__init__(*args, directory=str(settings.directory), **kwargs)

    def do_GET(self) -> None:
        parts = self.path.split("?")[0].strip("/").split("/")
        if parts in ([""], ["index.html"]):
            fresh_front(self.settings)
        elif len(parts) == 2 and parts[1] == "status":
            entry = registrations().get(parts[0])
            if entry is None:
                self.send_error(404)
                return
            self.send_text(full_status(entry))
            return
        elif len(parts) <= 2 and parts[1:] in ([], ["index.html"]):
            entry = registrations().get(parts[0])
            if entry is not None:
                fresh(self.settings, entry)
        super().do_GET()

    def send_text(self, text: str) -> None:
        data = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        sender = ip_address(self.client_address[0])
        if not any(sender in network for network in TAILNET):
            self.send_error(403)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        parts = self.path.strip("/").split("/")
        campaign = parts[0] if len(parts) == 2 and parts[1] == "messages" else None
        if campaign not in registrations() or not 0 < length <= MAXIMUM_MESSAGE_BYTES:
            self.send_error(400)
            return
        form = parse_qs(self.rfile.read(length).decode(errors="replace"))
        text = form.get("text", [""])[0].replace("\r\n", "\n").strip()
        if text:
            with REGENERATION:
                append_message(campaign, text)
                last_regeneration.pop(campaign, None)
        self.send_response(303)
        self.send_header("Location", f"/{campaign}/")
        self.end_headers()

    def end_headers(self) -> None:
        if not self.path.endswith(".js"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        return


class IPv6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def run(settings: MonitorCLI) -> None:
    settings.directory.mkdir(parents=True, exist_ok=True)
    fresh_front(settings)
    handler = partial(Handler, settings=settings)
    servers = []
    for host in settings.hosts:
        server_class = IPv6Server if ":" in host else ThreadingHTTPServer
        servers.append(server_class((host, settings.port), handler))
        print(f"serving on {host} port {settings.port}", flush=True)
    for server in servers[1:]:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    servers[0].serve_forever()


if __name__ == "__main__":
    run(MonitorCLI())
