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
from pathlib import Path

import plotly
import plotly.graph_objects as go
import yaml
from plotly.offline import get_plotlyjs
from pydantic import Field

from knee_abnormality.autoresearch.runtime.layout import (
    PREDICTIONS,
    campaign_fold,
    fold_dir,
)
from knee_abnormality.cli import CLISettings, describe

CONTROL = "control"
WATCHDOG = "watchdog"
ADOPTION_DELTA = 0.001
GAINS: dict[tuple, list[float]] = {}
PLOTLY_JS = f"plotly-{plotly.__version__}.min.js"
BANNER = {
    "RUNNING": "var(--good)",
    "CALIBRATING": "var(--s1)",
    "STOPPING": "var(--warn)",
    "STOPPED": "var(--warn)",
}
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
    "--other": "#a8a79f",
    "--s1": "#2a78d6",
    "--s2": "#eb6834",
    "--good": "#0ca30c",
}
PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}
STYLE = """
:root { color-scheme: light; --bg: #fcfcfb; --card: #f0efec; --ink: #0b0b0b; --ink2: #52514e; --grid: #e4e3df;
  --other: #a8a79f; --s1: #2a78d6; --s2: #eb6834; --good: #0ca30c; --warn: #eda100; --bad: #e34948; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --bg: #1a1a19;
  --card: #262624; --ink: #ffffff; --ink2: #c3c2b7; --grid: #383835; --other: #75746e; --s1: #3987e5;
  --s2: #d95926; --good: #1baf7a; --warn: #c98500; --bad: #e66767; } }
:root[data-theme="dark"] { color-scheme: dark; --bg: #1a1a19; --card: #262624; --ink: #ffffff; --ink2: #c3c2b7;
  --grid: #383835; --other: #75746e; --s1: #3987e5; --s2: #d95926; --good: #1baf7a; --warn: #c98500; --bad: #e66767; }
body { margin: 0 auto; max-width: 1440px; padding: 12px 16px; background: var(--bg); color: var(--ink);
  font: 14px system-ui, sans-serif; }
h1 { font-size: 18px; margin: 4px 0 8px; }
h2 { font-size: 13px; color: var(--ink2); text-transform: uppercase; letter-spacing: .04em; margin: 22px 0 6px; }
.banner { color: #fff; border-radius: 8px; padding: 10px 12px; font-weight: 600; }
.banner small { display: block; font-weight: 400; margin-top: 4px; }
.card { background: var(--card); border-radius: 8px; padding: 8px 10px; min-width: 0; overflow: hidden; }
.plotly-graph-div { width: 100% !important; }
.sub { color: var(--ink2); font-size: 13px; }
.grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 8px; }
@media (max-width: 900px) { .grid { grid-template-columns: minmax(0, 1fr); } }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 8px; }
.tile .value { font-size: 22px; font-weight: 600; margin-top: 2px; }
.scroll { overflow-x: auto; margin-top: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
th, td { text-align: right; padding: 5px 6px; border-bottom: 1px solid var(--grid); white-space: nowrap;
  font-variant-numeric: tabular-nums; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
th { color: var(--ink2); font-weight: 600; }
tr.incumbent td:first-child::before { content: "★ "; color: var(--s1); }
.accept { color: var(--good); font-weight: 600; } .reject { color: var(--bad); font-weight: 600; }
ul, ol { margin: 6px 0 2px 18px; padding: 0; } li { margin: 2px 0; }
#age { font-size: 12px; margin-top: 8px; color: var(--ink2); }
.stale { color: var(--bad) !important; font-weight: 600; }
pre { font-size: 11px; line-height: 1.35; white-space: pre-wrap; background: var(--card); padding: 8px; border-radius: 6px; }
details { margin-top: 18px; }
.activity { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 1.3fr) minmax(0, 0.8fr); gap: 12px;
  margin-top: 8px; align-items: stretch; }
table.agents th:nth-child(3), table.agents td:nth-child(3) { text-align: left; }
.card.wide { overflow-x: auto; }
.card.wide .plotly-graph-div { min-width: 720px; }
.pair { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 2fr); gap: 12px; margin-top: 12px; }
@media (max-width: 900px) { .pair { grid-template-columns: minmax(0, 1fr); } }
@media (max-width: 1100px) { .activity { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); } }
@media (max-width: 720px) { .activity { grid-template-columns: minmax(0, 1fr); } }
table.jobs { margin-top: 6px; font-size: 13px; }
table.jobs th, table.jobs td { border-bottom: none; padding: 4px 5px; white-space: normal; }
table.jobs td:nth-child(2), table.jobs td:nth-child(3) { color: var(--ink2); }
table.jobs th:nth-child(2), table.jobs td:nth-child(2) { text-align: left; }
.up { color: var(--good); font-weight: 600; } .down { color: var(--bad); font-weight: 600; }
"""
SCRIPT = """
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function theme() {
  document.querySelectorAll(".js-plotly-plot").forEach(gd => {
    const indices = gd.data.map((trace, index) => trace.meta ? index : -1).filter(index => index >= 0);
    const colors = indices.map(index => css(gd.data[index].meta));
    if (indices.length) Plotly.restyle(gd, {"line.color": colors, "marker.color": colors, "error_x.color": colors}, indices);
    Plotly.relayout(gd, {
      "font.color": css("--ink2"), "xaxis.gridcolor": css("--grid"), "yaxis.gridcolor": css("--grid"),
      "xaxis.zerolinecolor": css("--ink2"), "yaxis.zerolinecolor": css("--ink2"),
      "legend.font.color": css("--ink2"), "hoverlabel.bgcolor": css("--card"), "hoverlabel.font.color": css("--ink"),
      "shapes": (gd.layout.shapes || []).map(shape => ({...shape, line: {...shape.line, color: css(shape.name || "--ink2")}})),
    });
    gd.on("plotly_relayout", touched);
    gd.on("plotly_legendclick", touched);
  });
}
const age = document.getElementById("age");
const interval = Number(age.dataset.interval) * 1000;
let lastTouch = 0;
function touched() { lastTouch = Date.now(); }
function describeAge() {
  const minutes = (Date.now() - Number(age.dataset.updated)) / 60000;
  const stale = minutes * 60000 > 2.5 * interval;
  age.textContent = "updated " + Math.round(minutes) + " min ago" + (stale ? " · updater may be down" : "")
    + " · reloads every " + Math.round(interval / 1000) + " s unless you are zooming";
  age.className = stale ? "stale" : "";
}
window.addEventListener("load", () => {
  document.querySelectorAll(".js-plotly-plot").forEach(gd => Plotly.Plots.resize(gd));
  theme();
  describeAge();
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", theme);
setInterval(() => { describeAge(); if (Date.now() - lastTouch > interval) location.reload(); }, interval);
"""
PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{campaign} monitor</title>
<script src="{plotly_js}"></script>
<style>{style}</style></head>
<body>
<h1>{campaign}</h1>
<div class="banner" style="background:{color}">{state} · supervisor {supervisor}<small>{detail}</small></div>
<div id="age" data-updated="{updated_ms}" data-interval="{refresh}"></div>
{body}
<details><summary>Full status</summary><pre>{status}</pre></details>
<script>{script}</script>
</body></html>
"""


class MonitorCLI(CLISettings):
    model_config = describe(
        "Serve an auto-refreshing page for one autoresearch campaign: its state, the decided runs "
        "against the control, per-epoch validation AUC and loss curves of the control, the "
        "incumbent and the current run, and per-label deltas, regenerated on each page request"
    )

    campaign: str = Field(description="Campaign ID")
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


def label(name: str) -> str:
    return re.sub(r"-2026\d{4}T\d{6}Z-[0-9a-f]{8}$", "", name)


def epoch_metrics(fold: Path) -> dict[int, dict]:
    metrics = {}
    for path in sorted((fold / "epochs").glob("metrics_epoch_*.json")):
        metrics[int(path.stem.rsplit("_", 1)[1])] = json.loads(path.read_text())
    return metrics


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def run_record(name: str, run: Path, fold: int) -> dict:
    directory = fold_dir(run, fold)
    selected = read_json(directory / "selected_checkpoint.json")
    epoch = re.search(r"best-epoch=(\d+)", selected.get("path", ""))
    return {
        "name": name,
        "epochs": epoch_metrics(directory),
        "x4": selected.get("prediction_macro_auc"),
        "best_epoch": int(epoch[1]) if epoch else None,
        "seconds": read_json(directory / "timing.json").get("training_seconds"),
    }


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
        evidence = exp.get("evidence") or {}
        scores = evidence.get("macro_auc")
        reference = exp.get("reference")
        record.update(
            run=training.resolve(),
            withdrawn=withdrawn,
            comparison=evidence.get("comparison"),
            level=scores["reference"] - control["x4"]
            if scores and control["x4"] is not None
            else None,
            verdict=verdict or ("training" if key == running else "pending"),
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


def signed(value) -> str:
    return "–" if value is None else f"{value:+.4f}"


def number(value, digits: int = 4) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def layout(title: str, **extra) -> dict:
    axis = {
        "showgrid": True,
        "gridcolor": "#e4e3df",
        "gridwidth": 1,
        "zeroline": False,
        "showline": False,
        "ticks": "",
        "automargin": True,
    }
    return {
        "title": {"text": title, "x": 0, "xanchor": "left", "font": {"size": 13}},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "system-ui, sans-serif", "size": 11, "color": "#52514e"},
        "margin": {"l": 8, "r": 8, "t": 36, "b": 8},
        "height": 330,
        "hovermode": "closest",
        "hoverlabel": {"font": {"size": 12}},
        "legend": {"orientation": "h", "y": -0.18, "font": {"size": 11}},
        "xaxis": axis,
        "yaxis": {**axis, "tickformat": ".4f"},
        **extra,
    }


def figure_html(figure: go.Figure, css: str = "card") -> str:
    return (
        f"<div class='{css}'>"
        + figure.to_html(full_html=False, include_plotlyjs=False, config=PLOT_CONFIG)
        + "</div>"
    )


def curve(title: str, series: list[tuple[dict, dict[int, float]]]) -> str:
    figure = go.Figure()
    for record, values in series:
        if not values:
            continue
        epochs = sorted(values)
        role = record["role"]
        figure.add_trace(
            go.Scatter(
                x=epochs,
                y=[values[epoch] for epoch in epochs],
                name=record["name"],
                meta=record["color"],
                mode="lines+markers",
                line={**LINE[role], "shape": "linear", "color": LIGHT[record["color"]]},
                marker={"size": 5, "color": LIGHT[record["color"]]},
                hovertemplate=f"<b>{html.escape(record['name'])}</b><br>epoch %{{x}}<br>%{{y:.4f}}<extra></extra>",
            )
        )
    extra = {"xaxis": {**layout("")["xaxis"], "title": {"text": "epoch"}, "dtick": 1}}
    figure.update_layout(layout(title, **extra))
    return figure_html(figure)


def per_label(run: dict) -> dict[str, dict]:
    path = Path(run["per_label"]) if run["per_label"] else None
    entries = (
        json.loads(path.read_text())["per_target"] if path and path.exists() else []
    )
    return {entry["target"]: entry for entry in entries}


def control_gain(control: dict, run: Path, fold: int, runs: list[dict]) -> list | None:
    report = next((r["comparison"] for r in runs if r["comparison"]), None)
    files = [fold_dir(path, fold) / PREDICTIONS for path in (control["run"], run)]
    key = (*files, report)
    if run == control["run"] or not report or not all(f.is_file() for f in files):
        return None
    if key not in GAINS:
        import numpy as np
        import polars as pl

        from knee_abnormality.data import splits
        from knee_abnormality.evaluation import metrics

        report = read_json(Path(report))
        inputs = {Path(item["path"]).name: item["path"] for item in report["inputs"]}
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
        groups, inverse = np.unique(
            frame["report_group"].to_numpy(), return_inverse=True
        )
        counts = np.random.default_rng(bootstrap["seed"]).multinomial(
            len(groups), np.full(len(groups), 1 / len(groups)), size=bootstrap["draws"]
        )[:, inverse]
        positive = counts @ hard
        weights = counts[
            ((positive > 0) & (positive < counts.sum(axis=1, keepdims=True))).all(
                axis=1
            )
        ]
        if not len(weights):
            return None
        reference, candidate = (
            metrics.bootstrap_auc(
                hard, score.select(targets).to_numpy().astype(float), weights
            )
            for score in scores
        )
        GAINS[key] = np.percentile(candidate - reference, [10, 90]).tolist()
    return GAINS[key]


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


def comparison(control: dict, runs: list[dict], fold: int) -> str:
    scored = [
        run
        for run in runs
        if run["delta"] is not None and run["interval"] and run["level"] is not None
    ]
    if not scored:
        return "<div class='card'><div class='sub'>No scored run yet.</div></div>"
    names = [
        ("<span style='color:#2a78d6'>★</span> " if run["role"] == "incumbent" else "")
        + run["name"]
        for run in scored
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
    base = layout("")
    rows = {
        "height": max(220, 150 + 26 * len(scored)),
        "margin": {"l": 8, "r": 8, "t": 56, "b": 94},
        "yaxis": {
            **base["yaxis"],
            "tickformat": None,
            "showgrid": False,
            "autorange": "reversed",
        },
    }
    campaign = {control["run"], *(run["run"] for run in runs)}
    levels = [run["level"] for run in scored]
    manual = {row for row, run in enumerate(scored) if run["reference"] not in campaign}
    everything = set(range(len(scored)))
    gains = {
        reference: control_gain(control, reference, fold, runs)
        for reference in {run["reference"] for run in scored}
    }
    intervals = go.Figure()
    for spans, style, color in (
        (segments(levels, everything - manual), {"width": 1}, "--ink2"),
        (
            segments(levels, manual, joined=False),
            {"width": 1, "dash": "dash"},
            "--ink2",
        ),
        (
            segments(levels, everything, ADOPTION_DELTA),
            {"width": 1, "dash": "dot"},
            "--good",
        ),
    ):
        intervals.add_trace(
            go.Scatter(
                **spans,
                meta=color,
                mode="lines",
                line={**style, "color": LIGHT[color]},
                showlegend=False,
                hoverinfo="skip",
            )
        )
    for row, (name, run) in enumerate(zip(names, scored)):
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
    intervals.update_layout(
        layout(
            "×4 macro Δ vs control, 80% interval vs each run's reference"
            "<br><sup>line: incumbent each run was judged against, dashed if set by hand"
            " · green: +0.001 above it</sup>",
            **{
                **rows,
                "yaxis": {
                    **base["yaxis"],
                    "tickformat": None,
                    "showgrid": False,
                    "tickmode": "array",
                    "tickvals": list(range(len(scored))),
                    "ticktext": names,
                    "range": [len(scored) - 0.5, -0.5],
                },
            },
            xaxis={**base["xaxis"], "tickformat": "+.3f", "automargin": False},
        )
    )
    boxes = figure_html(intervals)
    if targets:
        heatmap = go.Figure(
            go.Heatmap(
                x=targets,
                y=names,
                z=[
                    [
                        (row[t]["candidate"] - baseline[t]) * 1000 if t in row else None
                        for t in targets
                    ]
                    for row in cells
                ],
                customdata=[
                    [
                        f"{baseline[t]:.4f} → {row[t]['candidate']:.4f}"
                        if t in row
                        else ""
                        for t in targets
                    ]
                    for row in cells
                ],
                texttemplate="%{z:+.1f}",
                textfont={"size": 10, "color": "#0b0b0b"},
                colorscale=[[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]],
                zmin=-10,
                zmax=10,
                showscale=False,
                xgap=2,
                ygap=2,
                hovertemplate="<b>%{y}</b><br>%{x}: %{customdata}<br>Δ ×1000 %{z:+.1f}<extra></extra>",
            )
        )
        heatmap.update_layout(
            layout(
                "Per-label ×4 Δ vs control ×1000<br><sup>blue up, red down, full colour at ±10</sup>",
                **rows,
                xaxis={
                    **base["xaxis"],
                    "showgrid": False,
                    "tickangle": -40,
                    "automargin": False,
                },
            )
        )
        boxes += figure_html(heatmap, "card wide")
    return f"<div class='pair'>{boxes}</div>"


def curves(control: dict, runs: list[dict]) -> str:
    everything = [control, *(run for run in runs if run["role"] != "other")]
    validation = [
        (run, {e: m["macro_auc"] for e, m in run["epochs"].items()})
        for run in everything
    ]
    soft = [
        (
            run,
            {
                e: m["soft_macro_auc"]
                for e, m in run["epochs"].items()
                if "soft_macro_auc" in m
            },
        )
        for run in everything
    ]
    bce = [
        (
            run,
            {
                e: m["soft_target_bce"]
                for e, m in run["epochs"].items()
                if "soft_target_bce" in m
            },
        )
        for run in everything
    ]
    return (
        "<div class='sub'>Black: control. Blue: the incumbent. Orange: the run training now (dashed) or, "
        "until the next one logs an epoch, the last run that finished. "
        "Click a legend entry to hide a run, double-click to isolate it.</div>"
        "<div class='grid'>"
        + curve("×1 validation macro AUC", validation)
        + curve("×1 validation disease loss (soft-target BCE, lower is better)", bce)
        + curve("×1 soft macro AUC", soft)
        + "</div>"
    )


def tiles(control: dict, runs: list[dict], fold: int) -> str:
    incumbent = next((run for run in runs if run["role"] == "incumbent"), control)
    bounds = control_gain(control, incumbent["run"], fold, runs)
    decided = [
        run for run in runs if run["verdict"] not in {"training", "pending", "failed"}
    ]
    gain = (
        incumbent["x4"] - control["x4"]
        if incumbent["x4"] is not None and control["x4"] is not None
        else None
    )
    items = [
        ("Control ×4", number(control["x4"])),
        ("Incumbent", html.escape(incumbent["name"])),
        (
            "Gain over control",
            signed(gain)
            + (
                f"<div class='sub'>80% [{bounds[0]:+.4f}, {bounds[1]:+.4f}]</div>"
                if bounds
                else ""
            ),
        ),
        (
            "Decided · accepted",
            f"{len(decided)} · {sum(run['verdict'] == 'provisional_accept' for run in decided)}",
        ),
    ]
    return (
        "<div class='tiles'>"
        + "".join(
            f"<div class='card tile'><div class='sub'>{name}</div><div class='value'>{value}</div></div>"
            for name, value in items
        )
        + "</div>"
    )


def results(control: dict, runs: list[dict]) -> str:
    header = (
        "<tr><th>run</th><th>verdict</th><th>×4 macro</th><th>Δ vs its reference</th><th>80% interval</th>"
        "<th>soft Δ</th><th>Δ ×4 vs control</th><th>best ×1</th><th>best epoch</th><th>train s</th></tr>"
    )
    rows = []
    for run in [control, *runs]:
        verdict = run["verdict"]
        css = (
            "reject"
            if verdict.split()[0] in {"reject", "invalid", "failed"}
            else "accept"
            if verdict.endswith("accept")
            else ""
        )
        interval = run.get("interval")
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
            f"<tr class='{'incumbent' if starred else ''}'><td>{html.escape(run['name'])}</td>"
            f"<td class='{css}'>{html.escape(verdict)}</td><td>{number(run['x4'])}</td><td>{signed(run.get('delta'))}</td>"
            f"<td>{'–' if not interval else f'[{interval[0]:+.4f}, {interval[1]:+.4f}]'}</td><td>{signed(run.get('soft'))}</td>"
            f"<td>{signed(versus)}</td>"
            f"<td>{number(max(m['macro_auc'] for m in run['epochs'].values())) if run['epochs'] else '–'}</td>"
            f"<td>{'–' if run['best_epoch'] is None else run['best_epoch']}</td>"
            f"<td>{number(run['seconds'], 0)}</td></tr>"
        )
    return f"<div class='scroll card'><table>{header}{''.join(rows)}</table></div>"


def status_lines(status: str, heading: str) -> list[str]:
    lines = status.splitlines()
    if heading not in lines:
        return []
    block = []
    for line in lines[lines.index(heading) + 1 :]:
        if not line.strip():
            break
        block.append(" ".join(line.split()))
    return block


def job(line: str) -> str:
    role, _, rest = line.partition(" ")
    elapsed = re.match(r"((?:\d+h )?\d+m \d+s)\s+", rest)
    target = rest[elapsed.end() :] if elapsed else rest
    name, _, details = target.partition(",")
    stage = ROLES.get(role, role)
    epoch = re.search(r"epoch (\d+)/(\d+)", details)
    score = re.search(
        r"val AUC ([\d.]+) \(incumbent ([\d.]+) at this epoch, ([+-][\d.]+)\)", details
    )
    cells = ["", "", ""]
    if role == "training" and epoch and score:
        delta = float(score[3])
        cells = [
            f"{epoch[1]}/{epoch[2]}",
            f"{float(score[1]):.4f}",
            f"<span class='{'up' if delta >= 0 else 'down'}'>{delta:+.4f}</span>",
        ]
    elif role == "training":
        stage += " (starting)"
    return (
        f"<tr><td><b>{html.escape(label(name.strip()))}</b></td><td>{html.escape(stage)}</td>"
        f"<td>{elapsed[1] if elapsed else ''}</td>"
        + "".join(f"<td>{cell}</td>" for cell in cells)
        + "</tr>"
    )


def kilo(value: int) -> str:
    return f"{value / 1e6:.1f}M" if value >= 1e6 else f"{value / 1e3:.0f}k"


def minutes(seconds: float) -> str:
    return f"{seconds / 60:.0f} min" if seconds >= 60 else f"{seconds:.0f} s"


def agents(state: dict) -> str:
    configured = yaml.safe_load(Path("autoresearch/agents.yaml").read_text()).get(
        state["provider"], {}
    )
    root = Path(f"outputs/autoresearch/{state['campaign']}/jobs")
    used = {path.parent.name.rsplit("-", 3)[0] for path in root.glob("*/usage.json")}
    rows = []
    for role in [
        role
        for role in [*configured, *sorted(used - set(configured))]
        if role != WATCHDOG
    ]:
        done = failed = tokens_in = tokens_out = 0
        model, effort = (
            configured.get(role, {}).get("model"),
            configured.get(role, {}).get("effort"),
        )
        working, last_end = None, None
        for path in sorted(root.glob(f"{role}-*")):
            usage, started, receipt = (
                read_json(path / name)
                for name in ("usage.json", "started.json", "receipt.json")
            )
            model, effort = usage.get("model", model), usage.get("effort", effort)
            tokens = usage.get("tokens", {})
            tokens_in += tokens.get("input", 0) + tokens.get("cache_read", 0)
            tokens_out += tokens.get("output", 0)
            if receipt:
                done += 1
                failed += bool(
                    receipt.get("exit_code")
                    or receipt.get("error")
                    or usage.get("provider_error")
                )
                if started:
                    last_end = started["time"] + receipt.get("seconds", 0)
            elif started:
                target = label(
                    read_json(path / "assignment.json").get("experiment")
                    or usage.get("experiment")
                    or ""
                )
                working = f"working {minutes(time.time() - started['time'])}" + (
                    f" · {html.escape(target)}" if target else ""
                )
        state_text = working or (
            f"idle · last {datetime.fromtimestamp(last_end).astimezone():%H:%M}"
            if last_end
            else "idle"
        )
        rows.append(
            f"<tr><td><b>{html.escape(role.replace('_', ' '))}</b></td>"
            f"<td>{html.escape(str(model or '–'))}{' · ' + html.escape(str(effort)) if effort else ''}</td>"
            f"<td>{f'<span class=up>{state_text}</span>' if working else state_text}</td>"
            f"<td>{done}{f' <span class=down>({failed} failed)</span>' if failed else ''}</td>"
            f"<td>{kilo(tokens_in)} / {kilo(tokens_out)}</td></tr>"
        )
    return (
        "<table class='jobs agents'><tr><th>agent</th><th>model</th><th>state</th><th>jobs</th><th>tokens in / out</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def activity(status: str, state: dict) -> str:
    rows = "".join(job(line) for line in status_lines(status, "Now"))
    jobs = (
        "<table class='jobs'><tr><th>run</th><th>stage</th><th>time</th><th>epoch</th><th>×1</th>"
        f"<th title='×1 macro AUC minus the incumbent at the same epoch'>Δ inc.</th></tr>{rows}</table>"
        if rows
        else "<div class='sub'>idle</div>"
    )
    queue = "".join(
        f"<li>{html.escape(idea)} <span class='sub'>{html.escape(state['ideas'][idea].get('status', ''))}</span></li>"
        for idea in state["queue"]
    )
    return (
        "<div class='activity'>"
        f"<div class='card'><div class='sub'>Now</div>{jobs}</div>"
        f"<div class='card'><div class='sub'>Agents · {html.escape(state['provider'])}</div>{agents(state)}</div>"
        f"<div class='card'><div class='sub'>Up next</div><ol>{queue or '<li>empty</li>'}</ol></div>"
        "</div>"
    )


def regenerate(settings: MonitorCLI) -> None:
    settings.directory.mkdir(parents=True, exist_ok=True)
    if not (settings.directory / PLOTLY_JS).exists():
        (settings.directory / PLOTLY_JS).write_text(get_plotlyjs())
    status = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.tools.autoresearch",
            "status",
            settings.campaign,
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    header = next(
        (
            line
            for line in status.splitlines()
            if line.startswith(f"{settings.campaign} · ")
        ),
        "",
    )
    parts = [part.strip() for part in header.split("·")]
    state_name = parts[2] if len(parts) > 2 else "UNKNOWN"
    supervisor = parts[3].removeprefix("supervisor ") if len(parts) > 3 else "unknown"
    state = json.loads(
        Path(f"outputs/autoresearch/{settings.campaign}/state.json").read_text()
    )
    control, runs = collect(state)
    fold = campaign_fold(state)
    detail = state.get("reason") or {
        "RUNNING": f"working normally · agents on {state['provider']}",
        "CALIBRATING": "measuring the budget on CPU",
        "STOPPED": "stopped; nothing runs until resumed",
        "STOPPING": "finishing the current run before stopping",
        "PAUSED": "paused; needs a repair",
    }.get(state_name, "")
    color = (
        "var(--bad)"
        if supervisor != "alive" or state_name == "PAUSED"
        else BANNER.get(state_name, "var(--ink2)")
    )
    body = (
        tiles(control, runs, fold)
        + activity(status, state)
        + "<h2>Results</h2>"
        + results(control, runs)
        + comparison(control, runs, fold)
        + "<h2>Curves</h2>"
        + curves(control, runs)
    )
    page = PAGE.format(
        campaign=html.escape(settings.campaign),
        plotly_js=PLOTLY_JS,
        style=STYLE,
        state=html.escape(state_name),
        supervisor=html.escape(supervisor),
        color=color,
        detail=html.escape(detail),
        updated_ms=int(time.time() * 1000),
        refresh=settings.interval,
        body=body,
        status=html.escape(status),
        script=SCRIPT,
    )
    (settings.directory / "index.tmp").write_text(page)
    (settings.directory / "index.tmp").replace(settings.directory / "index.html")
    print(
        f"{datetime.now().astimezone():%H:%M} {state_name} supervisor {supervisor}",
        flush=True,
    )


MINIMUM_AGE_SECONDS = 5
REGENERATION = threading.Lock()
last_regeneration = [float("-inf")]


def fresh(settings: MonitorCLI) -> None:
    with REGENERATION:
        if time.monotonic() - last_regeneration[0] < MINIMUM_AGE_SECONDS:
            return
        try:
            regenerate(settings)
        except (OSError, ValueError, KeyError) as error:
            print(f"regeneration failed: {error}", flush=True)
        last_regeneration[0] = time.monotonic()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, settings: MonitorCLI, **kwargs):
        self.settings = settings
        super().__init__(*args, directory=str(settings.directory), **kwargs)

    def do_GET(self) -> None:
        if self.path.split("?")[0] in {"/", "/index.html"}:
            fresh(self.settings)
        super().do_GET()

    def end_headers(self) -> None:
        if not self.path.endswith(".js"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        return


class IPv6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def run(settings: MonitorCLI) -> None:
    fresh(settings)
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
