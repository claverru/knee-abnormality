import json
import re
from pathlib import Path

import polars as pl
import yaml
from pydantic import Field
from scipy.stats import mannwhitneyu

from knee_abnormality.cli import CLISettings, describe

STAMP = re.compile(r"(\d{8}T\d{6}Z)-[0-9a-f]{8}$")
REFERENCE = re.compile(r'"reference": "(/[^"]*)"')
SEED_ID = re.compile(r"^\s*-?\s*id:\s*(\S+)", re.M)


class CampaignScreeningCLI(CLISettings):
    model_config = describe(
        "Check whether early-epoch validation deltas predict final autoresearch verdicts "
        "and where in each campaign adoptions arrived"
    )

    outputs: Path = Field(default=Path("outputs/autoresearch"), description="Campaign outputs root")
    campaigns: Path = Field(default=Path("autoresearch/campaigns"), description="Campaign definitions root")
    outlier: float = Field(default=0.01, gt=0, description="Final |delta| excluded from correlations")
    kill_epochs: list[int] = Field(default=[4, 5, 6, 7], description="Epochs tested for an abort rule")
    kill_thresholds: list[float] = Field(
        default=[-0.003, -0.005, -0.01], description="Epoch deltas below which a run is aborted"
    )
    streaks: list[int] = Field(
        default=[3, 4, 5, 6, 7, 8, 9, 10],
        description="Consecutive non-adopted researcher runs after which a campaign stops",
    )
    caps: list[int] = Field(default=[8, 9, 10, 11, 12, 13, 14], description="Total runs per campaign")
    gain: float = Field(default=0.001, description="Final delta counted as clearing the adoption gain")


def epoch_curve(run: Path) -> dict[int, float]:
    for root in (run, run / "training"):
        files = sorted(root.glob("fold_*/logs/version_*/metrics.csv"))
        if files:
            frame = pl.read_csv(files[-1], infer_schema_length=0).filter(
                pl.col("val_macro_auc").is_not_null() & (pl.col("val_macro_auc") != "")
            )
            return {int(e): float(v) for e, v in frame.select("epoch", "val_macro_auc").iter_rows()}
    return {}


def seed_ids(campaigns: Path, campaign: str) -> set[str]:
    return {i for f in (campaigns / campaign).glob("seed-ideas*.yaml") for i in SEED_ID.findall(f.read_text())}


def lineage(campaigns: Path) -> dict[tuple[str, str], str]:
    kinds = {}
    for path in campaigns.glob("*/ideas.yaml"):
        for idea in yaml.safe_load(path.read_text()).get("ideas") or []:
            kinds[(path.parent.name, idea["id"])] = "follow_up" if idea.get("parent_idea_id") else "new_family"
    return kinds


def replay(frame: pl.DataFrame, gain: float, stop) -> dict[str, int]:
    saved = lost = lost_gain = 0
    for rows in frame.sort("campaign", "stamp").partition_by("campaign", maintain_order=True):
        streak = 0
        for index, row in enumerate(rows.iter_rows(named=True)):
            if stop(index, streak):
                rest = rows.slice(index)
                saved += rest.height
                lost += rest.filter(pl.col("decision") == "provisional_accept").height
                lost_gain += rest.filter(pl.col("final") >= gain).height
                break
            if row["decision"] == "provisional_accept":
                streak = 0
            elif row["origin"] == "researcher":
                streak += 1
    return {"runs": frame.height, "saved": saved, "adoptions_lost": lost, "gain_runs_lost": lost_gain}


def collect(args: CampaignScreeningCLI) -> pl.DataFrame:
    rows = []
    for evaluation in args.outputs.glob("*/experiments/*/evaluation.json"):
        experiment = evaluation.parent
        record = json.loads(evaluation.read_text())
        stamp = STAMP.search(experiment.name)
        comparison = experiment / "comparison" / "comparison.json"
        if "macro_auc" not in record or not stamp or not comparison.exists():
            continue
        campaign = experiment.parent.parent.name
        idea = experiment.name[: stamp.start() - 1]
        reference = REFERENCE.search(comparison.read_text())
        candidate_curve = epoch_curve(experiment / "training")
        reference_curve = epoch_curve(Path(reference.group(1))) if reference else {}
        deltas = {f"epoch_{e}": v - reference_curve[e] for e, v in candidate_curve.items() if e in reference_curve}
        rows.append(
            {
                "campaign": campaign,
                "stamp": stamp.group(1),
                "idea": idea,
                "origin": "seed" if idea in seed_ids(args.campaigns, campaign) else "researcher",
                "decision": record["eligible_decision"],
                "final": record["macro_auc"]["candidate"] - record["macro_auc"]["reference"],
                **deltas,
            }
        )
    return (
        pl.DataFrame(rows)
        .sort("campaign", "stamp")
        .with_columns(position=pl.int_range(1, pl.len() + 1).over("campaign"))
        .with_columns(origin_position=pl.int_range(1, pl.len() + 1).over("campaign", "origin"))
    )


def run(args: CampaignScreeningCLI) -> None:
    frame = collect(args)
    epochs = sorted(int(c.split("_")[1]) for c in frame.columns if c.startswith("epoch_"))
    curves = frame.drop_nulls([f"epoch_{e}" for e in epochs])
    inliers = curves.filter(pl.col("final").abs() < args.outlier)
    correlation = [
        {
            "epoch": e,
            "n": inliers.height,
            "spearman": inliers.select(pl.corr(f"epoch_{e}", "final", method="spearman")).item(),
            "pearson": inliers.select(pl.corr(f"epoch_{e}", "final")).item(),
        }
        for e in epochs
    ]
    kills = []
    for e in args.kill_epochs:
        for threshold in args.kill_thresholds:
            killed = curves.filter(pl.col(f"epoch_{e}") < threshold)
            kills.append(
                {
                    "epoch": e,
                    "threshold": threshold,
                    "runs": curves.height,
                    "killed": killed.height,
                    "killed_final_positive": killed.filter(pl.col("final") > 0).height,
                    "killed_adopted": killed.filter(pl.col("decision") == "provisional_accept").height,
                    "epochs_saved_fraction": killed.height * (epochs[-1] - e) / ((epochs[-1] + 1) * curves.height),
                }
            )
    campaigns = (
        frame.group_by("campaign")
        .agg(
            runs=pl.len(),
            seeds=(pl.col("origin") == "seed").sum(),
            researcher=(pl.col("origin") == "researcher").sum(),
            adoptions=pl.struct("position", "origin", "origin_position", "idea").filter(
                pl.col("decision") == "provisional_accept"
            ),
        )
        .sort("campaign")
    )
    stops = [{"streak": n, **replay(frame, args.gain, lambda i, s, n=n: s >= n)} for n in args.streaks]
    stops += [{"cap": n, **replay(frame, args.gain, lambda i, s, n=n: i >= n)} for n in args.caps]
    kinds = lineage(args.campaigns)
    researcher = frame.filter(pl.col("origin") == "researcher").with_columns(
        kind=pl.struct("campaign", "idea").map_elements(
            lambda r: kinds.get((r["campaign"], r["idea"]), "unknown"), return_dtype=pl.Utf8
        )
    )
    by_kind = researcher.group_by("kind").agg(
        runs=pl.len(),
        adoptions=(pl.col("decision") == "provisional_accept").sum(),
        gain_runs=(pl.col("final") >= args.gain).sum(),
        median=pl.col("final").median(),
    )
    follow_up, new_family = (researcher.filter(pl.col("kind") == k)["final"].to_list() for k in ("follow_up", "new_family"))
    by_origin = frame.group_by("origin").agg(
        runs=pl.len(), adoptions=(pl.col("decision") == "provisional_accept").sum()
    )
    print(json.dumps(
        {
            "decisions": dict(frame.group_by("decision").len().iter_rows()),
            "curve_runs": curves.height,
            "correlation": correlation,
            "kill_rules": kills,
            "adoptions_by_origin": by_origin.to_dicts(),
            "stop_rules": stops,
            "researcher_lineage": by_kind.sort("kind").to_dicts(),
            "follow_up_vs_new_family_mannwhitney_p": float(mannwhitneyu(follow_up, new_family).pvalue),
            "campaigns": campaigns.to_dicts(),
        },
        indent=1,
    ))


if __name__ == "__main__":
    run(CampaignScreeningCLI())
