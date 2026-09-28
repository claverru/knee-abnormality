import json
import math
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import BaseModel, Field
from pydantic_settings import CliApp, CliSubCommand

from knee_abnormality.cli import CLISettings, command, describe
from knee_abnormality.data.labels import STUDY
from scripts.tools.prepare_rebuilt_labels import PrepareCLI, families, load_roots

NOT_MENTIONED = {"nm", "M", "not_mentioned", "unk"}
SKIPPED = {"U"}
POSITIVE_STATES = {"POS", "HEDGED"}
STATES = {"POS", "HEDGED", "LOW", "NEG", "NOT_MENTIONED"}
SEED = 20260928
PER_ROOT = 20
LOW_OTHERS, HIGH_OTHERS = 0.3, 0.7


class Paths(BaseModel):
    output_dir: Path = Field(
        default=Path("outputs/analysis/label-rebuild-20260928/source_audit"),
        description="Audit directory for the blind items, key, rates, judgments and scores",
    )


def sides(root, target: str) -> np.ndarray:
    values = root.values[target].to_numpy().astype(float)
    symbols = root.symbols[target].to_numpy().astype(object)
    silent = np.array(
        [
            s is not None and (s in NOT_MENTIONED or str(s).startswith("NOT_MENTIONED"))
            for s in symbols
        ]
    )
    skipped = np.array([s in SKIPPED for s in symbols])
    side = np.full(len(values), np.nan)
    side[np.isfinite(values) & (values >= 0.5)] = 1.0
    side[np.isfinite(values) & (values < 0.5)] = 0.0
    side[silent] = 0.0
    side[skipped] = np.nan
    return side


def wilson(successes: int, n: int, z: float = 1.2816) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


class Sample(Paths):
    model_config = describe(
        "Find each root's silver cells that contradict the other families' mean and draw a "
        "stratified random sample per root, writing blind items and a separate key"
    )

    def cli_cmd(self) -> None:
        official = pl.read_csv("data/raw/train.csv", infer_schema_length=None)
        targets = official.columns[2:]
        studies = official.select(STUDY).sort(STUDY)
        reports = studies.join(official.select(STUDY, "Report"), on=STUDY)
        silver = (
            studies.join(official.select(STUDY, "ACL"), on=STUDY)["ACL"]
            .is_null()
            .to_numpy()
        )
        roots = load_roots(
            studies, targets, PrepareCLI.model_fields["jevk5_attributes"].default
        )
        groups, _ = families(roots, targets, silver)
        family_of = {m: i for i, g in enumerate(groups) for m in g}
        rng = np.random.default_rng(SEED)
        rates, key = {}, []
        for index, root in enumerate(roots):
            candidates, rates[root.name] = [], {}
            for t in targets:
                stack = [
                    np.nanmean(
                        np.stack(
                            [roots[m].values[t].to_numpy().astype(float) for m in g]
                        ),
                        axis=0,
                    )
                    for i, g in enumerate(groups)
                    if i != family_of[index]
                ]
                others = np.nanmean(np.stack(stack), axis=0)
                side = sides(root, t)
                valid = silver & np.isfinite(side) & np.isfinite(others)
                disagree = valid & (
                    ((side == 1) & (others < LOW_OTHERS))
                    | ((side == 0) & (others > HIGH_OTHERS))
                )
                rates[root.name][t] = {
                    "disagreements": int(disagree.sum()),
                    "valid": int(valid.sum()),
                    "rate": float(disagree.sum() / max(valid.sum(), 1)),
                }
                candidates += [
                    (t, int(i), int(side[i]), float(others[i]))
                    for i in np.flatnonzero(disagree)
                ]
            if not candidates:
                continue
            counts = {t: rates[root.name][t]["disagreements"] for t in targets}
            total = sum(counts.values())
            quota = {t: PER_ROOT * c / total for t, c in counts.items()}
            alloc = {t: math.floor(q) for t, q in quota.items()}
            for t in sorted(quota, key=lambda t: quota[t] - alloc[t], reverse=True)[
                : min(PER_ROOT, total) - sum(alloc.values())
            ]:
                alloc[t] += 1
            for t in targets:
                pool = [c for c in candidates if c[0] == t]
                take = min(alloc[t], len(pool))
                for j in rng.choice(len(pool), size=take, replace=False):
                    _, i, side, others = pool[j]
                    key.append(
                        {
                            "root": root.name,
                            STUDY: studies[STUDY][i],
                            "target": t,
                            "root_side": side,
                            "others_mean": round(others, 4),
                        }
                    )
        self.output_dir.mkdir(parents=True, exist_ok=False)
        key_frame = pl.DataFrame(key)
        items = (
            key_frame.select(STUDY, "target")
            .unique()
            .sort(STUDY, "target")
            .with_row_index("item")
            .join(reports, on=STUDY)
        )
        key_frame.join(
            items.select("item", STUDY, "target"), on=[STUDY, "target"]
        ).write_csv(self.output_dir / "key.csv")
        items.select("item", "target", "Report").write_csv(
            self.output_dir / "items_blind.csv"
        )
        (self.output_dir / "disagreement_rates.json").write_text(
            json.dumps(rates, indent=2) + "\n"
        )
        print(
            json.dumps(
                {"sampled_cells": key_frame.height, "unique_items": items.height},
                indent=2,
            )
        )


class Score(Paths):
    model_config = describe(
        "Score each root's outlier precision from the blind judgments with Wilson 80% "
        "intervals and apply the pre-registered drop rule"
    )

    def cli_cmd(self) -> None:
        key = pl.read_csv(self.output_dir / "key.csv")
        judgments = pl.read_csv(self.output_dir / "judgments.csv")
        if not set(judgments["state"]).issubset(STATES):
            raise ValueError("Unknown judged state")
        joined = key.join(judgments.select("item", "state"), on="item", validate="m:1")
        if joined.height != key.height:
            raise ValueError("Some sampled items have no judgment")
        joined = joined.with_columns(
            (
                pl.col("state").is_in(list(POSITIVE_STATES)).cast(pl.Int64)
                == pl.col("root_side")
            ).alias("root_right")
        )
        rates = json.loads((self.output_dir / "disagreement_rates.json").read_text())
        result = {}
        for (root,), frame in joined.group_by("root", maintain_order=True):
            n, right = frame.height, int(frame["root_right"].sum())
            low, high = wilson(right, n)
            valid = sum(v["valid"] for v in rates[root].values())
            disagreements = sum(v["disagreements"] for v in rates[root].values())
            per_target = {}
            for (target,), part in frame.group_by("target", maintain_order=True):
                tn, tr = part.height, int(part["root_right"].sum())
                tl, th = wilson(tr, tn)
                per_target[target] = {
                    "n": tn,
                    "right": tr,
                    "precision": tr / tn,
                    "wilson80": [tl, th],
                    "drop": tn >= 8 and tr / tn < 0.35 and th < 0.5,
                }
            result[root] = {
                "n": n,
                "right": right,
                "precision": right / n,
                "wilson80": [low, high],
                "disagreement_rate": disagreements / valid,
                "positive_side_share": float(frame["root_side"].mean()),
                "precision_positive_side": float(
                    frame.filter(pl.col("root_side") == 1)["root_right"].mean()
                    or float("nan")
                ),
                "precision_negative_side": float(
                    frame.filter(pl.col("root_side") == 0)["root_right"].mean()
                    or float("nan")
                ),
                "drop": n >= 15 and right / n < 0.35 and high < 0.5,
                "per_target": per_target,
            }
        (self.output_dir / "scores.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        for root, r in sorted(result.items(), key=lambda x: x[1]["precision"]):
            print(
                f"{root:12} n={r['n']:2d} rate={r['disagreement_rate']:.3f} "
                f"precision={r['precision']:.2f} [{r['wilson80'][0]:.2f}, {r['wilson80'][1]:.2f}] "
                f"drop={r['drop']} target_drops={[t for t, v in r['per_target'].items() if v['drop']]}"
            )


class AuditCLI(CLISettings):
    model_config = describe(
        "Blind report audit of each label root's disagreements with the other source families"
    )

    sample: CliSubCommand[Sample] = command(Sample)
    score: CliSubCommand[Score] = command(Score)

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    np.seterr(all="ignore")
    CliApp.run(AuditCLI)


if __name__ == "__main__":
    main()
