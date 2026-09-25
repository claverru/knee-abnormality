import json
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import BaseModel, Field
from pydantic_settings import CliApp, CliSubCommand
from sklearn.metrics import roc_auc_score

from knee_abnormality.cli import CLISettings, command, describe
from knee_abnormality.data.labels import STUDY

TARGETS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)
LANGUAGE_WORDS = (
    ("turkish", (" ve ", "menisküs", "normal olarak", "izlen")),
    ("spanish", (" de la ", " del ", "menisco", "rodilla")),
    ("croatian", (" je ", "menisk", "zgloba", "uredn")),
    ("german", (" der ", " die ", " und ", "meniskus")),
    ("dutch", (" het ", " een ", "meniscus", "geen ")),
    ("french", (" le ", " les ", "ménisque", " du ")),
    ("english", (" the ", " of ", "meniscus", " is ")),
)
OSTEOPHYTE_WORDS = (
    r"osteof|osteoph|spur|osteofit|οστεόφυ|остеоф|osteofyt|randzack|exostos"
)
ARTHROSIS_WORDS = (
    r"arthro|artros|artroz|osteoarth|gonart|αρθρίτ|οστεοαρθρ|артроз|degenerative joint"
)
DEPTH_LOSS = ("unspecified_loss", "partial_low", "partial_high", "full_thickness")
DEPTH_HIGH = ("partial_high", "full_thickness", "bone_on_bone")


class Paths(BaseModel):
    attributes: Path = Field(
        default=Path("outputs/labels/jevk5-attributes-v3/attributes.parquet"),
        description="Extraction output",
    )
    question_set: Path = Field(
        default=Path("configs/labels/jevk5-report-attributes-v3.json"),
        description="Question set the extraction used",
    )
    reports: Path = Field(default=Path("data/raw/train.csv"), description="Reports")
    targets: Path = Field(
        default=Path("data/labels/prepared/combined-v1/labels.csv"),
        description="Training targets to compare with",
    )


def language(text: str) -> str:
    low = text.lower()
    if any("Ͱ" <= ch <= "Ͽ" or ch == "∆" for ch in text):
        return "greek"
    if any("Ѐ" <= ch <= "ӿ" for ch in text):
        return "bulgarian"
    for name, words in LANGUAGE_WORDS:
        if sum(word in low for word in words) >= 2:
            return name
    return "other"


def answers(long: pl.DataFrame) -> pl.DataFrame:
    def top(column: str) -> pl.DataFrame:
        return (
            long.sort(column, descending=True)
            .group_by(STUDY, "question", maintain_order=True)
            .agg(
                pl.col("option").first().alias(column),
                pl.col(column).first().alias(f"{column}_p"),
            )
        )

    return (
        top("probability")
        .join(top("forward"), on=[STUDY, "question"])
        .join(top("reverse"), on=[STUDY, "question"])
    )


def wide(long: pl.DataFrame) -> pl.DataFrame:
    return long.with_columns(
        pl.format("{}|{}", "question", "option").alias("column")
    ).pivot(on="column", index=STUDY, values="probability")


def mass(question: str, options: tuple[str, ...]) -> pl.Expr:
    return pl.sum_horizontal(pl.col(f"{question}|{option}") for option in options)


def show(title: str, frame: pl.DataFrame) -> None:
    print(f"\n### {title}")
    with pl.Config(tbl_rows=80, tbl_width_chars=200, fmt_str_lengths=60):
        print(frame)


class Answers(Paths):
    model_config = describe(
        "Label-free checks over every answer: option usage, position bias between the two "
        "orders, per-question agreement and confidence, parent/child consistency, language, "
        "osteophyte and arthrosis words behind order flips, and agreement with the training "
        "targets by extracted grade"
    )

    def cli_cmd(self) -> None:
        long = pl.read_parquet(self.attributes)
        questions = json.loads(self.question_set.read_text())["questions"]
        position = {
            (q["id"], o): i for q in questions for i, o in enumerate(q["options"])
        }
        count = {q["id"]: len(q["options"]) for q in questions}
        long = long.with_columns(
            pl.struct("question", "option")
            .map_elements(
                lambda r: position[(r["question"], r["option"])], return_dtype=pl.Int64
            )
            .alias("position"),
            pl.col("question").replace_strict(count, return_dtype=pl.Int64).alias("k"),
        )
        top = answers(long)
        chosen = (
            top.group_by("question", "probability")
            .len()
            .rename({"probability": "option", "len": "chosen"})
        )
        usage = (
            long.group_by("question", "option")
            .agg(pl.col("probability").max().alias("max_p"))
            .join(chosen, on=["question", "option"], how="left")
            .with_columns(pl.col("chosen").fill_null(0))
        )
        show(
            "Options chosen fewer than 10 times",
            usage.filter(pl.col("chosen") < 10).sort("question", "option"),
        )
        bias = long.group_by("question", "option", "position", "k").agg(
            (pl.col("forward") - pl.col("reverse"))
            .mean()
            .alias("forward_minus_reverse")
        )
        show(
            "Position bias above 0.03",
            bias.filter(pl.col("forward_minus_reverse").abs() > 0.03).sort(
                pl.col("forward_minus_reverse").abs(), descending=True
            ),
        )
        entropy = (
            long.with_columns(
                (
                    -pl.col("probability") * pl.col("probability").clip(1e-12).log()
                ).alias("h")
            )
            .group_by(STUDY, "question")
            .agg(pl.col("h").sum())
        )
        show(
            "Per question",
            top.join(entropy, on=[STUDY, "question"])
            .group_by("question")
            .agg(
                (pl.col("forward") == pl.col("reverse"))
                .mean()
                .alias("order_agreement"),
                pl.col("probability_p").mean().alias("mean_top_p"),
                (pl.col("probability_p") < 0.5).mean().alias("low_confidence"),
                pl.col("h").mean().alias("entropy"),
            )
            .sort("order_agreement"),
        )
        show(
            "Largest order flips",
            top.filter(pl.col("forward") != pl.col("reverse"))
            .group_by("question", "forward", "reverse")
            .len()
            .sort("len", descending=True)
            .head(25),
        )
        values = wide(long)
        pairs = {
            "medial tear: status vs location": (
                mass(
                    "medial_meniscus_status",
                    ("possible_tear", "tear", "displaced_or_truncated"),
                ),
                1 - pl.col("medial_meniscus_tear_location|no_tear"),
            ),
            "lateral tear: status vs location": (
                mass(
                    "lateral_meniscus_status",
                    ("possible_tear", "tear", "displaced_or_truncated"),
                ),
                1 - pl.col("lateral_meniscus_tear_location|no_tear"),
            ),
            "contusion: status vs location": (
                mass("contusion_status", ("contusion", "possible")),
                1 - pl.col("contusion_location|none"),
            ),
            "Baker's: size vs rupture": (
                mass("baker_size", ("small", "moderate", "large", "unspecified")),
                1 - pl.col("baker_ruptured|no_cyst"),
            ),
            "MCL: status vs timing": (
                1 - mass("mcl_status", ("not_mentioned", "intact")),
                1 - pl.col("mcl_timing|no_injury"),
            ),
            "medial cartilage: depth vs extent": (
                mass("medial_cartilage_depth", (*DEPTH_LOSS, "bone_on_bone")),
                mass(
                    "medial_cartilage_extent",
                    ("focal", "large", "diffuse", "not_stated"),
                ),
            ),
            "PF cartilage: depth vs surface": (
                mass("pf_cartilage_depth", (*DEPTH_LOSS, "bone_on_bone")),
                1 - pl.col("pf_cartilage_surface|no_damage"),
            ),
        }
        rows = []
        for name, (parent, child) in pairs.items():
            d = values.select(parent.alias("parent"), child.alias("child"))
            rows.append(
                {
                    "pair": name,
                    "parent_only": d.filter(
                        (pl.col("parent") > 0.5) & (pl.col("child") < 0.5)
                    ).height,
                    "child_only": d.filter(
                        (pl.col("parent") < 0.5) & (pl.col("child") > 0.5)
                    ).height,
                    "both": d.filter(
                        (pl.col("parent") > 0.5) & (pl.col("child") > 0.5)
                    ).height,
                }
            )
        show("Parent/child consistency", pl.DataFrame(rows))
        reports = pl.read_csv(self.reports, infer_schema_length=0).select(
            STUDY, pl.col("Report").fill_null("")
        )
        reports = reports.with_columns(
            pl.col("Report")
            .map_elements(language, return_dtype=pl.String)
            .alias("language"),
            pl.col("Report").str.len_chars().alias("chars"),
        )
        by_language = top.join(reports, on=STUDY)
        show(
            "By language",
            by_language.group_by("language")
            .agg(
                (pl.col("forward") == pl.col("reverse"))
                .mean()
                .alias("order_agreement"),
                pl.col("probability_p").mean().alias("mean_top_p"),
                pl.col(STUDY).n_unique().alias("reports"),
            )
            .sort("order_agreement"),
        )
        lowered = reports.with_columns(pl.col("Report").str.to_lowercase())
        for question, forward, reverse, words in (
            ("medial_osteophytes", "unspecified", "none", OSTEOPHYTE_WORDS),
            ("overall_osteoarthritis", "not_mentioned", "none", ARTHROSIS_WORDS),
        ):
            flips = top.filter(
                pl.col("question") == question,
                pl.col("forward") == forward,
                pl.col("reverse") == reverse,
            ).join(lowered, on=STUDY)
            show(
                f"{question} flips {forward}/{reverse}: report names the finding",
                flips.group_by(
                    pl.col("Report").str.contains(words).alias("named")
                ).len(),
            )
        targets = pl.read_csv(self.targets, infer_schema_length=None)
        for label, question in (
            ("Effusion", "effusion_size"),
            ("Baker's", "baker_size"),
            ("Medial OA", "medial_cartilage_depth"),
        ):
            d = (
                top.filter(pl.col("question") == question)
                .select(STUDY, pl.col("probability").alias("answer"))
                .join(targets.select(STUDY, label), on=STUDY)
                .drop_nulls(label)
            )
            show(
                f"Training {label} target by extracted {question}",
                d.group_by("answer")
                .agg(
                    pl.len(),
                    pl.col(label).mean().alias("mean_target"),
                    (pl.col(label) >= 0.5).mean().alias("positive_share"),
                )
                .sort("mean_target"),
            )


class Gold(Paths):
    model_config = describe(
        "Gold labels of the official training studies against the extracted effusion, "
        "Baker's and cartilage grades, and gold AUC of an any-finding rule against an "
        "official-threshold rule built from the same answers"
    )

    def cli_cmd(self) -> None:
        raw = pl.read_csv(self.reports, infer_schema_length=None)
        gold = raw.filter(pl.col("ACL").is_not_null()).select(STUDY, *TARGETS)
        long = pl.read_parquet(self.attributes)
        top = (
            answers(long)
            .select(STUDY, "question", "probability")
            .pivot(on="question", index=STUDY, values="probability")
        )
        joined = gold.join(top, on=STUDY).join(wide(long), on=STUDY)
        print("gold studies", joined.height)
        for label, question in (
            ("Effusion", "effusion_size"),
            ("Baker's", "baker_size"),
            ("Medial OA", "medial_cartilage_depth"),
            ("Lateral OA", "lateral_cartilage_depth"),
            ("PF OA", "pf_cartilage_depth"),
        ):
            show(
                f"Gold {label} by extracted {question}",
                joined.group_by(question)
                .agg(pl.len().alias("n"), pl.col(label).sum().alias("gold_positive"))
                .sort(question),
            )
        cartilage = {"Medial OA": "medial", "Lateral OA": "lateral", "PF OA": "pf"}
        loose = {
            "Effusion": mass(
                "effusion_size", ("small", "moderate", "large", "unspecified")
            ),
            "Baker's": mass(
                "baker_size", ("small", "moderate", "large", "unspecified")
            ),
            **{
                label: 1
                - mass(
                    f"{key}_cartilage_depth",
                    (
                        "not_mentioned",
                        "normal",
                        *(("only_elsewhere",) if key != "pf" else ()),
                    ),
                )
                for label, key in cartilage.items()
            },
        }
        strict = {
            "Effusion": mass("effusion_size", ("moderate", "large"))
            + 0.5 * pl.col("effusion_size|unspecified"),
            "Baker's": mass("baker_size", ("moderate", "large"))
            + 0.5 * pl.col("baker_size|unspecified"),
            **{
                label: mass(f"{key}_cartilage_depth", DEPTH_HIGH)
                + 0.5 * pl.col(f"{key}_cartilage_depth|unspecified_loss")
                for label, key in cartilage.items()
            },
        }
        print("\ngold AUC: any finding vs official threshold")
        for label, any_finding in loose.items():
            scores = joined.select(
                any_finding.alias("loose"), strict[label].alias("strict"), pl.col(label)
            )
            y = scores[label].to_numpy()
            print(
                f"{label:11s} positives {int(y.sum()):2d}/{len(y)}  any finding {roc_auc_score(y, scores['loose'].to_numpy()):.3f}  official threshold {roc_auc_score(y, scores['strict'].to_numpy()):.3f}"
            )


class AnalyzeCLI(CLISettings):
    model_config = describe(
        "Analyze JevK5 report-attribute answers without re-extraction"
    )

    answers: CliSubCommand[Answers] = command(Answers)
    gold: CliSubCommand[Gold] = command(Gold)

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    np.seterr(all="ignore")
    CliApp.run(AnalyzeCLI)


if __name__ == "__main__":
    main()
