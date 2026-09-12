import html
import json
import platform
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from knee_abnormality.cli import CLISettings
from knee_abnormality.provenance import sha256_file


class Settings(CLISettings):
    input: Path = Path("data/raw/train.csv")
    output_dir: Path = Path("data/metadata/report_normalization")


HEADINGS = (
    "findings", "imaging findings", "results", "impression", "impressions",
    "conclusion", "conclusions", "coclusion", "interpretation", "hallazgos",
    "resultados", "impresion", "conclusiones", "bulgular", "sonuc",
    "bevindingen", "besluit", "conclusie", "befund", "beurteilung",
    "constatations", "resultats", "trouvailles", "мр находка", "заключение",
    "ευρηματα", "συμπερασμα",
)
FINDINGS = (
    "findings", "imaging findings", "results", "hallazgos", "resultados",
    "bulgular", "bevindingen", "befund", "constatations", "resultats",
    "trouvailles", "мр находка", "ευρηματα",
)
PLACEHOLDERS = re.compile(
    r"\[(?:date|time|redacted|id|name|age|year|month|identifier|gender|profession|"
    r"affiliation|day|title|bilateral note[^\]]*)\]"
)
FINDINGS_START = re.compile(
    r"(?:^|(?<=[\n.!?]))\s*(?:" + "|".join(FINDINGS) + r")\s*:"
)
HEADING = re.compile(r"\b(?:" + "|".join(HEADINGS) + r")\s*:")
ADMIN_LINE = re.compile(
    r"(?m)^\s*(?:signed by|signed on|electronically signed|dictated by|reported by|"
    r"exam date and time|exam type|accession number)\b[^\n]*"
)
LATERALITY = re.compile(
    r"\b(?:left|right|lt|rt|izquierd[oa]s?|derech[oa]s?|links|rechts|linken|"
    r"rechten|gauche|droit[e]?|sol|sag|дясна(?:та)?|дясно(?:то)?|лява(?:та)?|"
    r"ляво(?:то)?|αριστερου|δεξιου|αριστερο|δεξιο)\b"
)
REPLACEMENTS = (
    (r"\b(?:anterior cruciate ligaments?|a\s+c\s+l)\b(?:\s*\(acl\))?", "acl"),
    (r"\b(?:posterior cruciate ligaments?|p\s+c\s+l)\b(?:\s*\(pcl\))?", "pcl"),
    (r"\b(?:medial collateral ligaments?|m\s+c\s+l)\b(?:\s*\(mcl\))?", "mcl"),
    (r"\b(?:lateral collateral ligaments?|l\s+c\s+l)\b(?:\s*\(lcl\))?", "lcl"),
    (r"\b(?:magnetic resonance imaging|mri|rmn|mrg|irm)\b", "mr"),
    (r"\boedema\b", "edema"),
    (r"\bhaemarthrosis\b", "hemarthrosis"),
    (r"\bmenisci\b", "meniscus"),
    (r"\bmeniscuses\b", "meniscus"),
    (r"\bpatellae\b", "patella"),
    (r"\b(?:baker['’]?s|bakers)\b", "baker"),
    (r"\b(?:hoffa['’]?s|hoffas)\b", "hoffa"),
    (r"\b(?:femoro[ -]?tibial|tibio[ -]?femoral)\b", "femorotibial"),
    (r"\bpatello[ -]?femoral\b", "patellofemoral"),
    (r"\b(?:cannot|can['’]t)\b", "can not"),
    (r"\b(?:isn['’]t|aren['’]t|wasn['’]t|weren['’]t)\b", "not"),
    (r"\b(?:doesn['’]t|don['’]t|didn['’]t)\b", "not"),
    (r"\b(?:grade|grado|stage|stade|grad)\s+[ivx]+\b", "grade number"),
)
REPLACEMENTS = tuple((re.compile(pattern), value) for pattern, value in REPLACEMENTS)
STOPWORDS = frozenset(
    [
        "a", "an", "the", "and", "or", "of", "to", "in", "at", "by", "for", "from",
        "with", "as", "is", "are", "was", "were", "be", "been", "being", "this", "that",
        "these", "those", "there", "it", "its", "which", "within", "also",
        "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "en",
        "y", "o", "con", "por", "para", "se", "es", "son",
        "le", "les", "des", "du", "au", "aux", "une", "et", "ou", "dans", "avec", "sur",
        "est", "sont", "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer",
        "einem", "einen", "und", "oder", "im", "am", "mit", "von", "zu", "zum", "zur",
        "ist", "sind", "de", "het", "een", "van", "en", "of", "in", "op", "met", "voor",
        "aan", "is", "zijn", "ve", "ile", "bir", "olarak",
        "и", "в", "във", "на", "за", "от", "с", "със", "по", "при",
        "και", "του", "της", "των", "το", "τα", "την", "τον", "σε", "με",
    ]
)
TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)


def unicode_text(text: str) -> str:
    text = html.unescape(text).casefold().replace("ı", "i").replace("œ", "oe")
    text = text.replace("æ", "ae").replace("ø", "o").replace("ł", "l")
    text = "".join(
        char for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char) and unicodedata.category(char) != "Cf"
    )
    return text.replace("\r\n", "\n").replace("\r", "\n")


def heavy_text(text: str) -> str:
    text = PLACEHOLDERS.sub(" ", unicode_text(text))
    heading = FINDINGS_START.search(text)
    if heading is not None:
        text = text[heading.end():]
    text = ADMIN_LINE.sub(" ", text)
    text = HEADING.sub(". ", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s*", "", text)
    for pattern, value in REPLACEMENTS:
        text = pattern.sub(value, text)
    text = re.sub(r"\btechnique\s*:\s*mr\s+(?:of\s+the\s+)?knee\s*\.", " ", text)
    text = re.sub(r"\b\d+(?:[.,]\d+)*(?:\s*[-–/]\s*\d+(?:[.,]\d+)*)?\b", " number ", text)
    text = re.sub(
        r"\bnumber(?:\s*[x×]\s*number)*\s*(?:mm|cm|millimet(?:er|re)s?|centimet(?:er|re)s?)\b",
        " measurement ", text,
    )
    text = LATERALITY.sub(" ", text)
    text = re.sub(r"['’]s\b", "", text)
    text = re.sub(r"(?<=\w)[-‐‑–](?=\w)", "", text)
    text = text.replace("&", " and ")
    clauses = []
    for clause in re.split(r"[.!?;]+|\n\s*\n", text):
        tokens = [token for token in TOKEN.findall(clause) if token not in STOPWORDS]
        tokens = [token for i, token in enumerate(tokens) if not i or token != tokens[i - 1]]
        if tokens:
            clauses.append(" ".join(tokens))
    return " ; ".join(dict.fromkeys(clauses))


def stage_summary(frame: pl.DataFrame, column: str) -> dict:
    nonempty = frame.filter(pl.col(column).str.strip_chars() != "")
    groups = nonempty.group_by(column).len()
    duplicates = groups.filter(pl.col("len") > 1)
    return {
        "unique_nonempty": groups.height,
        "empty_rows": frame.height - nonempty.height,
        "duplicate_groups": duplicates.height,
        "rows_in_duplicate_groups": int(duplicates["len"].sum()),
        "redundant_rows": nonempty.height - groups.height,
        "largest_group": int(groups["len"].max() or 0),
    }


def main() -> None:
    settings = Settings()
    frame = pl.read_csv(settings.input, columns=["StudyInstanceUID", "Report"])
    if frame["Report"].null_count():
        raise ValueError("Report contains null values")
    frame = frame.with_columns(
        pl.col("Report").map_elements(
            lambda text: " ".join(text.casefold().split()), return_dtype=pl.String,
        ).alias("case_whitespace"),
        pl.col("Report").map_elements(
            lambda text: " ".join(re.findall(r"[^\W_]+", unicode_text(text))),
            return_dtype=pl.String,
        ).alias("unicode_punctuation"),
        pl.col("Report").map_elements(heavy_text, return_dtype=pl.String).alias("heavy"),
    ).with_columns(
        pl.col("heavy").map_elements(
            lambda text: " ; ".join(sorted(set(text.split(" ; ")))),
            return_dtype=pl.String,
        ).alias("sentence_set"),
        pl.col("heavy").map_elements(
            lambda text: " ".join(sorted(set(TOKEN.findall(text)))), return_dtype=pl.String,
        ).alias("token_set"),
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = settings.output_dir / "normalized_reports.parquet"
    frame.write_parquet(normalized_path)
    groups_path = settings.output_dir / "duplicate_groups.csv"
    frame.filter(pl.col("token_set") != "").group_by("token_set").agg(
        pl.len().alias("rows"),
        pl.col("Report").n_unique().alias("raw_variants"),
        pl.col("StudyInstanceUID").sort().str.join("|").alias("study_ids"),
    ).filter(pl.col("rows") > 1).sort(
        ["rows", "token_set"], descending=[True, False],
    ).write_csv(groups_path)
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "input": str(settings.input),
        "input_sha256": sha256_file(settings.input),
        "script_sha256": sha256_file(Path(__file__)),
        "settings": settings.model_dump(mode="json"),
        "python_version": platform.python_version(),
        "polars_version": pl.__version__,
        "unicode_version": unicodedata.unidata_version,
        "seed": None,
        "rows": frame.height,
        "stages": {
            column: stage_summary(frame, column)
            for column in (
                "Report", "case_whitespace", "unicode_punctuation", "heavy",
                "sentence_set", "token_set",
            )
        },
        "artifacts": {
            str(path): sha256_file(path) for path in (normalized_path, groups_path)
        },
        "limitations": [
            "Exact equality after deterministic lexical transformations; no fuzzy clustering or translation.",
            "Heavy normalization discards preambles, numeric values, laterality and selected function words.",
            "Negation words are retained, but sentence_set and token_set discard order; token_set also discards multiplicity and negation scope.",
            "Empty normalized reports are excluded from distinct-text and duplicate-group counts.",
            "Groups do not establish identical clinical meaning, patients or images.",
        ],
    }
    (settings.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
