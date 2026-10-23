from pathlib import Path

import polars as pl

SOURCES = {"fastmri": Path("/workspace/data/fastmri"), "oai": Path("/workspace/data/oai")}
OUT = Path("/workspace/data/external")
EXCLUDED = {
    "fastmri": Path("/workspace/data/labels/prepared/ensemble10-avg-labels-x4-tau085-tta/excluded_from_training.csv"),
    "oai": Path("/workspace/data/oai/excluded_from_training.csv"),
}
raw = OUT / "raw"
(raw / "test_series").mkdir(parents=True, exist_ok=True)
tables, submissions, studies, excluded = [], [], [], []
for name, root in SOURCES.items():
    for study in sorted((root / "raw" / "test_series").iterdir()):
        link = raw / "test_series" / study.name
        if not link.exists():
            link.symlink_to(study.resolve())
    tables.append(pl.read_csv(root / "raw" / "test_series.csv"))
    submissions.append(pl.read_csv(root / "raw" / "sample_submission.csv"))
    studies.append(pl.read_csv(root / "studies.csv").select("StudyInstanceUID").with_columns(pl.lit(name).alias("source")))
    excluded.append(pl.read_csv(EXCLUDED[name]).select("StudyInstanceUID").with_columns(pl.lit(name).alias("source")))
table = pl.concat(tables)
table.write_csv(raw / "test_series.csv")
table.write_csv(raw / "train_series.csv")
if not (raw / "train_series").exists():
    (raw / "train_series").symlink_to("test_series")
pl.concat(submissions).write_csv(raw / "sample_submission.csv")
all_studies = pl.concat(studies)
all_studies.write_csv(OUT / "studies.csv")
pl.concat(excluded).write_csv(OUT / "excluded_from_training.csv")
assert all_studies["StudyInstanceUID"].n_unique() == all_studies.height
print("studies", all_studies.group_by("source").len().rows(), "series", table.height, "links", len(list((raw / "test_series").iterdir())), "excluded", pl.concat(excluded).group_by("source").len().rows())
