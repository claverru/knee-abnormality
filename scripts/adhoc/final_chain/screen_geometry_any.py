import sys
from pathlib import Path

import polars as pl
import pytorch_lightning as L
from torch.utils.data import DataLoader, Dataset

from knee_abnormality.config import load_config
from knee_abnormality.training.lightning import KneeDataModule

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
TABLE = Path(sys.argv[2]) / "placeholder_table.parquet"
OUT = Path(sys.argv[2]) / "geometry_screen.csv"
studies = pl.read_csv(Path(sys.argv[1]) / "studies.csv").select("StudyInstanceUID").unique().sort("StudyInstanceUID")
pl.concat([studies.select("StudyInstanceUID", pl.lit(fold, dtype=pl.Int64).alias("fold"), *(pl.lit(0.5).alias(name) for name in TARGETS)) for fold in range(5)]).write_parquet(TABLE)
config = load_config(
    Path("configs/training.yaml").resolve(),
    overrides={"validation": {"folds": [0]}, "extra_training": {"data_root": sys.argv[1], "labels": str(TABLE)}},
)
L.seed_everything(config.seed, workers=True)
data = KneeDataModule(config, 0, limit_train=64)
data.setup("fit")
dataset = data.extra_dataset
ids = dataset.studies["StudyInstanceUID"].to_list()


class Screen(Dataset):
    def __len__(self) -> int:
        return len(ids)

    def __getitem__(self, index: int) -> tuple:
        item = dataset[index]
        series = [value for group in item["series"] for value in group]
        valid = [bool(value.geometry_valid.all()) for value in series]
        return ids[index], len(series), sum(valid), float(item["knee_side"].value != 0)


rows = list(DataLoader(Screen(), batch_size=None, num_workers=48))
frame = pl.DataFrame(rows, schema=["StudyInstanceUID", "series", "valid_series", "side_known"], orient="row")
frame.write_csv(OUT)
bad = frame.filter(pl.col("valid_series") == 0)
print("studies", frame.height, "none valid", bad.height, "by series count", bad["series"].value_counts().sort("series").rows(), flush=True)
