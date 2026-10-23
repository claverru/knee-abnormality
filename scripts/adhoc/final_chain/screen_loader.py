import sys
import time
from pathlib import Path

import polars as pl
from torch.utils.data import DataLoader, Dataset

from knee_abnormality.config import load_config
from knee_abnormality.inference.engine import prediction_data

ROOT = Path(sys.argv[2])
OUT = Path(sys.argv[1])
config = load_config(
    Path("configs/training.yaml").resolve(), data_root=ROOT, label_manifest=None
)
dataset, _, studies = prediction_data(config, "test", 1, image_multiplier=1)
ids = studies["StudyInstanceUID"].to_list()


class Screen(Dataset):
    def __len__(self) -> int:
        return len(ids)

    def __getitem__(self, index: int) -> tuple:
        try:
            item = dataset[index]
        except Exception as error:
            return ids[index], f"{type(error).__name__}: {error}"[:160], 0, 0
        series = sum(len(group) for group in item["series"])
        return ids[index], "ok", series, len(item["failures"])


started = time.perf_counter()
rows = list(DataLoader(Screen(), batch_size=None, num_workers=48))
frame = pl.DataFrame(
    rows,
    schema=["StudyInstanceUID", "status", "prepared_series", "conditions"],
    orient="row",
)
frame.write_csv(OUT / "screen.csv")
bad = frame.filter(pl.col("status") != "ok")
print(
    f"screened {frame.height} studies in {time.perf_counter() - started:.0f} s; rejected {bad.height}; with partial conditions {frame.filter(pl.col('conditions') > 0).height}"
)
print(
    bad.with_columns(reason=pl.col("status").str.replace(r"Study \S+ ", ""))["reason"]
    .value_counts(sort=True)
    .rows()
)
table = pl.read_csv(ROOT / "studies.csv").join(frame, on="StudyInstanceUID", how="left")
print(
    "rejected by part",
    table.filter(pl.col("status") != "ok").group_by("part").len().rows(),
)
print(
    "prepared series per study among accepted",
    table.filter(pl.col("status") == "ok")["prepared_series"]
    .value_counts()
    .sort("prepared_series")
    .rows(),
)
