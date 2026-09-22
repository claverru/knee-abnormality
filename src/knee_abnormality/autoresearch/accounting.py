import hashlib
import json
import os
import time
from pathlib import Path

import pytorch_lightning as L

from knee_abnormality.autoresearch.storage import read_json, write_json


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class TripletLedger:
    def __init__(self, destination: Path, limits: dict | None = None):
        self.destination = destination
        self.limits = limits
        self.rows = {}
        self.history = {"train": [], "validation": []}
        self.first = {}
        self.changed_training_studies = set()

    def begin(self):
        self.rows = {}

    def observe(self, batch):
        counts = batch["slice_mask"].sum(dim=1).cpu().tolist()
        indices = batch["series_study_index"].cpu().tolist()
        types = batch["series_type_index"].cpu().tolist()
        if batch["valid_rects"].shape[2] != 3:
            raise ValueError("A valid input must contain exactly three slices")
        for index, uid in enumerate(batch["study_uid"]):
            if uid in self.rows:
                raise ValueError(f"Duplicate study visit within one epoch: {uid}")
            entries = [entry for entry, study in enumerate(indices) if study == index]
            self.rows[uid] = {
                "triplets": sum(counts[entry] for entry in entries),
                "membership": digest(
                    sorted(
                        (types[entry], tuple(paths))
                        for entry in entries
                        for paths in batch["selected_paths"][entry]
                    )
                ),
                "selection": digest(
                    sorted(
                        (
                            types[entry],
                            [str(path) for path in batch["selected_paths"][entry]],
                        )
                        for entry in entries
                    )
                ),
            }

    def end(self, phase, epoch):
        rows = dict(sorted(self.rows.items()))
        cohort = digest(list(rows))
        initial = self.first.setdefault(phase, rows)
        if set(rows) != set(initial):
            raise ValueError("Study cohort changed between visits")
        if phase == "train":
            self.changed_training_studies.update(
                uid
                for uid in rows
                if rows[uid]["membership"] != initial[uid]["membership"]
            )
        elif self.limits is not None and rows != initial:
            raise ValueError("Validation selection is not deterministic")
        self.history[phase].append(
            {
                "epoch": epoch,
                "visits": len(rows),
                "triplets": sum(row["triplets"] for row in rows.values()),
                "cohort_sha256": cohort,
                "selection_sha256": digest(rows),
            }
        )
        self.save(False)
        if self.limits is not None:
            total = sum(row["triplets"] for row in self.history["train"])
            if total > self.limits["training_triplets"]:
                raise ValueError("Original campaign training triplet budget exceeded")

    def save(self, complete):
        write_json(
            self.destination,
            {
                "complete": complete,
                "unit": "valid three-slice inputs before model series dropout; padding excluded",
                "train": self.history["train"],
                "validation": self.history["validation"],
                "changed_training_studies": len(self.changed_training_studies),
                "training_triplets": sum(
                    row["triplets"] for row in self.history["train"]
                ),
                "training_visits": sum(row["visits"] for row in self.history["train"]),
            },
        )

    def finish(self):
        if self.limits is not None:
            expected = self.limits
            actual = sum(row["triplets"] for row in self.history["train"])
            visits = sum(row["visits"] for row in self.history["train"])
            if (
                actual != expected["training_triplets"]
                or visits != expected["training_visits"]
            ):
                raise ValueError(
                    "Average realized training triplets changed from the sealed baseline"
                )
            if len(self.history["train"]) != expected["epochs"]:
                raise ValueError("Training did not complete the fixed epoch schedule")
            if not self.changed_training_studies:
                raise ValueError("No training selection changed across epochs")
            for row in self.history["train"]:
                if row["cohort_sha256"] != expected["training_cohort_sha256"]:
                    raise ValueError("Training cohort changed")
            for row in self.history["validation"]:
                if row["cohort_sha256"] != expected["validation_cohort_sha256"]:
                    raise ValueError("Validation cohort changed")
                if row["triplets"] != expected["validation_triplets"]:
                    raise ValueError("Validation x1 triplet count changed")
        self.save(True)


class TripletAccounting(L.Callback):
    def __init__(self, destination: Path):
        path = os.environ.get("KNEE_AUTORESEARCH_LIMITS")
        self.ledger = TripletLedger(
            destination, read_json(Path(path)) if path else None
        )

    def on_fit_start(self, trainer, pl_module):
        data = trainer.datamodule
        if (
            data.validation_dataset.training_seed is not None
            or data.validation_dataset.image_limit_multiplier != 1
        ):
            raise ValueError("Validation must be x1 with no training seed")
        if data.train_dataset.training_seed != pl_module.config.seed:
            raise ValueError("Training dataset must use the recorded run seed")

    def on_train_epoch_start(self, trainer, pl_module):
        self.ledger.begin()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.ledger.observe(batch)

    def on_train_epoch_end(self, trainer, pl_module):
        self.ledger.end("train", trainer.current_epoch)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.training_rows = self.ledger.rows
        self.ledger.begin()

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self.ledger.observe(batch)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.ledger.end("validation", trainer.current_epoch)
        self.ledger.rows = self.training_rows

    def on_fit_end(self, trainer, pl_module):
        self.ledger.finish()


def calibrate(config_path: Path, destination: Path):
    from knee_abnormality.config import load_config
    from knee_abnormality.data.dataset import build_data_loader
    from knee_abnormality.training.checkpoints import training_input_identity
    from knee_abnormality.training.lightning import KneeDataModule

    started = time.monotonic()
    config = load_config(config_path, folds=[0])
    data = KneeDataModule(config, 0)
    data.setup("fit")
    ledger = TripletLedger(destination / "triplet_accounting.json")
    train_loader = build_data_loader(
        data.train_dataset, config.trainer, config.trainer.batch_size
    )
    for epoch in range(config.trainer.max_epochs):
        data.train_dataset.set_epoch(epoch)
        ledger.begin()
        for batch in train_loader:
            ledger.observe(batch)
        ledger.end("train", epoch)
    validation_loader = data.val_dataloader()
    for epoch in range(2):
        ledger.begin()
        for batch in validation_loader:
            ledger.observe(batch)
        ledger.end("validation", epoch)
    if (
        ledger.history["validation"][0]["selection_sha256"]
        != ledger.history["validation"][1]["selection_sha256"]
    ):
        raise ValueError("Baseline validation is not deterministic")
    ledger.finish()
    write_json(
        destination / "input_identity.json",
        training_input_identity(
            config,
            data.labels,
            data.labels.training,
            data.labels.validation,
        ),
    )
    write_json(destination / "timing.json", {"seconds": time.monotonic() - started})
