import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys

TARGET = "knee_abnormality.data.dataset"
LOG_DIR = os.environ.get("KNEE_BUDGET_LOG")
SLOT = os.environ.get("KNEE_BUDGET_SLOT")
SCALE = os.environ.get("KNEE_BUDGET_SCALE")
HERE = os.path.dirname(os.path.abspath(__file__))


def scaled(budget):
    if SCALE == "half":
        return max(1, budget // 2)
    if SCALE == "double":
        return 2 * budget
    raise ValueError(f"Unsupported KNEE_BUDGET_SCALE: {SCALE}")


def patch(module):
    dataset = module.KneeStudyDataset
    original_budgets = dataset._study_budgets
    original_item = dataset.__getitem__

    def study_budgets(self, slot_sources):
        base = tuple(original_budgets(self, slot_sources))
        budgets = list(base)
        if SLOT is not None:
            budgets[int(SLOT)] = scaled(budgets[int(SLOT)])
        self._budget_trace = {
            "sources": [len(sources) for _, sources in slot_sources],
            "base": list(base),
            "budget": budgets,
        }
        return tuple(budgets)

    def item(self, index):
        value = original_item(self, index)
        record = {
            "study": value["study_uid"],
            **self._budget_trace,
            "triplets": [
                sum(series.triplet_count for series in acquisitions)
                for acquisitions in value["series"]
            ],
            "capacity": [metadata["capacity"] for metadata in value["sampling"]],
            "requested": [
                metadata["requested_triplets"] for metadata in value["sampling"]
            ],
        }
        path = os.path.join(LOG_DIR, f"trace-{os.getpid()}.jsonl")
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        return value

    dataset._study_budgets = study_budgets
    dataset.__getitem__ = item


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None:
            return None
        execute = spec.loader.exec_module

        def exec_module(module):
            execute(module)
            patch(module)

        spec.loader.exec_module = exec_module
        return spec


if LOG_DIR:
    if SLOT is not None:
        scaled(2)
    sys.meta_path.insert(0, Finder())

shadowed = importlib.machinery.PathFinder.find_spec(
    "sitecustomize",
    [entry for entry in sys.path if os.path.abspath(entry or ".") != HERE],
)
if shadowed is not None and shadowed.origin is not None:
    shadowed.loader.exec_module(importlib.util.module_from_spec(shadowed))
