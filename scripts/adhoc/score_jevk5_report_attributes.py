import copy
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import transformers
from huggingface_hub import snapshot_download
from pydantic import Field
from tqdm import tqdm

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.data.splits import STUDY
from knee_abnormality.provenance import copy_sources, sha256_file

MODEL = "alibiserikbay/JevK5"
MODEL_REVISION = "05b7ad9408a979c0e56af8c5592d216547e14c53"
LETTERS = "ABCDEFGHIJKLMNOP"
MAX_TOKENS = 16384
SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
ORDERS = ("forward", "reverse")
REPLACEMENTS = {
    "\u2206": "\u0394",
    "\u00b5": "\u03bc",
    "VKB": "VKB (ACL)",
    "HKB": "HKB (PCL)",
    "FCL": "FCL (lateral collateral ligament)",
}


class ReportAttributeCLI(CLISettings):
    model_config = describe(
        "Ask JevK5 a fixed multiple-choice question set about each training report and "
        "store every option's probability in both option orders; resumable by shard"
    )

    questions: Path = Field(
        default=Path("configs/labels/jevk5-report-attributes-v3.json"),
        description="Question set",
    )
    reports: Path = Field(
        default=Path("data/raw/train.csv"), description="Reports to score"
    )
    output_dir: Path = Field(
        description="Run directory; an existing one with the same identity resumes"
    )
    sample: int | None = Field(
        default=None, gt=0, description="Score only this many sampled reports"
    )
    seed: int = Field(default=20260924, description="Sampling seed")
    batch_size: int = Field(default=16, gt=0, description="Prompts per forward pass")
    shard_size: int = Field(default=200, gt=0, description="Reports per saved shard")


class Reader:
    def __init__(self, device: str):
        path = snapshot_download(MODEL, revision=MODEL_REVISION)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(path)
        config = transformers.AutoConfig.from_pretrained(path)
        if config.model_type not in {"qwen3_5", "qwen3_5_text"}:
            raise ValueError(f"Unexpected JevK5 model type {config.model_type}")
        self.model = (
            transformers.Qwen3_5ForCausalLM.from_pretrained(
                path,
                config=config.get_text_config(),
                dtype=torch.bfloat16,
            )
            .to(device)
            .eval()
        )
        slots = [
            self.tokenizer.encode(letter, add_special_tokens=False)
            for letter in LETTERS
        ]
        if any(len(ids) != 1 for ids in slots):
            raise ValueError("Every answer letter must be one token")
        self.slot_weight = self.model.lm_head.weight[[ids[0] for ids in slots]].detach()
        self.temperature = float(
            json.loads((Path(path) / "jevk5_config.json").read_text())["temperature"]
        )
        self.pad = self.tokenizer.pad_token_id or 0
        self.device = device

    def encode(self, report: str, criterion: str, texts: list[str]) -> list[int]:
        payload = {
            "evidence": report,
            "criterion": criterion,
            "options": [
                {"letter": LETTERS[i], "description": text}
                for i, text in enumerate(texts)
            ],
        }
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return self.tokenizer.encode(prompt, add_special_tokens=False)

    @torch.inference_mode()
    def study_probabilities(
        self, sequences: list[list[int]], counts: list[int], batch_size: int
    ) -> list[np.ndarray]:
        shared = shared_prefix(sequences)
        prefix = self.model.model(
            input_ids=torch.tensor([sequences[0][:shared]], device=self.device),
            use_cache=True,
        ).past_key_values
        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        out = [None] * len(sequences)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            tails = [sequences[i][shared:] for i in batch]
            ids = torch.full(
                (len(batch), max(map(len, tails))), self.pad, dtype=torch.long
            )
            for row, tail in enumerate(tails):
                ids[row, : len(tail)] = torch.tensor(tail)
            last = torch.tensor([len(t) - 1 for t in tails], device=self.device)
            hidden = self.model.model(
                input_ids=ids.to(self.device),
                past_key_values=expand_cache(prefix, len(batch)),
                use_cache=True,
            ).last_hidden_state
            logits = (
                hidden[torch.arange(len(batch), device=self.device), last]
                @ self.slot_weight.T
            )
            logits = logits.float().cpu().numpy() / self.temperature
            for index, row in zip(batch, logits, strict=True):
                values = np.exp(row[: counts[index]] - row[: counts[index]].max())
                out[index] = values / values.sum()
        return out


def shared_prefix(sequences: list[list[int]]) -> int:
    length = min(map(len, sequences)) - 1
    first = sequences[0]
    for sequence in sequences[1:]:
        length = next((i for i in range(length) if sequence[i] != first[i]), length)
    return length


def expand_cache(cache: transformers.Cache, rows: int) -> transformers.Cache:
    expanded = copy.deepcopy(cache)
    for layer in expanded.layers:
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if isinstance(tensor, torch.Tensor):
                setattr(
                    layer, name, tensor.expand(rows, *tensor.shape[1:]).contiguous()
                )
        for states in (
            getattr(layer, "conv_states", {}),
            getattr(layer, "recurrent_states", {}),
        ):
            for index, tensor in states.items():
                if tensor is not None:
                    states[index] = tensor.expand(rows, *tensor.shape[1:]).contiguous()
    return expanded


def load_questions(path: Path) -> tuple[str, list[dict]]:
    document = json.loads(path.read_text())
    questions = document["questions"]
    ids = [q["id"] for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("Question ids must be unique")
    if any(not 2 <= len(q["options"]) <= len(LETTERS) for q in questions):
        raise ValueError(f"Every question needs 2 to {len(LETTERS)} options")
    return document["preamble"], questions


def load_reports(settings: ReportAttributeCLI) -> pl.DataFrame:
    reports = (
        pl.read_csv(settings.reports, infer_schema_length=0)
        .select(STUDY, pl.col("Report").fill_null("").str.replace_many(REPLACEMENTS))
        .sort(STUDY)
    )
    if settings.sample is not None:
        reports = reports.sample(settings.sample, seed=settings.seed).sort(STUDY)
    return reports


def score_shard(
    reader: Reader,
    preamble: str,
    questions: list[dict],
    shard: pl.DataFrame,
    batch_size: int,
    label: str,
) -> tuple[pl.DataFrame, int]:
    items = []
    per_study = len(questions) * len(ORDERS)
    for study, report in shard.iter_rows():
        for question in questions:
            keys = list(question["options"])
            criterion = f"{preamble} {question['criterion']}"
            for order in ORDERS:
                ordered = keys if order == "forward" else keys[::-1]
                texts = [question["options"][key] for key in ordered]
                ids = reader.encode(report, criterion, texts)
                if len(ids) > MAX_TOKENS:
                    raise ValueError(f"{study} {question['id']}: {len(ids)} tokens")
                items.append((study, question["id"], order, ordered, ids))
    probabilities = []
    for start in tqdm(range(0, len(items), per_study), desc=label, unit="study"):
        study_items = items[start : start + per_study]
        probabilities.extend(
            reader.study_probabilities(
                [item[4] for item in study_items],
                [len(item[3]) for item in study_items],
                batch_size,
            )
        )
    rows = {}
    for (study, question, order, ordered, _), result in zip(
        items, probabilities, strict=True
    ):
        for key, value in zip(ordered, result, strict=True):
            rows.setdefault((study, question, key), {})[order] = float(value)
    frame = pl.DataFrame(
        [
            {STUDY: study, "question": question, "option": key, **values}
            for (study, question, key), values in rows.items()
        ]
    ).with_columns(probability=(pl.col("forward") + pl.col("reverse")) / 2)
    return frame, max(len(item[4]) for item in items)


def run(settings: ReportAttributeCLI) -> None:
    if not torch.cuda.is_available():
        raise ValueError("A CUDA device is required")
    preamble, questions = load_questions(settings.questions)
    reports = load_reports(settings)
    identity = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "questions": str(settings.questions),
        "questions_sha256": sha256_file(settings.questions),
        "reports": str(settings.reports),
        "reports_sha256": sha256_file(settings.reports),
        "sample": settings.sample,
        "seed": settings.seed,
        "shard_size": settings.shard_size,
        "studies": reports.height,
        "question_count": len(questions),
        "orders": list(ORDERS),
    }
    shards_dir = settings.output_dir / "shards"
    run_path = settings.output_dir / "run.json"
    if run_path.exists():
        if json.loads(run_path.read_text())["identity"] != identity:
            raise ValueError(
                f"{settings.output_dir} holds a run with a different identity"
            )
    else:
        settings.output_dir.mkdir(parents=True, exist_ok=False)
        shards_dir.mkdir()
        copy_sources([Path(__file__)], settings.output_dir / "source")
        run_path.write_text(json.dumps({"identity": identity}, indent=2) + "\n")
    reader = Reader("cuda")
    started = time.perf_counter()
    scored = 0
    longest = 0
    for number, offset in enumerate(range(0, reports.height, settings.shard_size)):
        path = shards_dir / f"shard-{number:05d}.parquet"
        if path.exists():
            continue
        shard = reports.slice(offset, settings.shard_size)
        label = f"shard {number + 1}/{-(-reports.height // settings.shard_size)}"
        frame, length = score_shard(
            reader, preamble, questions, shard, settings.batch_size, label
        )
        frame.write_parquet(path)
        scored += shard.height
        longest = max(longest, length)
        elapsed = time.perf_counter() - started
        print(
            f"shard {number}: {offset + shard.height}/{reports.height} studies, "
            f"{elapsed / scored:.2f} s per study, longest prompt {length} tokens",
            flush=True,
        )
    attributes = pl.concat(
        [pl.read_parquet(path) for path in sorted(shards_dir.glob("shard-*.parquet"))]
    )
    attributes.write_parquet(settings.output_dir / "attributes.parquet")
    summary = {
        "identity": identity,
        "temperature": reader.temperature,
        "studies_scored_this_run": scored,
        "seconds_this_run": round(time.perf_counter() - started, 1),
        "longest_prompt_tokens_this_run": longest,
        "rows": attributes.height,
        "gpu": torch.cuda.get_device_name(0),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    (settings.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    run(ReportAttributeCLI())


if __name__ == "__main__":
    main()
