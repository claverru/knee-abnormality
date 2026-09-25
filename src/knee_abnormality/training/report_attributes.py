import torch
from torch import nn
from torch.nn import functional as F

from knee_abnormality.data.labels import STUDY
from knee_abnormality.data.report_attributes import ReportAttributeTargets
from knee_abnormality.modeling.model import MIL_ATTENTION_WIDTH, PROJECTION_WIDTH


class ReportAttributeHeads(nn.Module):
    def __init__(self, attributes: ReportAttributeTargets) -> None:
        super().__init__()
        self.question_ids = tuple(question.id for question in attributes.questions)
        self.option_counts = tuple(
            len(question.options) for question in attributes.questions
        )
        self.attention = nn.Sequential(
            nn.Linear(PROJECTION_WIDTH, MIL_ATTENTION_WIDTH),
            nn.Tanh(),
            nn.Linear(MIL_ATTENTION_WIDTH, len(self.option_counts)),
        )
        self.classifier = nn.Linear(PROJECTION_WIDTH, sum(self.option_counts))
        self.register_buffer(
            "option_question",
            torch.repeat_interleave(
                torch.arange(len(self.option_counts)),
                torch.tensor(self.option_counts),
            ),
            persistent=False,
        )
        uids = attributes.frame[STUDY].to_list()
        self.study_rows = {uid: row for row, uid in enumerate(uids)}
        self.register_buffer(
            "table",
            torch.from_numpy(
                attributes.frame.select(attributes.value_columns).to_numpy()
            ).float(),
            persistent=False,
        )
        self.register_buffer(
            "weights",
            torch.from_numpy(
                attributes.frame.select(attributes.weight_columns).to_numpy()
            ).float(),
            persistent=False,
        )

    def targets(
        self, study_uids: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = torch.tensor(
            [self.study_rows.get(uid, -1) for uid in study_uids],
            device=self.table.device,
        )
        observed = rows >= 0
        safe = rows.clamp_min(0)
        return self.table[safe], self.weights[safe], observed

    def forward(
        self,
        tokens: torch.Tensor,
        token_study: torch.Tensor,
        token_valid: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            tokens = tokens.float()
            scores = self.attention(tokens).transpose(0, 1)
            membership = (
                torch.arange(batch_size, device=tokens.device)[:, None]
                == token_study[None]
            ) & token_valid[None]
            attention = (
                scores.unsqueeze(0)
                .expand(batch_size, -1, -1)
                .masked_fill(~membership[:, None], -torch.inf)
                .softmax(-1)
            )
            pooled = torch.einsum("bqn,nf->bqf", attention, tokens)
            return (pooled[:, self.option_question] * self.classifier.weight[None]).sum(
                -1
            ) + self.classifier.bias


def report_attribute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    observed: torch.Tensor,
    option_counts: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    cross_entropy = []
    divergence = []
    for question_logits, question_targets in zip(
        logits.float().split(option_counts, -1),
        targets.split(option_counts, -1),
        strict=True,
    ):
        loss = -(question_targets * F.log_softmax(question_logits, -1)).sum(-1)
        entropy = -(question_targets * question_targets.clamp_min(1e-12).log()).sum(-1)
        cross_entropy.append(loss)
        divergence.append(loss - entropy)
    if not observed.any():
        zero = logits.sum() * 0
        return zero, torch.zeros(len(option_counts), device=logits.device)
    cross_entropy = torch.stack(cross_entropy, 1)[observed]
    divergence = torch.stack(divergence, 1)[observed].detach()
    weights = weights[observed]
    loss = (cross_entropy * weights).sum() / weights.sum().clamp_min(1e-6)
    return loss, divergence.mean(0)
