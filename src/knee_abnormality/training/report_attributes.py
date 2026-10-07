from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from knee_abnormality.data.report_attributes import (
    ReportAttributeTargets,
    value_column,
    weight_column,
)
from knee_abnormality.data.splits import STUDY
from knee_abnormality.modeling.model import (
    MIL_ATTENTION_WIDTH,
    PROJECTION_WIDTH,
    ROUTES,
    ROUTING_MINIMUM_WEIGHT,
)

QUESTION_ROUTES = {
    "medial_meniscus_status": ("meniscus_status", "meniscus_medial"),
    "lateral_meniscus_status": ("meniscus_status", "meniscus_lateral"),
    "medial_meniscus_tear_type": ("meniscus_tear_type", "meniscus_medial"),
    "lateral_meniscus_tear_type": ("meniscus_tear_type", "meniscus_lateral"),
    "medial_cartilage_depth": ("cartilage_depth", "oa_medial"),
    "lateral_cartilage_depth": ("cartilage_depth", "oa_lateral"),
    "pf_cartilage_depth": ("cartilage_depth", "kneecap"),
    "medial_cartilage_extent": ("cartilage_extent", "oa_medial"),
    "lateral_cartilage_extent": ("cartilage_extent", "oa_lateral"),
    "pf_cartilage_extent": ("cartilage_extent", "kneecap"),
    "acl_status": ("acl_status", "core"),
    "extra_pcl": ("extra_pcl", "core"),
}


class Region(NamedTuple):
    family: str
    route: str
    question: str
    located: tuple[str, ...]
    unplaced: tuple[str, ...]


REGIONS = (
    *(
        Region(
            "meniscus_tear_place",
            f"tear_{side}_{part}",
            f"{side}_meniscus_tear_location",
            (*options, "multiple"),
            ("not_stated",),
        )
        for side in ("medial", "lateral")
        for part, options in (
            ("anterior", ("anterior", "anterior_body")),
            ("body", ("body", "posterior_body", "anterior_body")),
            ("posterior", ("posterior", "posterior_body")),
        )
    ),
    Region(
        "marrow_edema_place",
        "oa_medial",
        "marrow_edema_location",
        ("medial", "medial_lateral"),
        ("not_stated", "patellofemoral_femorotibial"),
    ),
    Region(
        "marrow_edema_place",
        "oa_lateral",
        "marrow_edema_location",
        ("lateral", "medial_lateral"),
        ("not_stated", "patellofemoral_femorotibial"),
    ),
    Region(
        "marrow_edema_place",
        "kneecap",
        "marrow_edema_location",
        ("patellofemoral", "patellofemoral_femorotibial"),
        ("not_stated",),
    ),
    Region(
        "patellar_facet_place",
        "facet_medial",
        "patellar_facet",
        ("medial_facet", "both_or_ridge"),
        ("not_stated",),
    ),
    Region(
        "patellar_facet_place",
        "facet_lateral",
        "patellar_facet",
        ("lateral_facet", "both_or_ridge"),
        ("not_stated",),
    ),
)
REGION_FAMILIES = tuple(dict.fromkeys(region.family for region in REGIONS))
REGION_IDS = tuple(f"{region.family}_{region.route}" for region in REGIONS)


class ReportAttributeHeads(nn.Module):
    def __init__(self, attributes: ReportAttributeTargets) -> None:
        super().__init__()
        options = {question.id: question.options for question in attributes.questions}
        placed = {region.question for region in REGIONS}
        missing = sorted((placed | set(QUESTION_ROUTES)) - set(options))
        if missing:
            raise ValueError(
                f"Routed report attributes require the questions {missing}"
            )
        questions = [
            question for question in attributes.questions if question.id not in placed
        ]
        self.question_ids = tuple(question.id for question in questions)
        self.option_counts = tuple(len(question.options) for question in questions)
        question_heads = [
            QUESTION_ROUTES.get(question.id, (question.id,))[0]
            for question in questions
        ]
        head_ids = tuple(dict.fromkeys(question_heads))
        head_options = {}
        for question, head in zip(questions, question_heads, strict=True):
            if head_options.setdefault(head, question.options) != question.options:
                raise ValueError(f"Shared report-attribute options differ for {head}")
        head_offsets = [0]
        for head in head_ids:
            head_offsets.append(head_offsets[-1] + len(head_options[head]))
        question_head = [head_ids.index(head) for head in question_heads]
        routed = [
            index
            for index, question in enumerate(self.question_ids)
            if question in QUESTION_ROUTES
        ]
        unrouted = [
            index
            for index, question in enumerate(self.question_ids)
            if question not in QUESTION_ROUTES
        ]
        self.attention = nn.Sequential(
            nn.Linear(PROJECTION_WIDTH, MIL_ATTENTION_WIDTH),
            nn.Tanh(),
            nn.Linear(MIL_ATTENTION_WIDTH, len(head_ids) + len(REGION_FAMILIES)),
        )
        self.classifier = nn.Linear(
            PROJECTION_WIDTH, head_offsets[-1] + len(REGION_FAMILIES)
        )
        for name, values in (
            ("unrouted", unrouted),
            ("routed", routed),
            ("unrouted_channel", [question_head[index] for index in unrouted]),
            ("routed_channel", [question_head[index] for index in routed]),
            (
                "routed_route",
                [
                    ROUTES.index(QUESTION_ROUTES[self.question_ids[index]][1])
                    for index in routed
                ],
            ),
            (
                "option_question",
                [
                    index
                    for index, count in enumerate(self.option_counts)
                    for _ in range(count)
                ],
            ),
            (
                "option_row",
                [
                    head_offsets[head] + option
                    for head, count in zip(
                        question_head, self.option_counts, strict=True
                    )
                    for option in range(count)
                ],
            ),
            (
                "region_channel",
                [
                    len(head_ids) + REGION_FAMILIES.index(region.family)
                    for region in REGIONS
                ],
            ),
            ("region_route", [ROUTES.index(region.route) for region in REGIONS]),
            (
                "region_row",
                [
                    head_offsets[-1] + REGION_FAMILIES.index(region.family)
                    for region in REGIONS
                ],
            ),
        ):
            self.register_buffer(name, torch.tensor(values), persistent=False)
        frame = attributes.frame
        uids = frame[STUDY].to_list()
        self.study_rows = {uid: row for row, uid in enumerate(uids)}

        def columns(names: list[str]) -> torch.Tensor:
            return torch.from_numpy(frame.select(names).to_numpy()).float()

        def mass(question: str, names: tuple[str, ...]) -> torch.Tensor:
            return columns([value_column(question, option) for option in names]).sum(1)

        self.register_buffer(
            "table",
            columns(
                [
                    value_column(question.id, option)
                    for question in questions
                    for option in question.options
                ]
            ),
            persistent=False,
        )
        self.register_buffer(
            "weights",
            columns([weight_column(question.id) for question in questions]),
            persistent=False,
        )
        self.register_buffer(
            "region_table",
            torch.stack(
                [mass(region.question, region.located) for region in REGIONS], 1
            ),
            persistent=False,
        )
        self.register_buffer(
            "region_weights",
            torch.stack(
                [
                    columns([weight_column(region.question)])[:, 0]
                    * (1 - mass(region.question, region.unplaced))
                    for region in REGIONS
                ],
                1,
            ),
            persistent=False,
        )

    def targets(self, study_uids: list[str]) -> tuple[torch.Tensor, ...]:
        rows = torch.tensor(
            [self.study_rows.get(uid, -1) for uid in study_uids],
            device=self.table.device,
        )
        observed = rows >= 0
        safe = rows.clamp_min(0)
        return (
            self.table[safe],
            self.weights[safe],
            self.region_table[safe],
            self.region_weights[safe],
            observed,
        )

    def _pool(
        self,
        tokens: torch.Tensor,
        membership: torch.Tensor,
        channels: torch.Tensor,
        routing: torch.Tensor | None,
    ) -> torch.Tensor:
        scores = self.attention(tokens).transpose(0, 1)[channels][None]
        if routing is not None:
            routed = membership.float() @ routing.T >= 1
            scores = scores + torch.where(
                routed[:, :, None],
                routing.clamp_min(ROUTING_MINIMUM_WEIGHT).log()[None],
                0.0,
            )
        attention = scores.masked_fill(~membership[:, None], -torch.inf).softmax(-1)
        return torch.einsum("bqn,nf->bqf", attention, tokens)

    def forward(
        self,
        tokens: torch.Tensor,
        token_study: torch.Tensor,
        token_valid: torch.Tensor,
        shared_tokens: torch.Tensor,
        routing: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            tokens = tokens.float()
            shared_tokens = shared_tokens.float()
            routing = routing.float()
            studies = torch.arange(batch_size, device=tokens.device)[:, None]
            membership = (studies == token_study[None]) & token_valid[None]
            pooled = tokens.new_zeros(
                batch_size, len(self.question_ids), tokens.shape[-1]
            )
            pooled[:, self.unrouted] = self._pool(
                tokens, membership, self.unrouted_channel, None
            )
            pooled[:, self.routed] = self._pool(
                shared_tokens,
                membership,
                self.routed_channel,
                routing[self.routed_route],
            )
            regions = self._pool(
                shared_tokens,
                membership,
                self.region_channel,
                routing[self.region_route],
            )
            weight = self.classifier.weight
            bias = self.classifier.bias
            return (
                (pooled[:, self.option_question] * weight[self.option_row][None]).sum(
                    -1
                )
                + bias[self.option_row],
                (regions * weight[self.region_row][None]).sum(-1)
                + bias[self.region_row],
            )


def report_attribute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    region_logits: torch.Tensor,
    region_targets: torch.Tensor,
    region_weights: torch.Tensor,
    observed: torch.Tensor,
    option_counts: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        zero = (logits.sum() + region_logits.sum()) * 0
        return (
            zero,
            torch.zeros(len(option_counts), device=logits.device),
            torch.zeros(region_logits.shape[-1], device=logits.device),
        )
    cross_entropy = torch.stack(cross_entropy, 1)[observed]
    divergence = torch.stack(divergence, 1)[observed].detach()
    weights = weights[observed]
    region_weights = region_weights[observed]
    region_loss = F.binary_cross_entropy_with_logits(
        region_logits.float()[observed], region_targets[observed], reduction="none"
    )
    loss = ((cross_entropy * weights).sum() + (region_loss * region_weights).sum()) / (
        weights.sum() + region_weights.sum()
    ).clamp_min(1e-6)
    return (
        loss,
        divergence.mean(0),
        (region_loss.detach() * region_weights).sum(0)
        / region_weights.sum(0).clamp_min(1e-6),
    )
