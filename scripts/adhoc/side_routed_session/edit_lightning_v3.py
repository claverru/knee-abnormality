from pathlib import Path

p = Path("src/knee_abnormality/training/lightning.py")
s = p.read_text()


def sub(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)


sub(
    "SIDE_LOSS_WEIGHT = 0.1\n",
    'SIDE_LOSS_WEIGHT = 0.1\nROUTED_HALF_LOSS_WEIGHT = 0.075\nROUTED_HALF_TARGETS = ("Medial Meniscus", "Lateral Meniscus")\n',
)
sub(
    '''class KneeDataModule(L.LightningDataModule):
''',
    '''def routed_half_loss(
    model: KneeAbnormalityModel,
    encoded: dict,
    batch: dict,
    indices: list[int],
    class_balance_weights: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    valid = encoded["visual_valid"] & encoded["routable"][:, None]
    routing = model.routing_weights(
        encoded["knee_x"],
        encoded["anterior_axis"],
        encoded["left_probability"][:, None],
        encoded["routable"][:, None],
        encoded["centre_shift"][:, None],
    )[:2]
    tokens = model.study_tokens(encoded)
    with torch.autocast(device_type=tokens.device.type, enabled=False):
        values = model.half_detector(tokens.float()).squeeze(-1)
        membership = (
            torch.arange(encoded["batch_size"], device=tokens.device)[:, None]
            == encoded["study"][:, None].expand_as(valid).flatten()[None]
        )
        mass = (routing.float() * valid).flatten(1)[:, None] * membership[None]
        total = mass.sum(-1)
        scores = (
            torch.logsumexp(values[None, None] + mass.clamp_min(1e-30).log(), -1)
            - total.clamp_min(1e-6).log()
        ).transpose(0, 1)
        used = (total >= 1).all(0) & batch["target_observed"][:, indices].all(-1)
        if not used.any():
            return values.sum() * 0, 0
        targets = batch["target"][:, indices][used].float()
        losses = F.binary_cross_entropy_with_logits(
            scores[used], targets, reduction="none"
        )
        weights = class_balance_weights[:, indices]
        losses = losses * torch.where(targets >= 0.5, weights[0], weights[1])
        return losses.mean(), int(used.sum())


class KneeDataModule(L.LightningDataModule):
''',
)
sub(
    '''        loss = loss + SIDE_LOSS_WEIGHT * self._side_step(self.model, batch, encoded, "")
''',
    '''        loss = loss + SIDE_LOSS_WEIGHT * self._side_step(self.model, batch, encoded, "")
        auxiliary, used_studies = routed_half_loss(
            self.model,
            encoded,
            batch,
            [self.targets.index(name) for name in ROUTED_HALF_TARGETS],
            self.class_balance_weights,
        )
        loss = loss + ROUTED_HALF_LOSS_WEIGHT * auxiliary
        if used_studies:
            self._log_epoch_metric("routed_half_loss", auxiliary, used_studies)
''',
)
p.write_text(s)

p = Path("src/knee_abnormality/inference/checkpoint.py")
s = p.read_text()
sub(
    '''        if model.view_label_bias is not None:
            model.view_label_bias.div_(temperature)
''',
    '''        if model.view_label_bias is not None:
            model.view_label_bias.div_(temperature)
        model.position_prior.weight.div_(temperature)
        model.position_prior.bias.div_(temperature)
        model.routing_temperature.mul_(temperature)
''',
)
p.write_text(s)
