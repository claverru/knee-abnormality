from pathlib import Path

p = Path("scripts/adhoc/render_routing_views.py")
s = p.read_text()


def sub(old, new):
    global s
    assert s.count(old) == 1, old[:50]
    s = s.replace(old, new)


def between(start, end, new):
    global s
    a = s.index(start)
    b = s.index(end, a)
    s = s[:a] + new + s[b:]


sub("from knee_abnormality.modeling.model import LANDMARK_PRIOR_MM\n", "")
sub('        encoded["centre_shift"][:, None],\n', "")
between(
    "    tokens = model.mil_norm(model._base_tokens(encoded))\n",
    "    left = encoded[\"left_probability\"]\n",
    "    lateral, front = model.centre_offsets().unbind()\n"
    "    anterior = encoded[\"anterior_axis\"]\n"
    "    prior = model.position_priors(encoded[\"position\"], encoded[\"routable\"])\n",
)
sub(
    '        "landmark_front": values.flatten().softmax(0).view_as(values).cpu(),\n',
    '        "prior": prior.cpu(),\n',
)
sub(
    '((2 * left.mean() - 1) * lateral, encoded["study_centre_shift"][0, 0])',
    "((2 * left.mean() - 1) * lateral, front)",
)
between(
    '                peak = float(maps["landmark_front"].max())\n',
    "                other = wide_index.get(paths[item])\n",
    '''                values = maps["prior"][prior_target, item]
                axes[line][3].imshow(image, cmap="gray", vmin=0, vmax=255)
                axes[line][3].imshow(
                    values.reshape(grid, grid).numpy(),
                    cmap="coolwarm",
                    vmin=-1,
                    vmax=1,
                    alpha=0.45,
                    extent=extent,
                    interpolation="nearest",
                )
                axes[line][3].set_title(f"position prior, {settings.prior_target}", fontsize=7)
                axes[line][3].set_xticks([])
                axes[line][3].set_yticks([])
''',
)
sub(
    '    studies: int = Field(default=8, gt=0, description="Studies to render")\n',
    '    studies: int = Field(default=8, gt=0, description="Studies to render")\n'
    '    prior_target: str = Field(\n'
    '        default="ACL", description="Target whose position prior is drawn"\n'
    "    )\n",
)
sub(
    "    origin_y, origin_x, stride_y, stride_x = model.feature_mapping\n",
    "    origin_y, origin_x, stride_y, stride_x = model.feature_mapping\n"
    "    prior_target = targets.index(settings.prior_target)\n",
)
sub(
    '"knee\'s frame, the front boundary, the routing weights, the front landmark "\n        "weights and the crop\'s "',
    '"knee\'s frame, the front boundary, the routing weights, one target\'s position "\n        "prior and the crop\'s "',
)
p.write_text(s)
