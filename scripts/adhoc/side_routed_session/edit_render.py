from pathlib import Path

p = Path("scripts/adhoc/render_routing_views.py")
s = p.read_text()


def sub(old, new):
    global s
    assert s.count(old) == 1, old[:50]
    s = s.replace(old, new)


start = s.index("    tokens = model.study_tokens(encoded).unflatten(0, valid.shape)\n")
end = s.index("def side_state(")
s = (
    s[:start]
    + '''    tokens = model.mil_norm(model._base_tokens(encoded))
    lateral, front, _ = model.centre_offsets().unbind()
    anterior = encoded["anterior_axis"]
    eligible = (
        valid & (encoded["routable"] & (encoded["image_plane"] != 1))[:, None]
    )
    values = model.landmark_scorers(tokens.float())[..., 0] - (
        anterior - front
    ).square() / (2 * LANDMARK_PRIOR_MM**2)
    values = values.masked_fill(~eligible, -torch.inf)
    left = encoded["left_probability"]
    axis = encoded["study_axis"][0]
    return {
        "knee_x": encoded["knee_x"].cpu(),
        "anterior": anterior.cpu(),
        "medial": routing[0].cpu(),
        "front": routing[2].cpu(),
        "landmark_front": values.flatten().softmax(0).view_as(values).cpu(),
        "shift": torch.stack(
            ((2 * left.mean() - 1) * lateral, encoded["study_centre_shift"][0, 0])
        ).cpu(),
        "rotation": float(torch.rad2deg(torch.atan2(axis[1], axis[0]))),
        "left": left.cpu(),
        "type": encoded["image_type"].cpu(),
        "valid": valid.cpu(),
    }


'''
    + s[end:]
)
sub(
    '''    scores = model.landmark_scorers(tokens.float())
    fitted = valid & encoded["routable"][:, None]
    landmarks = []
    for index, (coordinate, excluded) in enumerate(
        ((encoded["knee_x"], 2), (encoded["anterior_axis"], 1))
    ):
        eligible = fitted & (encoded["image_plane"] != excluded)[:, None]
        values = scores[..., index] - coordinate.square() / (2 * LANDMARK_PRIOR_MM**2)
        values = values.masked_fill(~eligible, -torch.inf)
        landmarks.append(values.flatten().softmax(0).view_as(values))
''',
    "",
) if "landmarks = []" in s else None
start = s.index('                peak = float(maps["landmark_midline"].max())\n')
end = s.index('                peak = float(maps["landmark_front"].max())\n')
s = s[:start] + s[end:]
sub("len(present), 6, figsize=(13.2,", "len(present), 5, figsize=(11.0,")
sub("axes[line][4],\n                    image,", "axes[line][3],\n                    image,")
s = s.replace("axes[line][5]", "axes[line][4]")
sub(
    '''                    "shift_midline_mm": float(maps["shift"][0]),
''',
    '''                    "shift_midline_mm": float(maps["shift"][0]),
                    "rotation_deg": maps["rotation"],
''',
)
sub(
    '''                f"midline shift {record['shift_midline_mm']:+.1f} mm (yellow; fitted "
                f"white dashed)  front''',
    '''                f"rotation {record['rotation_deg']:+.1f} deg  "
                f"midline shift {record['shift_midline_mm']:+.1f} mm (yellow; fitted "
                f"white dashed)  front''',
)
sub(
    '"middle slice as the model sees it with the fitted and learned midline, the "\n        "front boundary, the routing weights, the landmark weights and the crop\'s "',
    '"middle slice as the model sees it with the fitted and learned midline in the "\n        "knee\'s frame, the front boundary, the routing weights, the front landmark "\n        "weights and the crop\'s "',
)
p.write_text(s)
