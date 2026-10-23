import math
import sys
import types

import torch

ROOT2 = 1 / math.sqrt(2)
stub = types.ModuleType("pywt")


class Wavelet:
    def __init__(self, name):
        assert name == "db1"
        self.dec_lo = [ROOT2, ROOT2]
        self.dec_hi = [-ROOT2, ROOT2]
        self.rec_lo = [ROOT2, ROOT2]
        self.rec_hi = [ROOT2, -ROOT2]


stub.Wavelet = Wavelet
sys.modules["pywt"] = stub
sys.path.insert(0, "/home/claudio/.cache/wtconv-src/WTConv")
sys.path.insert(
    0,
    "/home/claudio/src/knee-abnormality/outputs/autoresearch/backbone-families/experiments/wtconvnext-tiny-20261016T005805Z-b03b5f00/workspace/src",
)

from wtconv.wtconv2d import WTConv2d as Original

from knee_abnormality.modeling.wtconv import WTConv2d as Vendored

torch.manual_seed(0)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
for channels, size, levels in ((96, 56, 5), (192, 28, 4), (384, 14, 3), (768, 7, 2)):
    original = Original(channels, channels, kernel_size=5, wt_levels=levels).cuda()
    with torch.no_grad():
        original.base_conv.weight.normal_(0, 0.1)
        original.base_conv.bias.normal_(0, 0.1)
        original.base_scale.weight.uniform_(0.5, 1.5)
        for conv, scale in zip(original.wavelet_convs, original.wavelet_scale):
            conv.weight.normal_(0, 0.1)
            scale.weight.uniform_(0.05, 0.2)
    vendored = Vendored(channels, levels).cuda()
    with torch.no_grad():
        vendored.base_weight.copy_(original.base_conv.weight)
        vendored.base_bias.copy_(original.base_conv.bias)
        vendored.base_scale.copy_(original.base_scale.weight)
        for index in range(levels):
            vendored.wt_weights[index].copy_(original.wavelet_convs[index].weight)
            vendored.wt_scales[index].copy_(original.wavelet_scale[index].weight)
    x = torch.randn(16, channels, size, size, device="cuda")
    grad = torch.randn(16, channels, size, size, device="cuda")
    results = []
    for layer, data in (
        (original, x.clone()),
        (vendored, x.clone().contiguous(memory_format=torch.channels_last)),
    ):
        data.requires_grad_(True)
        layer.zero_grad()
        out = layer(data)
        out.backward(grad)
        missing = [name for name, p in layer.named_parameters() if p.grad is None]
        if missing:
            print(
                type(layer).__module__, channels, size, levels, "no gradient:", missing
            )
        params = {
            name: (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
            for name, p in layer.named_parameters()
        }
        results.append((out.detach().float(), data.grad.float(), params))
    (o1, g1, p1), (o2, g2, p2) = results

    def rel(a, b):
        return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()

    base_pairs = [
        ("base_conv.weight", "base_weight"),
        ("base_conv.bias", "base_bias"),
        ("base_scale.weight", "base_scale"),
    ]
    level_pairs = [
        (f"wavelet_convs.{i}.weight", f"wt_weights.{i}") for i in range(levels)
    ] + [(f"wavelet_scale.{i}.weight", f"wt_scales.{i}") for i in range(levels)]
    param_errors = {b: rel(p2[b], p1[a]) for a, b in base_pairs + level_pairs}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        bf_original = original(x)
        bf_vendored = vendored(x.contiguous(memory_format=torch.channels_last))
    print(
        f"C{channels} {size}px L{levels}: out {rel(o2, o1):.2e} input-grad {rel(g2, g1):.2e} "
        f"max param-grad {max(param_errors.values()):.2e} ({max(param_errors, key=param_errors.get)}) "
        f"bf16 out {rel(bf_vendored.float(), bf_original.float()):.2e}"
    )
