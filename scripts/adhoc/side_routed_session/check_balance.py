import torch

from knee_abnormality.modeling.model import BalancedSides

weights = torch.randn(1000)
y = torch.randn(1000, requires_grad=True)
z = torch.randn(1000, requires_grad=True)
a, b = BalancedSides.apply(y, z)
(10.0 * (a * weights).sum() + 0.1 * (b * weights).sum()).backward()
print("natural", round(float(10.0 * weights.norm()), 3), round(float(0.1 * weights.norm()), 3), "balanced", round(float(y.grad.norm()), 3), round(float(z.grad.norm()), 3))
y = torch.randn(1000, requires_grad=True)
z = torch.randn(1000, requires_grad=True)
a, b = BalancedSides.apply(y, z)
(10.0 * (a * weights).sum() + 0.0 * b.sum()).backward()
print("one side silent", round(float(y.grad.norm()), 3), round(float(z.grad.norm()), 3))
