"""Tangent-point tables for S-shaped activations, built like auto_LiRPA's BoundSShaped.precompute_relaxation."""
import torch

STEP = 0.01
POINTS = int(500 / STEP) + 5


def _tangent_table(func, dfunc):
    """For an upper end u >= 0: the point d <= 0 closest to 0 whose tangent stays below func on [d, u]."""
    upper = STEP * torch.arange(0, POINTS, dtype=torch.float32)
    below = lambda d: dfunc(d) * (upper - d) + func(d) <= func(upper)
    left, right = -torch.ones_like(upper), torch.zeros_like(upper)
    while True:
        checked = below(left)
        left = torch.where(checked, left, left * 2)
        if checked.all():
            break
    for _ in range(100):
        middle = (left + right) / 2
        checked = below(middle)
        left, right = torch.where(checked, middle, left), torch.where(checked, right, middle)
    return left


_SIGMOID = _tangent_table(torch.sigmoid, lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x)))
_TANH = _tangent_table(torch.tanh, lambda x: 1 - torch.tanh(x).pow(2))


def _lookup(table, u):
    index = (u / STEP).long().clamp(min=0, max=POINTS - 2) + 1
    return table.to(u.device, u.dtype)[index]


def sigma_tangent(u):
    return _lookup(_SIGMOID, u)


def tanh_tangent(u):
    return _lookup(_TANH, u)


FUNCTIONS = {'sigma_tangent': sigma_tangent, 'tanh_tangent': tanh_tangent}
