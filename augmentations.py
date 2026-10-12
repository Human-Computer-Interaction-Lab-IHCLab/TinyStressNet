"""Augmentation conditions, framed as approximate-invariance operators.

  - "none"      : standardized tensor, unchanged.
  - "gaussian"  : Gaussian jitter, x' = x + e,  e ~ N(0, 0.1^2 I).
  - "radial"    : radial-rescaling family (magnitude warp + directional radial
                  noise + symmetric scaling), each applied with probability 0.7.
                  This is exactly `apply_symmetry_augmentations` from the v2 code.
  - "affine"    : bounded affine family (global scale + channel offsets + a
                  bounded shift along an arousal-oriented axis).

Each function maps a (3, 1) float tensor to a (3, 1) float tensor.

NOTE ON "affine": this family is described in the manuscript but was NOT present
in the released v2 scripts. It is implemented here to make code and manuscript
consistent; its reported numbers should be re-confirmed by running this repo.
The "none", "gaussian" and "radial" conditions match the v2 code exactly.
"""

import torch

CONDITIONS = ("none", "gaussian", "radial", "affine")


def gaussian_jitter(x, sigma: float = 0.1):
    return x + torch.randn_like(x) * sigma


# ---- radial-rescaling family (verbatim behaviour of v2 symmetry_augmentations) ----
def magnitude_warping(x, alpha: float = 0.08):
    a = torch.rand(1, device=x.device) * 2 * alpha - alpha
    mu = x.mean(dim=0, keepdim=True)
    return x + a * (x - mu)


def radial_noise(x, sigma: float = 0.02):
    mu = x.mean(dim=0, keepdim=True)
    d = x - mu
    unit = d / (d.norm(dim=0, keepdim=True) + 1e-8)
    eps = torch.normal(0.0, sigma, size=(1,), device=x.device)
    return x + eps * unit


def scaling(x, strength: float = 0.04):
    beta = 1 + strength * (2 * torch.rand(1, device=x.device) - 1)  # beta ~ U(0.96, 1.04)
    mu = x.mean(dim=0, keepdim=True)
    return mu + beta * (x - mu)


def radial_family(x):
    for t in (magnitude_warping, radial_noise, scaling):
        if torch.rand(1) > 0.3:  # applied with prob 0.7
            x = t(x)
    return x


# ---- bounded affine family (manuscript spec; see NOTE above) ----
_AXIS = torch.tensor([[1.0], [1.0], [-0.5]])


def affine_family(x):
    beta = torch.empty(1, device=x.device).uniform_(0.94, 1.06)
    delta = torch.empty_like(x)
    delta[0] = torch.empty(1, device=x.device).uniform_(-0.03, 0.03)
    delta[1] = torch.empty(1, device=x.device).uniform_(-0.03, 0.03)
    delta[2] = torch.empty(1, device=x.device).uniform_(-0.02, 0.02)
    eta = torch.empty(1, device=x.device).uniform_(-0.03, 0.03)
    return beta * x + delta + eta * _AXIS.to(x.device)


def augment(x, condition: str):
    if condition == "none":
        return x
    if condition == "gaussian":
        return gaussian_jitter(x)
    if condition == "radial":
        return radial_family(x)
    if condition == "affine":
        return affine_family(x)
    raise ValueError(f"unknown condition: {condition!r}")
