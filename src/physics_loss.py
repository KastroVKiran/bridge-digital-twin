# src/physics_loss.py
"""
Differentiable fatigue-relevant loss term: mean(|successive stress
difference|^m), a standard narrow-band-process fatigue damage surrogate,
fully differentiable and computed every training step.

E_STEEL_MPA and SN_SLOPE_M are kept consistent with src/fatigue_physics.py
(Phase 5) for direct conceptual correspondence, though the proxy is computed
in whatever units the input tensor is in (normalized or physical) -- since
successive-difference-based proxies preserve relative ordering under linear
rescaling, this is valid as a training signal in normalized space and does
NOT need to be de-normalized at every training step (only the periodic real
rainflow validation check, Section 2, requires de-normalized physical units).
"""

import torch

SN_SLOPE_M = 3  # matches src/fatigue_physics.py DETAIL_CATEGORIES slope assumption


def differentiable_damage_proxy(strain: torch.Tensor, m: int = SN_SLOPE_M) -> torch.Tensor:
    """
    strain: (batch, sequence_length, n_channels)
    returns: (batch, n_channels) -- a fatigue-relevant scalar proxy per channel per event
    """
    diffs = strain[:, 1:, :] - strain[:, :-1, :]
    proxy = torch.mean(torch.abs(diffs) ** m, dim=1)
    return proxy


def physics_loss(strain_pred: torch.Tensor, strain_true: torch.Tensor, m: int = SN_SLOPE_M) -> torch.Tensor:
    proxy_pred = differentiable_damage_proxy(strain_pred, m)
    proxy_true = differentiable_damage_proxy(strain_true, m)
    return torch.nn.functional.mse_loss(proxy_pred, proxy_true)


if __name__ == "__main__":
    # Sanity check: identical signals should give zero physics loss;
    # different signals should give a positive, finite value.
    torch.manual_seed(0)
    a = torch.randn(2, 1000, 8)
    b = a.clone()
    c = torch.randn(2, 1000, 8)

    loss_identical = physics_loss(a, b)
    loss_different = physics_loss(a, c)

    print(f"Physics loss (identical signals): {loss_identical.item():.6e} (should be ~0.0)")
    print(f"Physics loss (different signals): {loss_different.item():.6e} (should be > 0)")
    assert loss_identical.item() < 1e-6, "Identical signals should give near-zero physics loss"
    assert loss_different.item() > loss_identical.item(), "Different signals should give higher physics loss"
    print("Sanity checks passed.")
