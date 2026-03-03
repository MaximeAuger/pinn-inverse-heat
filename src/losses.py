"""
Loss functions for the 1D inverse heat problem.

PDE  : -d²T/dx² = f(x),   x in (0,1)
BCs  : T(0) = T_left,  T(1) = T_right
Data : T(x_obs) ≈ T_measured  (sparse noisy observations)

Total loss = w_pde * L_pde  +  w_bc * L_bc  +  w_data * L_data  +  w_reg * L_reg
"""

import torch
import torch.nn as nn


def pde_residual(
    pinn: nn.Module,
    source_net: nn.Module,
    x_colloc: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the PDE residual  r(x) = d²T/dx² + f(x)
    using automatic differentiation.
    """
    x = x_colloc.requires_grad_(True)
    T = pinn(x)

    # First derivative
    dT_dx = torch.autograd.grad(
        T, x,
        grad_outputs=torch.ones_like(T),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Second derivative
    d2T_dx2 = torch.autograd.grad(
        dT_dx, x,
        grad_outputs=torch.ones_like(dT_dx),
        create_graph=True,
        retain_graph=True,
    )[0]

    f = source_net(x_colloc)
    residual = d2T_dx2 + f          # should be 0
    return residual


def boundary_loss(
    pinn: nn.Module,
    T_left: float,
    T_right: float,
) -> torch.Tensor:
    """Dirichlet BC loss at x=0 and x=1."""
    x0 = torch.tensor([[0.0]], requires_grad=False)
    x1 = torch.tensor([[1.0]], requires_grad=False)
    loss_bc = (pinn(x0) - T_left) ** 2 + (pinn(x1) - T_right) ** 2
    return loss_bc.squeeze()


def data_loss(
    pinn: nn.Module,
    x_obs: torch.Tensor,
    T_obs: torch.Tensor,
) -> torch.Tensor:
    """MSE between PINN predictions and noisy observations."""
    T_pred = pinn(x_obs)
    return torch.mean((T_pred - T_obs) ** 2)


def tikhonov_regularization(
    source_net: nn.Module,
    x_colloc: torch.Tensor,
    order: int = 1,
) -> torch.Tensor:
    """
    Tikhonov regularization of order 1 (penalises |df/dx|²)
    or order 0 (penalises |f|²).
    """
    x = x_colloc.requires_grad_(True)
    f = source_net(x)

    if order == 0:
        return torch.mean(f ** 2)

    # order == 1 : penalise gradient of f
    df_dx = torch.autograd.grad(
        f, x,
        grad_outputs=torch.ones_like(f),
        create_graph=True,
    )[0]
    return torch.mean(df_dx ** 2)


def total_loss(
    pinn: nn.Module,
    source_net: nn.Module,
    x_colloc: torch.Tensor,
    x_obs: torch.Tensor,
    T_obs: torch.Tensor,
    T_left: float = 0.0,
    T_right: float = 0.0,
    w_pde: float = 1.0,
    w_bc: float = 100.0,
    w_data: float = 100.0,
    w_reg: float = 1e-3,
    reg_order: int = 1,
) -> dict:
    """Return dict of individual losses and their weighted sum."""
    L_pde  = torch.mean(pde_residual(pinn, source_net, x_colloc) ** 2)
    L_bc   = boundary_loss(pinn, T_left, T_right)
    L_data = data_loss(pinn, x_obs, T_obs)
    L_reg  = tikhonov_regularization(source_net, x_colloc, order=reg_order)

    L_total = w_pde * L_pde + w_bc * L_bc + w_data * L_data + w_reg * L_reg

    return {
        "total": L_total,
        "pde":   L_pde.item(),
        "bc":    L_bc.item(),
        "data":  L_data.item(),
        "reg":   L_reg.item(),
    }
