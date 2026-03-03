"""
Training utilities for the PINN inverse heat problem.
Supports:
  - Adam warm-up  →  L-BFGS fine-tuning  (standard PINN practice)
  - Loss history logging
  - Snapshot saving for GIF generation
"""

import torch
import torch.optim as optim
from torch import nn

from losses import total_loss


def train(
    pinn: nn.Module,
    source_net: nn.Module,
    x_colloc: torch.Tensor,
    x_obs: torch.Tensor,
    T_obs: torch.Tensor,
    # ── boundary conditions ──────────────────────────
    T_left: float = 0.0,
    T_right: float = 0.0,
    # ── loss weights ────────────────────────────────
    w_pde: float = 1.0,
    w_bc: float = 100.0,
    w_data: float = 100.0,
    w_reg: float = 1e-3,
    reg_order: int = 1,
    # ── optimiser schedule ──────────────────────────
    adam_epochs: int = 5000,
    lbfgs_epochs: int = 500,
    lr_adam: float = 1e-3,
    # ── snapshot settings (for GIF) ─────────────────
    snapshot_every: int = 100,
) -> dict:
    """
    Train PINN + source network.

    Returns
    -------
    history : dict with keys
        'loss', 'pde', 'bc', 'data', 'reg'  → lists of floats (Adam phase)
        'snapshots' → list of (epoch, x_eval, T_pred, f_pred) tuples
    """
    all_params = list(pinn.parameters()) + list(source_net.parameters())
    optimizer_adam = optim.Adam(all_params, lr=lr_adam)

    history = {"loss": [], "pde": [], "bc": [], "data": [], "reg": [], "snapshots": []}

    # ── evaluation grid for snapshots ────────────────
    x_eval = torch.linspace(0, 1, 300).unsqueeze(1)

    # ════════════════════════════════════════════════
    # Phase 1 : Adam
    # ════════════════════════════════════════════════
    print("Phase 1 — Adam optimiser")
    for epoch in range(1, adam_epochs + 1):
        optimizer_adam.zero_grad()
        losses = total_loss(
            pinn, source_net, x_colloc, x_obs, T_obs,
            T_left, T_right, w_pde, w_bc, w_data, w_reg, reg_order,
        )
        losses["total"].backward()
        optimizer_adam.step()

        history["loss"].append(losses["total"].item())
        history["pde"].append(losses["pde"])
        history["bc"].append(losses["bc"])
        history["data"].append(losses["data"])
        history["reg"].append(losses["reg"])

        # Snapshot
        if epoch % snapshot_every == 0 or epoch == 1:
            with torch.no_grad():
                T_snap = pinn(x_eval).squeeze().numpy()
                f_snap = source_net(x_eval).squeeze().numpy()
            history["snapshots"].append((epoch, x_eval.squeeze().numpy(), T_snap, f_snap))

        if epoch % 500 == 0:
            print(f"  Epoch {epoch:5d}  |  Loss {losses['total'].item():.4e}  "
                  f"PDE {losses['pde']:.2e}  Data {losses['data']:.2e}")

    # ════════════════════════════════════════════════
    # Phase 2 : L-BFGS fine-tuning
    # ════════════════════════════════════════════════
    print("\nPhase 2 — L-BFGS fine-tuning")
    optimizer_lbfgs = optim.LBFGS(
        all_params, lr=1.0, max_iter=20,
        history_size=50, line_search_fn="strong_wolfe",
    )

    lbfgs_losses = []

    def closure():
        optimizer_lbfgs.zero_grad()
        losses = total_loss(
            pinn, source_net, x_colloc, x_obs, T_obs,
            T_left, T_right, w_pde, w_bc, w_data, w_reg, reg_order,
        )
        losses["total"].backward()
        lbfgs_losses.append(losses["total"].item())
        return losses["total"]

    for step in range(lbfgs_epochs):
        optimizer_lbfgs.step(closure)
        if (step + 1) % 100 == 0:
            print(f"  L-BFGS step {step+1:4d}  |  Loss {lbfgs_losses[-1]:.4e}")

    # Final snapshot after L-BFGS
    with torch.no_grad():
        T_snap = pinn(x_eval).squeeze().numpy()
        f_snap = source_net(x_eval).squeeze().numpy()
    history["snapshots"].append(("final", x_eval.squeeze().numpy(), T_snap, f_snap))

    return history
