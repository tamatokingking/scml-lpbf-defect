"""
pinn_ved_defect.py
==================
Physics-Informed Neural Network for the VED → defect area relationship.

The governing physics:
    Defect area fraction f obeys a simplified nucleation model:

        df/dVED = -alpha * (f - f_eq(VED))

    where f_eq(VED) = f_max * exp(-beta * (VED - VED_critical)^2)
    is the equilibrium defect fraction at a given VED.

    At VED >> VED_critical: powder fully melts, f_eq → 0 (stable regime).
    At VED << VED_critical: lack-of-fusion pores dominate, f_eq → f_max.

The PINN loss has two terms:
    L = L_data  +  lambda_phys * L_physics

    L_data    = MSE between network prediction and observed defect fractions
    L_physics = MSE of the ODE residual  df/dVED - (-alpha*(f - f_eq))

This is a 1D PINN (input: VED scalar, output: defect fraction f).
Everything is self-contained; no image data required.

Usage:
    python pinn_ved_defect.py

Outputs (saved to pinn_outputs/):
    pinn_training_loss.png   – data vs physics loss curves
    pinn_prediction.png      – predicted f(VED) vs ground truth + physics ODE
    pinn_residual.png        – ODE residual across VED range
    pinn_lambda_ablation.png – effect of lambda_phys on prediction accuracy
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -----------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

OUT_DIR = "pinn_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# Physics parameters  (calibrated to LPBF 316L SS literature values)
# -----------------------------------------------------------------------
ALPHA        = 0.15    # relaxation rate [mm³/J]
BETA         = 0.003   # Gaussian width  [(mm³/J)^{-2}]
F_MAX        = 0.08    # max defect fraction (8%)
VED_CRIT     = 30.0    # critical VED [J/mm³] — LoF threshold
VED_MIN      = 5.0
VED_MAX      = 120.0
NOISE_STD    = 0.004   # measurement noise on defect fraction


# -----------------------------------------------------------------------
# Ground-truth physics: ODE solved with scipy
# -----------------------------------------------------------------------
def f_eq(ved):
    """Equilibrium defect fraction at given VED (numpy scalar or array)."""
    return F_MAX * np.exp(-BETA * (ved - VED_CRIT) ** 2)


def ode_rhs(ved, f):
    """df/dVED = -alpha * (f - f_eq(VED))"""
    return -ALPHA * (f - f_eq(ved))


def solve_physics_ode(ved_span=(VED_MIN, VED_MAX),
                      f0=F_MAX * 0.95,
                      n_eval=300):
    """Integrate the physics ODE to get the 'true' defect trajectory."""
    ved_eval = np.linspace(ved_span[0], ved_span[1], n_eval)
    sol = solve_ivp(ode_rhs, ved_span, [f0],
                    t_eval=ved_eval, method='RK45',
                    rtol=1e-8, atol=1e-10)
    return sol.t, sol.y[0]


# -----------------------------------------------------------------------
# Synthetic observation data
# (mimics what the UNet defect-area measurements give per parameter set)
# -----------------------------------------------------------------------
def make_observations(n_obs=60, noise_std=NOISE_STD, seed=0):
    """
    Sample VED values and compute noisy defect-area observations
    using the ODE solution as the ground truth.
    """
    rng = np.random.default_rng(seed)
    ved_true, f_true = solve_physics_ode()

    # Interpolate ODE solution at random VED points
    ved_obs = rng.uniform(VED_MIN, VED_MAX, n_obs)
    f_obs   = np.interp(ved_obs, ved_true, f_true)
    f_obs   = np.clip(f_obs + rng.normal(0, noise_std, n_obs), 0, 1)

    # Also include a few samples near the LoF boundary (important region)
    ved_lof = rng.uniform(VED_MIN, VED_CRIT + 5, n_obs // 3)
    f_lof   = np.interp(ved_lof, ved_true, f_true)
    f_lof   = np.clip(f_lof + rng.normal(0, noise_std, len(ved_lof)), 0, 1)

    ved_obs = np.concatenate([ved_obs, ved_lof])
    f_obs   = np.concatenate([f_obs,   f_lof])

    return ved_obs, f_obs, ved_true, f_true


# -----------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------
class PINN(nn.Module):
    """
    Small MLP: VED (scalar) → defect fraction f (scalar).
    Tanh activations work well for smooth physical solutions.
    """
    def __init__(self, hidden=64, depth=4):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1), nn.Sigmoid()]   # f ∈ [0, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, ved):
        return self.net(ved)


# -----------------------------------------------------------------------
# PINN losses
# -----------------------------------------------------------------------
def data_loss(model, ved_obs_t, f_obs_t):
    """MSE between predictions and noisy observations."""
    f_pred = model(ved_obs_t)
    return nn.functional.mse_loss(f_pred, f_obs_t)


def physics_loss(model, ved_col_t):
    """
    ODE residual loss:  r = df/dVED + alpha*(f - f_eq(VED))
    Evaluated on collocation points (no label needed).
    """
    ved_col_t = ved_col_t.requires_grad_(True)
    f_pred    = model(ved_col_t)

    # Automatic differentiation: df/dVED
    df_dved = torch.autograd.grad(
        f_pred, ved_col_t,
        grad_outputs=torch.ones_like(f_pred),
        create_graph=True
    )[0]

    # f_eq as a torch tensor
    ved_np  = ved_col_t.detach().cpu().numpy()
    feq_np  = f_eq(ved_np)
    feq_t   = torch.tensor(feq_np, dtype=torch.float32,
                            device=ved_col_t.device)

    # Residual
    alpha_t = torch.tensor(ALPHA, dtype=torch.float32, device=ved_col_t.device)
    residual = df_dved + alpha_t * (f_pred - feq_t.unsqueeze(1))
    return (residual ** 2).mean()


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------
def train_pinn(lambda_phys=1.0,
               n_epochs=5000,
               lr=5e-4,
               n_collocation=500,
               seed=42):
    """
    Train a PINN with a given physics penalty weight.
    Returns (model, history_dict).
    """
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ved_obs_np, f_obs_np, _, _ = make_observations(seed=seed)

    # Normalise VED to [0, 1] for numerical stability
    ved_scale = VED_MAX - VED_MIN

    ved_obs_t = torch.tensor(
        (ved_obs_np - VED_MIN) / ved_scale,
        dtype=torch.float32, device=device
    ).unsqueeze(1)
    f_obs_t   = torch.tensor(f_obs_np, dtype=torch.float32, device=device).unsqueeze(1)

    # Collocation points (uniform over VED range)
    ved_col_np = np.linspace(VED_MIN, VED_MAX, n_collocation)
    ved_col_t  = torch.tensor(
        (ved_col_np - VED_MIN) / ved_scale,
        dtype=torch.float32, device=device
    ).unsqueeze(1)

    model     = PINN(hidden=64, depth=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    history = {"epoch": [], "loss_data": [], "loss_phys": [], "loss_total": []}

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        l_data = data_loss(model, ved_obs_t, f_obs_t)
        l_phys = physics_loss(model, ved_col_t.clone())
        loss   = l_data + lambda_phys * l_phys
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0 or epoch == 1:
            history["epoch"].append(epoch)
            history["loss_data"].append(l_data.item())
            history["loss_phys"].append(l_phys.item())
            history["loss_total"].append(loss.item())
            print(f"  epoch={epoch:5d}  L_data={l_data.item():.6f}  "
                  f"L_phys={l_phys.item():.6f}  lambda={lambda_phys}")

    return model, history, device, ved_scale


# -----------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------
@torch.no_grad()
def predict(model, ved_np, ved_scale, device):
    ved_t = torch.tensor(
        (ved_np - VED_MIN) / ved_scale,
        dtype=torch.float32, device=device
    ).unsqueeze(1)
    return model(ved_t).cpu().numpy().flatten()


def compute_ode_residual_np(model, ved_np, ved_scale, device):
    """Compute ODE residual on a numpy VED grid (returns numpy array)."""
    ved_t = torch.tensor(
        (ved_np - VED_MIN) / ved_scale,
        dtype=torch.float32, device=device
    ).unsqueeze(1).requires_grad_(True)

    f_pred = model(ved_t)
    df_dved = torch.autograd.grad(
        f_pred, ved_t,
        grad_outputs=torch.ones_like(f_pred),
        create_graph=False
    )[0]

    feq_np = f_eq(ved_np)
    feq_t  = torch.tensor(feq_np, dtype=torch.float32, device=device).unsqueeze(1)
    res    = (df_dved + ALPHA * (f_pred - feq_t)).detach().cpu().numpy().flatten()
    return res


# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------
def plot_training_curves(history, lambda_phys, suffix=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.semilogy(history["epoch"], history["loss_data"],  label="L_data",  lw=2)
    ax1.semilogy(history["epoch"], history["loss_phys"],  label="L_phys",  lw=2)
    ax1.semilogy(history["epoch"], history["loss_total"], label="L_total", lw=2, ls='--')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss (log scale)")
    ax1.set_title(f"[PINN] Training Loss  (λ={lambda_phys})")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ratio = [p / (d + 1e-12) for d, p in
             zip(history["loss_data"], history["loss_phys"])]
    ax2.plot(history["epoch"], ratio, color="purple", lw=2)
    ax2.axhline(1.0, color="gray", ls="--", label="L_phys = L_data")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("L_phys / L_data")
    ax2.set_title("[PINN] Physics-to-Data Loss Ratio")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"pinn_training_loss{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")


def plot_prediction(model, ved_scale, device, lambda_phys, suffix=""):
    ved_plot = np.linspace(VED_MIN, VED_MAX, 400)
    f_pred   = predict(model, ved_plot, ved_scale, device)
    ved_ode, f_ode = solve_physics_ode()
    ved_obs_np, f_obs_np, _, _ = make_observations()
    f_eq_plot = f_eq(ved_plot)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ved_ode,   f_ode,     'k-',  lw=2,   label="Physics ODE (ground truth)")
    ax.plot(ved_plot,  f_pred,    'b--', lw=2,   label=f"PINN (λ={lambda_phys})")
    ax.plot(ved_plot,  f_eq_plot, 'g:',  lw=1.5, label="$f_{eq}$(VED) equilibrium")
    ax.scatter(ved_obs_np, f_obs_np, c='red', s=20, alpha=0.5, zorder=5,
               label="Noisy observations")
    ax.axvline(VED_CRIT, color='orange', ls='--', lw=1.5,
               label=f"VED_crit = {VED_CRIT} J/mm³")
    ax.fill_betweenx([0, F_MAX * 1.1], VED_MIN, VED_CRIT,
                     alpha=0.08, color='red', label="Lack-of-fusion zone")
    ax.set_xlabel("VED  [J/mm³]"); ax.set_ylabel("Defect area fraction")
    ax.set_title("[PINN] Defect Area vs VED")
    ax.set_ylim(0, F_MAX * 1.15); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"pinn_prediction{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")


def plot_residual(model, ved_scale, device, lambda_phys):
    ved_plot = np.linspace(VED_MIN, VED_MAX, 300)
    residual = compute_ode_residual_np(model, ved_plot, ved_scale, device)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ved_plot, residual, 'b-', lw=2)
    ax.axhline(0, color='gray', ls='--')
    ax.fill_between(ved_plot, residual, alpha=0.2)
    ax.set_xlabel("VED  [J/mm³]")
    ax.set_ylabel("ODE residual  $r = df/dVED + \\alpha(f - f_{eq})$")
    ax.set_title(f"[PINN] Physics Residual  (λ={lambda_phys})")
    ax.grid(True, alpha=0.3)
    rms = np.sqrt(np.mean(residual ** 2))
    ax.text(0.98, 0.95, f"RMS residual = {rms:.5f}",
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', fc='white', ec='gray'))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "pinn_residual.png")
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")


def plot_lambda_ablation():
    """
    Train PINNs with different lambda_phys values and compare predictions.
    This is the core ablation showing the effect of the physics penalty.
    """
    lambdas = [0.0, 0.1, 1.0, 10.0]
    colors  = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71']

    ved_plot    = np.linspace(VED_MIN, VED_MAX, 400)
    ved_ode, f_ode = solve_physics_ode()
    ved_obs_np, f_obs_np, _, _ = make_observations()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    mse_list = []
    for lam, col in zip(lambdas, colors):
        print(f"\n=== Ablation: lambda_phys = {lam} ===")
        model, history, device, ved_scale = train_pinn(
            lambda_phys=lam, n_epochs=3000, lr=5e-4)

        f_pred = predict(model, ved_plot, ved_scale, device)
        f_true = np.interp(ved_plot, ved_ode, f_ode)
        mse    = np.mean((f_pred - f_true) ** 2)
        mse_list.append(mse)

        axes[0].plot(ved_plot, f_pred, color=col, lw=2,
                     label=f"λ={lam}  (MSE={mse:.5f})")

    axes[0].plot(ved_ode, f_ode, 'k-', lw=2.5, label="Physics ODE (truth)", zorder=10)
    axes[0].scatter(ved_obs_np, f_obs_np, c='gray', s=15, alpha=0.4,
                    label="Observations", zorder=5)
    axes[0].axvline(VED_CRIT, color='orange', ls='--', lw=1.5)
    axes[0].set_xlabel("VED  [J/mm³]"); axes[0].set_ylabel("Defect fraction")
    axes[0].set_title("[PINN Ablation] Effect of Physics Penalty Weight λ")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].bar([str(l) for l in lambdas], mse_list, color=colors, alpha=0.85)
    axes[1].set_xlabel("λ_phys"); axes[1].set_ylabel("MSE vs ODE ground truth")
    axes[1].set_title("[PINN Ablation] MSE vs λ_phys")
    for i, (l, m) in enumerate(zip(lambdas, mse_list)):
        axes[1].text(i, m + max(mse_list) * 0.02,
                     f"{m:.5f}", ha='center', va='bottom', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "pinn_lambda_ablation.png")
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"\n  Saved: {path}")
    return dict(zip(lambdas, mse_list))


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("  PINN: VED → Defect Area (Physics-Informed Regression)")
    print("=" * 60)

    # ── Main training run ──
    print("\n[1] Training PINN with lambda_phys = 1.0 ...")
    model, history, device, ved_scale = train_pinn(
        lambda_phys=1.0, n_epochs=5000, lr=5e-4)

    print("\n[2] Plotting training curves ...")
    plot_training_curves(history, lambda_phys=1.0)

    print("\n[3] Plotting prediction vs ODE ground truth ...")
    plot_prediction(model, ved_scale, device, lambda_phys=1.0)

    print("\n[4] Plotting ODE residual ...")
    plot_residual(model, ved_scale, device, lambda_phys=1.0)

    # ── Ablation: lambda_phys ──
    print("\n[5] Lambda ablation study ...")
    mse_table = plot_lambda_ablation()

    print("\n" + "=" * 60)
    print("  PINN Results Summary")
    print("=" * 60)
    print("  λ_phys  |  MSE vs ODE")
    print("  --------|------------")
    for lam, mse in mse_table.items():
        print(f"  {lam:<7} |  {mse:.6f}")
    print(f"\n  All figures saved to ./{OUT_DIR}/")
    print("\n  Key takeaway:")
    print("  λ=0 (data-only) fits the observations but violates the ODE.")
    print("  λ>0 trades some data-fit for physics consistency.")
    print("  Optimal λ minimises MSE vs the true ODE trajectory.")
