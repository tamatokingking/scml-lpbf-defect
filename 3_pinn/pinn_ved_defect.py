"""
pinn_ved_defect.py
------------------
PINN for the VED -> defect area relationship.

The physics here is a simple nucleation ODE:

    df/dVED = -alpha * (f - f_eq(VED))

where f_eq(VED) = f_max * exp(-beta * (VED - VED_c)^2) is the
equilibrium defect fraction. Below VED_c (~30 J/mm^3 for 316L SS),
lack-of-fusion pores dominate so f_eq is large. Above VED_c the
melt pool consolidates and f_eq drops toward zero.

The PINN loss has two terms:
    L = L_data + lambda_phys * L_phys

L_data is standard MSE against noisy defect fraction measurements.
L_phys is the mean-squared ODE residual, evaluated at collocation
points (no labels needed there).

Everything here is self-contained. Synthetic observations are
generated from the ODE solution with added Gaussian noise.

Outputs go to pinn_outputs/:
    training_loss.png
    prediction.png
    residual.png
    lambda_ablation.png

Usage:
    python pinn_ved_defect.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

torch.manual_seed(42)
np.random.seed(42)

OUT = "pinn_outputs"
os.makedirs(OUT, exist_ok=True)

# physics constants -- calibrated to 316L SS literature
ALPHA    = 0.15     # relaxation rate  [mm^3 / J]
BETA     = 0.003    # Gaussian width
F_MAX    = 0.08     # max defect fraction
VED_C    = 30.0     # LoF threshold [J/mm^3]
VED_MIN  = 5.0
VED_MAX  = 120.0
NOISE    = 0.004


def f_eq(ved):
    return F_MAX * np.exp(-BETA * (ved - VED_C) ** 2)


def ode_rhs(ved, f):
    return -ALPHA * (f - f_eq(ved))


def solve_ode(n=300):
    ved = np.linspace(VED_MIN, VED_MAX, n)
    sol = solve_ivp(ode_rhs, (VED_MIN, VED_MAX), [F_MAX * 0.95],
                    t_eval=ved, method='RK45', rtol=1e-8, atol=1e-10)
    return sol.t, sol.y[0]


def make_obs(n=60, seed=0):
    rng = np.random.default_rng(seed)
    ved_true, f_true = solve_ode()

    ved_obs = rng.uniform(VED_MIN, VED_MAX, n)
    f_obs   = np.clip(np.interp(ved_obs, ved_true, f_true)
                      + rng.normal(0, NOISE, n), 0, 1)

    # oversample near LoF boundary since that's where things get interesting
    ved_lof = rng.uniform(VED_MIN, VED_C + 5, n // 3)
    f_lof   = np.clip(np.interp(ved_lof, ved_true, f_true)
                      + rng.normal(0, NOISE, len(ved_lof)), 0, 1)

    return (np.concatenate([ved_obs, ved_lof]),
            np.concatenate([f_obs, f_lof]),
            ved_true, f_true)


class PINN(nn.Module):
    def __init__(self, hidden=64, depth=4):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def data_loss(model, ved_t, f_t):
    return nn.functional.mse_loss(model(ved_t), f_t)


def phys_loss(model, ved_col):
    ved_col = ved_col.requires_grad_(True)
    f = model(ved_col)
    df = torch.autograd.grad(f, ved_col,
                             grad_outputs=torch.ones_like(f),
                             create_graph=True)[0]
    feq = torch.tensor(f_eq(ved_col.detach().cpu().numpy()),
                       dtype=torch.float32, device=ved_col.device)
    res = df + ALPHA * (f - feq.unsqueeze(1))
    return (res ** 2).mean()


def train(lam=1.0, n_epochs=5000, lr=5e-4, device='cpu', seed=42):
    torch.manual_seed(seed)
    scale = VED_MAX - VED_MIN

    ved_obs, f_obs, _, _ = make_obs(seed=seed)
    ved_t = torch.tensor((ved_obs - VED_MIN) / scale,
                         dtype=torch.float32, device=device).unsqueeze(1)
    f_t   = torch.tensor(f_obs,
                         dtype=torch.float32, device=device).unsqueeze(1)

    col_np = np.linspace(VED_MIN, VED_MAX, 500)
    col_t  = torch.tensor((col_np - VED_MIN) / scale,
                          dtype=torch.float32, device=device).unsqueeze(1)

    model = PINN().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)

    hist = []
    for ep in range(1, n_epochs + 1):
        opt.zero_grad()
        ld = data_loss(model, ved_t, f_t)
        lp = phys_loss(model, col_t.clone())
        loss = ld + lam * lp
        loss.backward()
        opt.step()
        sched.step()
        if ep % 500 == 0 or ep == 1:
            hist.append({'ep': ep, 'ld': ld.item(),
                         'lp': lp.item(), 'lt': loss.item()})
            print(f"  ep={ep:5d}  L_data={ld.item():.5f}"
                  f"  L_phys={lp.item():.5f}  lam={lam}")

    return model, hist, scale


@torch.no_grad()
def predict(model, ved_np, scale, device):
    t = torch.tensor((ved_np - VED_MIN) / scale,
                     dtype=torch.float32, device=device).unsqueeze(1)
    return model(t).cpu().numpy().flatten()


def ode_residual(model, ved_np, scale, device):
    t = torch.tensor((ved_np - VED_MIN) / scale,
                     dtype=torch.float32, device=device).unsqueeze(1)
    t = t.requires_grad_(True)
    f = model(t)
    df = torch.autograd.grad(f, t,
                             grad_outputs=torch.ones_like(f),
                             create_graph=False)[0]
    feq = torch.tensor(f_eq(ved_np), dtype=torch.float32,
                       device=device).unsqueeze(1)
    return (df + ALPHA * (f - feq)).detach().cpu().numpy().flatten()


def plot_loss(hist, lam):
    fig, ax = plt.subplots(figsize=(8, 4))
    eps = [h['ep'] for h in hist]
    ax.semilogy(eps, [h['ld'] for h in hist], label='L_data')
    ax.semilogy(eps, [h['lp'] for h in hist], label='L_phys')
    ax.semilogy(eps, [h['lt'] for h in hist], label='L_total', ls='--')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title(f'training loss  (lam={lam})')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/training_loss.png', dpi=150)
    plt.close()


def plot_pred(model, scale, device, lam):
    ved_plot = np.linspace(VED_MIN, VED_MAX, 400)
    ved_ode, f_ode = solve_ode()
    ved_obs, f_obs, _, _ = make_obs()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ved_ode, f_ode, 'k-', lw=2, label='ODE (ground truth)')
    ax.plot(ved_plot, predict(model, ved_plot, scale, device),
            'b--', lw=2, label=f'PINN (lam={lam})')
    ax.plot(ved_plot, f_eq(ved_plot), 'g:', lw=1.5, label='f_eq(VED)')
    ax.scatter(ved_obs, f_obs, c='red', s=15, alpha=0.5,
               label='observations', zorder=5)
    ax.axvline(VED_C, color='orange', ls='--', lw=1.5,
               label=f'VED_c = {VED_C}')
    ax.fill_betweenx([0, F_MAX * 1.1], VED_MIN, VED_C,
                     alpha=0.07, color='red', label='LoF zone')
    ax.set_xlabel('VED  [J/mm^3]')
    ax.set_ylabel('defect area fraction')
    ax.set_ylim(0, F_MAX * 1.15)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/prediction.png', dpi=150)
    plt.close()


def plot_residual(model, scale, device, lam):
    ved_plot = np.linspace(VED_MIN, VED_MAX, 300)
    res = ode_residual(model, ved_plot, scale, device)
    rms = np.sqrt(np.mean(res ** 2))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ved_plot, res, lw=2)
    ax.axhline(0, color='gray', ls='--')
    ax.fill_between(ved_plot, res, alpha=0.2)
    ax.set_xlabel('VED  [J/mm^3]')
    ax.set_ylabel('ODE residual')
    ax.set_title(f'physics residual  (lam={lam},  RMS={rms:.5f})')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/residual.png', dpi=150)
    plt.close()
    print(f'  residual RMS = {rms:.5f}')


def ablation(device):
    lambdas = [0.0, 0.1, 1.0, 10.0]
    colors  = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71']
    ved_plot = np.linspace(VED_MIN, VED_MAX, 400)
    ved_ode, f_ode = solve_ode()
    ved_obs, f_obs, _, _ = make_obs()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    mses = []

    for lam, col in zip(lambdas, colors):
        print(f'\n--- ablation: lam={lam} ---')
        model, _, scale = train(lam=lam, n_epochs=3000, device=device)
        fp   = predict(model, ved_plot, scale, device)
        ft   = np.interp(ved_plot, ved_ode, f_ode)
        mse  = float(np.mean((fp - ft) ** 2))
        mses.append(mse)
        axes[0].plot(ved_plot, fp, color=col, lw=2,
                     label=f'lam={lam}  MSE={mse:.5f}')

    axes[0].plot(ved_ode, f_ode, 'k-', lw=2.5, label='ODE', zorder=10)
    axes[0].scatter(ved_obs, f_obs, c='gray', s=12, alpha=0.4, zorder=5)
    axes[0].axvline(VED_C, color='orange', ls='--', lw=1.5)
    axes[0].set_xlabel('VED  [J/mm^3]')
    axes[0].set_ylabel('defect fraction')
    axes[0].set_title('effect of lambda_phys')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].bar([str(l) for l in lambdas], mses, color=colors, alpha=0.85)
    axes[1].set_xlabel('lambda_phys')
    axes[1].set_ylabel('MSE vs ODE')
    axes[1].set_title('ablation summary')
    for i, m in enumerate(mses):
        axes[1].text(i, m * 1.03, f'{m:.5f}', ha='center', fontsize=8)
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{OUT}/lambda_ablation.png', dpi=150)
    plt.close()
    return dict(zip(lambdas, mses))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    print('\n[1] main training run  (lam=1.0)')
    model, hist, scale = train(lam=1.0, n_epochs=5000, device=device)
    plot_loss(hist, lam=1.0)
    plot_pred(model, scale, device, lam=1.0)
    plot_residual(model, scale, device, lam=1.0)

    print('\n[2] lambda ablation')
    mse_table = ablation(device)

    print('\n--- ablation results ---')
    for lam, mse in mse_table.items():
        print(f'  lam={lam:<5}  MSE={mse:.6f}')
