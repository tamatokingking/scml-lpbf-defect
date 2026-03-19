"""
neural_ode_ved.py
-----------------
Neural ODE for layer-wise defect evolution in LPBF.

The physical picture: defects accumulate across layers. A void at
layer k can partially re-melt at layer k+1 if VED is sufficient,
or grow if VED is too low. The governing equation is

    df/dl = -gamma * VED(l) * f  +  s(VED(l))

where gamma is the healing rate and s models new defect nucleation
(large when VED < VED_c). This is tested in two ways:

    PureNODE  -- unconstrained MLP vector field, learns from data only
    PhysNODE  -- vector field has the healing-rate structure fixed;
                 only VED(l) and s(l) are learned

Synthetic trajectories are generated from the ODE with Gaussian
noise, so no image data is needed.

Requires torchdiffeq:  pip install torchdiffeq
Falls back to Euler integration if not installed.

Outputs go to neural_ode_outputs/:
    training.png
    comparison.png
    phase_portrait.png

Usage:
    python neural_ode_ved.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    print('torchdiffeq not found, using Euler fallback')
    print('install with: pip install torchdiffeq')

torch.manual_seed(0)
np.random.seed(0)

OUT      = "neural_ode_outputs"
os.makedirs(OUT, exist_ok=True)

GAMMA    = 0.02
A_SRC    = 0.001
VED_C    = 30.0
N_LAYERS = 100
VED_MEAN = 45.0
VED_STD  = 8.0


def ved_profile(layers):
    rng  = np.random.default_rng(7)
    base = VED_MEAN + 10 * np.sin(2 * np.pi * layers / N_LAYERS)
    return np.clip(base + rng.normal(0, VED_STD * 0.3, len(layers)),
                   VED_MEAN - 20, VED_MEAN + 25)


def src(ved):
    return A_SRC * np.maximum(0, VED_C - ved) / VED_C


def rhs(l, f, interp):
    ved = float(np.interp(l, *interp))
    return -GAMMA * ved * f[0] + src(ved)


def solve_true(f0=0.05, n=200):
    layers = np.linspace(0, N_LAYERS, n)
    ved    = ved_profile(layers)
    sol    = solve_ivp(rhs, [0, N_LAYERS], [f0],
                       t_eval=layers, args=((layers, ved),),
                       method='RK45', rtol=1e-8, atol=1e-10)
    return sol.t, sol.y[0], ved


def make_data(n_traj=12, noise=0.003):
    rng   = np.random.default_rng(42)
    trajs = []
    for _ in range(n_traj):
        f0 = rng.uniform(0.01, 0.10)
        layers, f_true, ved = solve_true(f0=f0)
        f_obs = np.clip(f_true + rng.normal(0, noise, len(f_true)), 0, 1)
        trajs.append({'layers': layers, 'f_obs': f_obs,
                      'f_true': f_true, 'ved': ved, 'f0': f0})
    return trajs


def euler(func, y0, t):
    ys = [y0]
    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        ys.append(ys[-1] + dt * func(t[i], ys[-1]))
    return torch.stack(ys)


class PureNODE(nn.Module):
    def __init__(self, h=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, 1)
        )

    def forward(self, t, y):
        t_n = (t / N_LAYERS).reshape(1)
        if y.dim() == 1:
            return self.net(torch.cat([y, t_n])).squeeze(0)
        return self.net(torch.cat([y, t_n.expand(y.shape[0], 1)], 1))


class PhysNODE(nn.Module):
    """
    df/dl = -gamma * VED_net(l) * f  +  src_net(l)
    VED_net and src_net are learned; the structure is fixed.
    """
    def __init__(self, h=32):
        super().__init__()
        self.ved_net = nn.Sequential(
            nn.Linear(1, h), nn.Tanh(), nn.Linear(h, 1), nn.Softplus())
        self.src_net = nn.Sequential(
            nn.Linear(1, h), nn.Tanh(), nn.Linear(h, 1), nn.Softplus())
        self.log_g = nn.Parameter(
            torch.tensor(float(np.log(GAMMA))))

    def forward(self, t, y):
        tn  = (t / N_LAYERS).reshape(1, 1)
        ved = self.ved_net(tn)
        s   = self.src_net(tn)
        g   = torch.exp(self.log_g)
        if y.dim() == 1:
            return (-g * ved.squeeze() * y + s.squeeze()).squeeze()
        return -g * ved * y + s.expand_as(y)


def integrate(func, f0, layers_np, device):
    lt = torch.tensor(layers_np, dtype=torch.float32, device=device)
    y0 = torch.tensor([f0], dtype=torch.float32, device=device)
    if HAS_TORCHDIFFEQ:
        out = odeint(func, y0, lt, method='rk4',
                     options={'step_size': 2.0})
        return out.squeeze(-1)
    return euler(func, y0, lt).squeeze(-1)


def loss_fn(func, trajs, device, n_sub=5):
    total = torch.tensor(0.0, device=device)
    idx   = np.random.choice(len(trajs), n_sub, replace=False)
    for i in idx:
        t  = trajs[i]
        lt = torch.tensor(t['layers'], dtype=torch.float32, device=device)
        ft = torch.tensor(t['f_obs'],  dtype=torch.float32, device=device)
        y0 = torch.tensor([t['f0']],   dtype=torch.float32, device=device)
        fp = integrate(func, t['f0'], t['layers'], device)
        total = total + nn.functional.mse_loss(fp, ft)
    return total / n_sub


def train(func, trajs, device, n_epochs=800, lr=1e-3, tag=''):
    opt   = torch.optim.Adam(func.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.5)
    hist  = []
    for ep in range(1, n_epochs + 1):
        opt.zero_grad()
        loss = loss_fn(func, trajs, device)
        loss.backward()
        nn.utils.clip_grad_norm_(func.parameters(), 1.0)
        opt.step()
        sched.step()
        if ep % 100 == 0 or ep == 1:
            hist.append({'ep': ep, 'loss': loss.item()})
            print(f'  [{tag}] ep={ep:4d}  loss={loss.item():.6f}')
    return hist


@torch.no_grad()
def predict(func, f0, layers_np, device):
    return integrate(func, f0, layers_np, device).cpu().numpy()


def plot_training(h_pure, h_phys):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy([h['ep'] for h in h_pure],
                [h['loss'] for h in h_pure], 'b-o', ms=4, label='pure NODE')
    ax.semilogy([h['ep'] for h in h_phys],
                [h['loss'] for h in h_phys], 'r-s', ms=4, label='phys NODE')
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE loss')
    ax.set_title('training loss')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/training.png', dpi=150)
    plt.close()


def plot_comparison(fp_pure, fp_phys, trajs, device):
    f0s = [0.02, 0.05, 0.09]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, f0 in zip(axes, f0s):
        layers, f_true, ved = solve_true(f0=f0)
        pu  = predict(fp_pure, f0, layers, device)
        ph  = predict(fp_phys, f0, layers, device)
        mpu = float(np.mean((pu - f_true) ** 2))
        mph = float(np.mean((ph - f_true) ** 2))

        ax.plot(layers, f_true, 'k-',  lw=2.5, label='true ODE')
        ax.plot(layers, pu,     'b--', lw=2,
                label=f'pure  MSE={mpu:.5f}')
        ax.plot(layers, ph,     'r-',  lw=2,
                label=f'phys  MSE={mph:.5f}')
        ax2 = ax.twinx()
        ax2.plot(layers, ved, color='gray', lw=1, alpha=0.35, ls=':')
        ax2.set_ylabel('VED', color='gray', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)
        ax.set_xlabel('layer')
        ax.set_ylabel('defect fraction')
        ax.set_title(f'f0 = {f0}')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle('pure NODE vs physics-constrained NODE', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{OUT}/comparison.png', dpi=150)
    plt.close()


def plot_phase(fp_pure, fp_phys, device):
    f_grid = np.linspace(0, 0.15, 100)
    l_mid  = torch.tensor(50.0, dtype=torch.float32, device=device)

    df_pu, df_ph, df_tr = [], [], []
    layers_ref = np.linspace(0, N_LAYERS, 200)
    ved_mid    = float(np.interp(50, layers_ref, ved_profile(layers_ref)))

    for fv in f_grid:
        ft = torch.tensor([fv], dtype=torch.float32, device=device)
        with torch.no_grad():
            df_pu.append(fp_pure(l_mid, ft).item())
            df_ph.append(fp_phys(l_mid, ft).item())
        df_tr.append(-GAMMA * ved_mid * fv + src(ved_mid))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(f_grid, df_tr, 'k-',  lw=2.5, label='true ODE')
    ax.plot(f_grid, df_pu, 'b--', lw=2,   label='pure NODE')
    ax.plot(f_grid, df_ph, 'r-',  lw=2,   label='phys NODE')
    ax.axhline(0, color='gray', ls=':', lw=1)
    ax.set_xlabel('defect fraction  f')
    ax.set_ylabel('df / dl')
    ax.set_title('phase portrait at layer 50')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/phase_portrait.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    print(f'torchdiffeq: {"yes" if HAS_TORCHDIFFEQ else "no (Euler fallback)"}')

    trajs = make_data(n_traj=12)
    print(f'{len(trajs)} trajectories, {len(trajs[0]["layers"])} points each')

    print('\n[1] pure NODE')
    fn_pure = PureNODE().to(device)
    h_pure  = train(fn_pure, trajs, device, n_epochs=800, tag='pure')

    print('\n[2] physics NODE')
    fn_phys = PhysNODE().to(device)
    h_phys  = train(fn_phys, trajs, device, n_epochs=800, tag='phys')

    plot_training(h_pure, h_phys)
    plot_comparison(fn_pure, fn_phys, trajs, device)
    plot_phase(fn_pure, fn_phys, device)

    print('\n--- test MSE ---')
    print(f'{"f0":<6}  {"pure":<12}  {"phys"}')
    for f0 in [0.02, 0.05, 0.09]:
        layers, ft, _ = solve_true(f0=f0)
        pu = predict(fn_pure, f0, layers, device)
        ph = predict(fn_phys, f0, layers, device)
        print(f'{f0:<6.2f}  {np.mean((pu-ft)**2):<12.6f}  '
              f'{np.mean((ph-ft)**2):.6f}')

    print(f'\nfigures saved to {OUT}/')
