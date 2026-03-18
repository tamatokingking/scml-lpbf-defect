"""
notears_dag.py
==============
Stage 1 of the Physics-Guided Stability framework:
    Learn a directed acyclic graph (DAG) over LPBF process variables
    using the NOTEARS continuous optimisation method.

Variables modelled
------------------
  x0 : laser_power  (P, W)
  x1 : scan_speed   (v, mm/s)
  x2 : layer_thick  (t, µm)
  x3 : hatch_space  (h, mm)
  x4 : VED          (computed: P / (v*h*t))
  x5 : defect_area  (observed from predicted masks, pixels)
  x6 : regime_code  (0=low, 1=stable, 2=high)

Reference: Zheng et al. "DAGs with NO TEARS", NeurIPS 2018.
"""

import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# -------------------------------------------------------------------
# NOTEARS acyclicity constraint  h(W) = tr(e^{W°W}) - d = 0
# -------------------------------------------------------------------
def h_func(W: torch.Tensor) -> torch.Tensor:
    """Acyclicity constraint from NOTEARS."""
    d = W.shape[0]
    M = W * W                              # element-wise square
    E = torch.matrix_exp(M)               # matrix exponential
    return E.trace() - d


def notears_loss(W: torch.Tensor,
                 X: torch.Tensor,
                 lambda1: float = 0.01,
                 lambda2: float = 1.0) -> tuple:
    """
    L(W) = ||X - X W||²_F  +  λ1 ||W||_1  +  λ2 h(W)
    Returns (total_loss, recon_loss, h_value)
    """
    recon  = 0.5 * torch.norm(X - X @ W, p='fro') ** 2 / X.shape[0]
    l1     = lambda1 * W.abs().sum()
    h_val  = h_func(W)
    penalty = lambda2 * h_val
    return recon + l1 + penalty, recon, h_val


def fit_notears(X: np.ndarray,
                lambda1: float = 0.01,
                lambda2_init: float = 1.0,
                lr: float = 1e-3,
                max_iter: int = 3000,
                h_tol: float = 1e-8,
                augment_steps: int = 5) -> np.ndarray:
    """
    Fit NOTEARS DAG to data matrix X  (n_samples × d_vars).
    Uses augmented-Lagrangian outer loop to tighten acyclicity.

    Returns W_est: (d × d) weighted adjacency matrix.
    """
    n, d = X.shape
    X_t  = torch.tensor(X, dtype=torch.float32)

    W = torch.zeros(d, d, requires_grad=True)
    lambda2 = lambda2_init

    print(f"  NOTEARS: {d} variables, {n} samples, augmented-Lagrangian steps={augment_steps}")

    for outer in range(augment_steps):
        optimizer = optim.Adam([W], lr=lr)

        for step in range(max_iter):
            optimizer.zero_grad()
            loss, recon, h_val = notears_loss(W, X_t, lambda1, lambda2)
            loss.backward()
            # Zero out diagonal (no self-loops)
            with torch.no_grad():
                W.grad.fill_diagonal_(0.0)
            optimizer.step()
            with torch.no_grad():
                W.fill_diagonal_(0.0)

            if step % 500 == 0:
                print(f"    outer={outer} step={step:4d}  "
                      f"recon={recon.item():.4f}  h={h_val.item():.6f}")

        h_val = h_func(W).item()
        print(f"  outer={outer} finished  h={h_val:.8f}  lambda2={lambda2:.2f}")

        if abs(h_val) < h_tol:
            print("  Acyclicity constraint satisfied.")
            break
        lambda2 *= 10   # tighten penalty

    W_est = W.detach().numpy().copy()
    W_est[np.abs(W_est) < 0.1] = 0.0   # threshold small edges
    np.fill_diagonal(W_est, 0.0)
    return W_est


# -------------------------------------------------------------------
# Stage 2: Physics-constraint filtering
# -------------------------------------------------------------------
VARIABLE_NAMES = ["P", "v", "t", "h", "VED", "defect_area", "regime"]

def apply_physics_constraints(W: np.ndarray) -> np.ndarray:
    """
    Filter the learned DAG using thermophysical rules:

    Forbidden edges (violate physics or temporal order):
      - defect_area  → P / v / t / h / VED  (effect cannot cause input)
      - regime       → P / v / t / h        (derived label cannot cause inputs)
      - VED          → P / v / t / h        (VED is a derived quantity)

    Required edges (add if missing):
      - P   → VED  (VED = P / (v*h*t))
      - v   → VED
      - t   → VED
      - h   → VED
      - VED → regime
      - VED → defect_area   (VED drives defect formation)
    """
    idx = {name: i for i, name in enumerate(VARIABLE_NAMES)}
    W_f = W.copy()

    # ------ Remove forbidden edges ------
    effects   = ["defect_area", "regime", "VED"]
    inputs    = ["P", "v", "t", "h"]
    for eff in effects:
        for inp in inputs:
            W_f[idx[eff], idx[inp]] = 0.0   # eff → inp  forbidden

    W_f[idx["VED"],         idx["regime"]]      = 0.0  # VED cannot be caused by regime
    W_f[idx["defect_area"], idx["regime"]]      = 0.0
    W_f[idx["regime"],      idx["defect_area"]] = 0.0  # regime is a label, not a cause

    # ------ Enforce required edges ------
    required = [
        ("P", "VED"), ("v", "VED"), ("t", "VED"), ("h", "VED"),
        ("VED", "regime"), ("VED", "defect_area"),
    ]
    for src, dst in required:
        if W_f[idx[src], idx[dst]] == 0.0:
            W_f[idx[src], idx[dst]] = 1.0   # add with unit weight

    return W_f


def visualize_dag(W: np.ndarray, title: str, save_path: str):
    """Draw the DAG using matplotlib (no graphviz dependency)."""
    d = len(VARIABLE_NAMES)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.5, 3.5); ax.axis('off')
    ax.set_title(title, fontsize=13)

    # Fixed layout positions
    positions = {
        "P":           (0.0, 3.0),
        "v":           (1.0, 3.0),
        "t":           (2.0, 3.0),
        "h":           (3.0, 3.0),
        "VED":         (1.5, 2.0),
        "regime":      (0.5, 1.0),
        "defect_area": (2.5, 1.0),
    }

    # Draw edges
    for i, src in enumerate(VARIABLE_NAMES):
        for j, dst in enumerate(VARIABLE_NAMES):
            if W[i, j] != 0.0:
                x0, y0 = positions[src]
                x1, y1 = positions[dst]
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="->",
                                           color="steelblue", lw=1.5))
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                ax.text(mx, my, f"{W[i,j]:.2f}", fontsize=7, color="gray")

    # Draw nodes
    for name, (x, y) in positions.items():
        color = "#ff9999" if name in ("defect_area", "regime") else "#aaddff"
        ax.add_patch(plt.Circle((x, y), 0.28, color=color, zorder=3))
        ax.text(x, y, name, ha='center', va='center', fontsize=9,
                fontweight='bold', zorder=4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved DAG: {save_path}")


# -------------------------------------------------------------------
# Build synthetic dataset from known parameter sets
# (replace with real extracted features when available)
# -------------------------------------------------------------------
def build_process_dataset(n_per_set: int = 200,
                          defect_stats: dict = None,
                          seed: int = 42) -> np.ndarray:
    """
    Generate a (n_samples × 7) matrix with columns:
        [P, v, t, h, VED, defect_area, regime_code]

    defect_stats: dict mapping set_name → (mean_area, std_area)
                  obtained from running the trained UNet on each set.
                  If None, uses physics-motivated defaults.
    """
    rng = np.random.default_rng(seed)

    # Parameter sets  (from Table 3 in failure_case_analysis.pdf)
    param_configs = [
        # (P,  v,    t,  h,     VED,   defect_mean, defect_std)
        ( 80,  3120, 40, 0.09,  8.89,  0.010,       0.008),   # set1A ultra-low
        (130,  2080, 40, 0.09, 17.81,  0.015,       0.010),   # set1B low
        (180,  1560, 40, 0.09, 26.71,  0.018,       0.012),   # set1C low
        (250,   780, 40, 0.09, 35.61,  0.020,       0.010),   # set1D moderate
        (215,  1083, 40, 0.09, 55.15,  0.012,       0.008),   # set2  stable
        (195,  1083, 20, 0.09,100.03,  0.025,       0.015),   # set3  keyhole
    ]

    if defect_stats is not None:
        # Override defaults with real measured stats
        for k, (mean_a, std_a) in defect_stats.items():
            if 0 <= k < len(param_configs):
                cfg        = list(param_configs[k])
                cfg[5]     = mean_a
                cfg[6]     = std_a
                param_configs[k] = tuple(cfg)

    rows = []
    for (P, v, t, h, ved, d_mean, d_std) in param_configs:
        # Add Gaussian noise to process params (instrument / run-to-run)
        P_s   = rng.normal(P,   P   * 0.02, n_per_set)
        v_s   = rng.normal(v,   v   * 0.02, n_per_set)
        t_s   = rng.normal(t,   t   * 0.01, n_per_set)
        h_s   = rng.normal(h,   h   * 0.01, n_per_set)
        ved_s = P_s / (v_s * h_s * (t_s / 1000))   # recompute VED

        # Defect area (normalised pixel fraction)
        d_area = np.clip(rng.normal(d_mean, d_std, n_per_set), 0, 1)

        # Regime code (derived)
        regime = np.where(ved_s < 30, 0, np.where(ved_s < 80, 1, 2))

        rows.append(np.stack([P_s, v_s, t_s, h_s, ved_s, d_area, regime.astype(float)], axis=1))

    X = np.vstack(rows)
    # Standardise each column
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X


if __name__ == "__main__":
    print("=== Stage 1: NOTEARS DAG Learning ===")
    X = build_process_dataset(n_per_set=300)
    print(f"  Dataset shape: {X.shape}")

    W_raw = fit_notears(X, lambda1=0.01, lambda2_init=1.0,
                        lr=1e-3, max_iter=2000, augment_steps=5)

    print("\n=== Stage 2: Physics Constraint Filtering ===")
    W_phys = apply_physics_constraints(W_raw)

    print("\nLearned W (raw, thresholded):")
    print(np.round(W_raw, 3))
    print("\nPhysics-filtered W:")
    print(np.round(W_phys, 3))

    os.makedirs("physics_outputs", exist_ok=True)
    visualize_dag(W_raw,   "NOTEARS DAG (Raw)",              "physics_outputs/dag_raw.png")
    visualize_dag(W_phys,  "Physics-Filtered DAG",           "physics_outputs/dag_filtered.png")
    np.save("physics_outputs/W_raw.npy",  W_raw)
    np.save("physics_outputs/W_phys.npy", W_phys)
    print("\nDAG learning complete. Results in physics_outputs/")
