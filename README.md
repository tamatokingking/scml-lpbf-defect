# Physics-Guided Invariant Learning for LPBF Defect Segmentation

**ME 395 – Scientific Machine Learning | Northwestern University | Spring 2026**

> *How much does knowing the physics actually help?*
> This project embeds the VED governing equation into a UNet training pipeline
> via causal structure learning (NOTEARS) and invariant risk minimisation (IRM),
> then benchmarks it against a small Physics-Informed Neural Network (PINN)
> that directly encodes the defect-nucleation physics.

---

## Overview

Laser Powder Bed Fusion (LPBF) is an additive manufacturing process where a
laser selectively melts metal powder layer by layer. The key process variable is
**Volumetric Energy Density (VED)**:

$$\text{VED} = \frac{P}{v \cdot h \cdot t}$$

where $P$ is laser power, $v$ scan speed, $h$ hatch spacing, and $t$ layer
thickness. When VED is too low, the powder fails to fully melt and
**lack-of-fusion (LoF)** pores appear. This project asks: if we tell the model
about VED, does segmentation improve—and can we *discover* the VED structure
from data without being told?

### Three-part framework

```
┌──────────────────────────────────────────────────────┐
│  Part 1 │  UNet Baseline + Hyperparameter Sweep      │
│         │  10 experiments (lr, loss, optimizer, τ)   │
├──────────────────────────────────────────────────────┤
│  Part 2 │  NOTEARS Causal Discovery + IRM Training   │
│         │  Learn P,v,h,t → VED → defect from data   │
│         │  IRM penalty enforces VED-env invariance   │
├──────────────────────────────────────────────────────┤
│  Part 3 │  PINN: defect area as a function of VED    │
│         │  Physics residual loss + Neural ODE toy    │
└──────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
.
├── README.md
│
├── 1_baseline/
│   ├── train3_pb.py            # Single-run UNet training (BCEDice, Adam, lr=1e-4)
│   ├── sweep_pb2.py            # 10-experiment hyperparameter sweep
│   └── visualize_pb.py         # ROC/PR curves, confusion matrices, predictions
│
├── 2_physics_irm/
│   ├── notears_dag.py          # NOTEARS DAG learning + physics-constraint filtering
│   ├── physics_guided_train.py # IRM-augmented UNet training (VED-stratified)
│   └── physics_visualize.py    # Per-VED-bin Dice, threshold curves, failure modes
│
├── 3_pinn/
│   ├── pinn_ved_defect.py      # PINN: learn defect_area(VED) with physics residual
│   └── neural_ode_ved.py       # Neural ODE: model VED evolution across layers
│
├── utils/
│   ├── file_finder_pb.py       # DataLoader for PB dataset (expected: user-provided)
│   └── ved_metadata.py         # Filename → VED parsing, regime assignment
│
├── results/                    # Auto-generated figures and CSVs (not tracked by git)
│   └── .gitkeep
│
└── report/
    └── report.pdf              # 2-page LaTeX report (submitted separately)
```

> **Note on data:** The PB dataset (EOS M290, 316L SS, 2,638 images) is not
> included due to size. Place images in `data/PB/` and labels in `data/PB_label/`
> and update the `IMAGE_DIR` / `LABEL_DIR` paths in each script.

---

## Scientific Machine Learning Angle

This project sits at the boundary of SciML and computer vision. The SciML
contributions are:

| Component | SciML concept | Where in code |
|---|---|---|
| NOTEARS on $(P,v,h,t,\text{VED})$ | Physics discovery via structure learning | `notears_dag.py` |
| IRM penalty stratified by VED regime | Invariant/causal representation learning | `physics_guided_train.py` |
| PINN with VED-defect residual | Physics-informed loss | `pinn_ved_defect.py` |
| Neural ODE on layer-wise VED | Continuous-depth dynamics modelling | `neural_ode_ved.py` |

The baseline UNet (Part 1) is standard CV/ML. Parts 2 and 3 incrementally add
physics in two different ways, letting us isolate how much each physics
injection actually helps.

---

## Results Summary

| Method | Dice | IoU | ROC AUC |
|---|---|---|---|
| UNet baseline (Exp 2, best sweep) | 0.787 | 0.680 | 0.9955 |
| UNet + IRM (stable VED) | 0.633 | — | — |
| UNet + IRM (low VED) | 0.525 | — | — |
| PINN regression (VED → defect area) | see `3_pinn/` | — | — |

The IRM Dice is lower than the baseline—intentionally. The IRM penalty
constrains the model to use features that are invariant across VED regimes,
trading raw metric gain for interpretable failure modes.

---

## Reproducing Results

### 1. Environment

```bash
conda create -n scml python=3.10
conda activate scml
pip install torch torchvision numpy matplotlib scikit-learn scipy torchdiffeq
```

### 2. Baseline sweep

```bash
cd 1_baseline
python sweep_pb2.py          # runs 10 experiments, saves pb_sweep_results.csv
python visualize_pb.py       # requires pb_best_model.pth from training
```

### 3. NOTEARS + IRM

```bash
cd 2_physics_irm
python notears_dag.py        # outputs physics_outputs/dag_*.png
python physics_guided_train.py
python physics_visualize.py
```

### 4. PINN + Neural ODE

```bash
cd 3_pinn
python pinn_ved_defect.py    # self-contained, no data needed
python neural_ode_ved.py     # self-contained, no data needed
```

The PINN and Neural ODE scripts are **self-contained**: they generate synthetic
data from the known physics and do not require the PB image dataset.

---

## Key References

- Ronneberger et al., *U-Net*, MICCAI 2015
- Zheng et al., *DAGs with NO TEARS*, NeurIPS 2018
- Arjovsky et al., *Invariant Risk Minimization*, arXiv 2019
- Raissi et al., *Physics-Informed Neural Networks*, J. Comput. Phys. 2019
- Chen et al., *Neural Ordinary Differential Equations*, NeurIPS 2018
