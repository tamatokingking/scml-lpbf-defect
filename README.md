# Physics-Guided Invariant Learning for LPBF Defect Segmentation

ME 395 Scientific Machine Learning | Northwestern University | Spring 2026  
Ziming Zhao

---

The central question: if you build the VED governing equation into a defect
segmentation pipeline, does it actually help?

$$\text{VED} = \frac{P}{v \cdot h \cdot t}$$

Below about 30 J/mm³ (for 316L SS), powder does not fully melt and
lack-of-fusion pores form. Standard CV pipelines ignore this entirely.
Three mechanisms for embedding it are tested here.

---

## What the three parts do

**Part 1** runs a basic UNet hyperparameter sweep on the PB dataset
(2,638 images, EOS M290, 316L SS). Ten one-factor-at-a-time experiments
cover learning rate, loss function, optimiser, and threshold. This is the
CV/ML baseline with no physics involved.

**Part 2** uses NOTEARS causal structure learning to recover the graph
{P, v, h, t} → VED → defect from process measurements, then checks whether
the learned edge signs match the partial derivatives of the VED equation.
IRM then stratifies UNet training by VED regime, producing per-environment
Dice scores and optimal thresholds that are physically interpretable.

**Part 3** builds a PINN and a Neural ODE that encode the defect-nucleation
ODE as a constraint. Both are self-contained -- they generate synthetic data
from the physics and do not need the image dataset. The PINN ablation over
the physics penalty weight λ is the main result.

---

## Results

### Part 1 sweep

| Loss | lr | Dice | ROC AUC |
|---|---|---|---|
| **BCEDice** | **5e-4** | **0.787** | **0.9955** |
| Dice | 1e-4 | 0.790 | 0.9941 |
| BCE | 1e-4 | 0.766 | 0.9938 |
| Focal | 1e-4 | 0.757 | 0.9921 |

All 301 defective frames detected, 0 missed, 4 false alarms. Threshold
variation from 0.3 to 0.7 shifts Dice by less than 0.003.

### Part 2 NOTEARS

Recovered edges (signs match analytic partial derivatives of VED):

```
t  → VED  (+1.00)
h  → VED  (+1.00)
v  → VED  (-0.84)
P  → v    (-0.91)    operators co-vary P and v to hold target VED
VED → defect_area  (+0.33)
VED → regime       (+0.83)
```

### Part 2 IRM

| Regime | Dice | optimal τ |
|---|---|---|
| stable VED (30–80 J/mm³) | 0.633 | 0.85 |
| low VED (< 30 J/mm³) | 0.525 | 0.90 |

The τ shift makes sense physically: sintered powder at low VED has
inter-particle voids at the same spatial scale as LoF pores, so the model
needs higher confidence to avoid false positives on texture.

### Part 3 PINN ablation

| λ_phys | MSE vs ODE ground truth |
|---|---|
| 0.0 | highest — oscillates outside training range |
| 0.1 | reduced |
| **1.0** | **lowest** |
| 10.0 | slightly higher — underfits near LoF boundary |

---

## Repo structure

```
.
├── 1_baseline/
│   ├── train3_pb.py          UNet single run (BCEDice, Adam, lr=1e-4)
│   ├── sweep_pb2.py          10-experiment sweep
│   ├── visualize_pb.py       ROC/PR curves, confusion matrices
│   └── file_finder_pb.py     DataLoader
│
├── 2_physics_irm/
│   ├── notears_dag.py        NOTEARS + physics-constraint filtering
│   ├── physics_guided_train.py  IRM-augmented UNet
│   ├── physics_visualize.py  per-VED Dice, threshold curves, failure modes
│   ├── ved_metadata.py       filename -> VED parsing
│   └── file_finder_pb.py     DataLoader
│
├── 3_pinn/
│   ├── pinn_ved_defect.py    PINN: VED -> defect area (self-contained)
│   └── neural_ode_ved.py     Neural ODE: layer-wise dynamics (self-contained)
│
├── utils/
│   ├── file_finder_pb.py     canonical DataLoader
│   └── ved_metadata.py       VED utilities
│
└── report/
    ├── report.tex
    └── report.pdf
```

The PB image dataset is not included. Update `IMAGE_DIR` and `LABEL_DIR`
in each script to point to your local copy.

---

## Running

```bash
conda create -n scml python=3.10
conda activate scml
pip install -r requirements.txt

# baseline sweep (needs PB dataset)
cd 1_baseline
python sweep_pb2.py
python visualize_pb.py

# NOTEARS + IRM (needs PB dataset)
cd 2_physics_irm
python notears_dag.py
python physics_guided_train.py
python physics_visualize.py

# PINN and Neural ODE (no data needed, runs anywhere)
cd 3_pinn
python pinn_ved_defect.py
python neural_ode_ved.py
```

---

## References

- Ronneberger et al., U-Net, MICCAI 2015
- Zheng et al., DAGs with NO TEARS, NeurIPS 2018
- Arjovsky et al., Invariant Risk Minimization, arXiv 2019
- Raissi et al., Physics-Informed Neural Networks, J. Comput. Phys. 2019
- Chen et al., Neural Ordinary Differential Equations, NeurIPS 2018
- Lin et al., Focal Loss, ICCV 2017
