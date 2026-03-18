"""
ved_metadata.py
===============
Utilities for parsing LPBF PB dataset filenames and assigning
VED-based environment IDs for IRM training.

Filename convention (EOS M290 dataset):
    <set_id>_<frame_id>.jpg
    e.g.  set1A_0042.jpg   →  parameter set 1A, frame 42

VED values are looked up from the known parameter table (Table 3 in the
original dataset paper). If the set_id is not recognised, a default
stable-VED is assigned.

VED regimes (environments for IRM):
    env 0: low    (VED < 30 J/mm³)  — lack-of-fusion regime
    env 1: stable (30 ≤ VED ≤ 80)  — optimal processing window
    env 2: high   (VED > 80 J/mm³) — keyhole / over-melting regime
"""

import re

# ---------------------------------------------------------------------------
# Known VED values for each parameter set
# (P [W], v [mm/s], h [mm], t [µm])  → VED = P / (v * h * t*1e-3)
# ---------------------------------------------------------------------------
VED_REF = {
    "set1A": 8.89,
    "set1B": 17.81,
    "set1C": 26.71,
    "set1D": 35.61,
    "set2":  55.15,
    "set3":  100.03,
    # Fallback if set cannot be identified
    "unknown": 45.0,
}

# VED regime boundaries [J/mm³]
VED_LOW_THRESH    = 30.0
VED_HIGH_THRESH   = 80.0


def ved_to_env_id(ved: float) -> int:
    """
    Convert a VED value to an IRM environment ID.

    Returns
    -------
    0  →  low-VED (lack-of-fusion)
    1  →  stable-VED (optimal)
    2  →  high-VED (keyhole)
    """
    if ved < VED_LOW_THRESH:
        return 0
    elif ved <= VED_HIGH_THRESH:
        return 1
    else:
        return 2


def parse_filename(filename: str) -> dict:
    """
    Parse a PB dataset filename and return metadata dict.

    Parameters
    ----------
    filename : str
        e.g. 'set1A_0042.jpg' or 'set2_0137.png'

    Returns
    -------
    dict with keys:
        set_id   : str   – parameter set label (e.g. 'set1A')
        frame_id : int   – frame index
        ved      : float – volumetric energy density [J/mm³]
        regime   : str   – 'low' | 'stable' | 'high'
        env_id   : int   – IRM environment index (0, 1, 2)
    """
    # Try to match known pattern  setXY_NNNN.ext
    match = re.search(r'(set\d+[A-Za-z]?)[\s_\-](\d+)', filename, re.IGNORECASE)
    if match:
        set_id   = match.group(1).lower()
        frame_id = int(match.group(2))
    else:
        set_id   = "unknown"
        frame_id = 0

    # Normalise set_id capitalisation
    set_id_norm = set_id[0].upper() + set_id[1:]  # 'set1a' → 'Set1a'
    # Match against VED_REF keys case-insensitively
    ved = None
    for key in VED_REF:
        if key.lower() == set_id:
            ved = VED_REF[key]
            break
    if ved is None:
        ved = VED_REF["unknown"]

    env_id = ved_to_env_id(ved)
    regime_names = {0: "low", 1: "stable", 2: "high"}

    return {
        "set_id":   set_id,
        "frame_id": frame_id,
        "ved":      ved,
        "regime":   regime_names[env_id],
        "env_id":   env_id,
    }


def get_regime_label(env_id: int) -> str:
    return {0: "low", 1: "stable", 2: "high"}.get(env_id, "unknown")


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_files = [
        "set1A_0001.jpg",
        "set1B_0042.jpg",
        "set1C_0100.jpg",
        "set1D_0200.jpg",
        "set2_0001.jpg",
        "set3_0001.jpg",
        "unknown_frame_0050.jpg",
    ]
    print(f"{'Filename':<30} {'set_id':<8} {'VED':>7} {'regime':<8} {'env_id'}")
    print("-" * 65)
    for f in test_files:
        m = parse_filename(f)
        print(f"{f:<30} {m['set_id']:<8} {m['ved']:>7.2f} {m['regime']:<8} {m['env_id']}")
