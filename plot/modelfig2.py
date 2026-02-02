# -*- coding: utf-8 -*-
"""
Generate conference-style diagnostic plots:
1) Sparse adjacency heatmap A_s (C x C) with top-k per row and renorm
2) Value activation heatmap H_val,s (C x Tseg) showing non-stationary magnitude patterns
3) Gate value distribution (mostly near 0, few activations)

Run:
    python make_diagnostics_plots.py

Outputs:
    outputs/adjacency_heatmap.(png|pdf)
    outputs/value_heatmap.(png|pdf)
    outputs/gate_distribution.(png|pdf)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Utils
# ----------------------------
def set_global_style():
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 500,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
    })


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def row_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def topk_sparsify_and_renorm(A: np.ndarray, k: int) -> np.ndarray:
    """
    Keep top-k per row, set others to 0, then renormalize each row to sum to 1.
    A: (C, C), assumed non-negative
    """
    C = A.shape[0]
    A_sp = np.zeros_like(A)
    idx = np.argpartition(A, -k, axis=1)[:, -k:]  # top-k indices per row (unordered)
    rows = np.arange(C)[:, None]
    A_sp[rows, idx] = A[rows, idx]
    row_sum = A_sp.sum(axis=1, keepdims=True) + 1e-12
    A_sp = A_sp / row_sum
    return A_sp


def minmax_scale(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mn, mx = np.min(x), np.max(x)
    return (x - mn) / (mx - mn + eps)


# ----------------------------
# 1) Adjacency heatmap generator
# ----------------------------
def generate_structured_lowrank_adjacency(C: int, r: int, seed: int = 7) -> np.ndarray:
    """
    Build a 'reasonable' adjacency with:
      - low-rank factorization (U V^T)
      - mild cluster structure (few groups)
      - then row-softmax
    """
    rng = np.random.default_rng(seed)

    # Cluster assignment to induce block-ish relations
    n_groups = max(3, min(5, C // 4))
    group_ids = np.repeat(np.arange(n_groups), repeats=int(np.ceil(C / n_groups)))[:C]
    rng.shuffle(group_ids)

    # Group embeddings (r-dim) to create structure in U/V
    group_embed_U = rng.normal(0, 1.0, size=(n_groups, r))
    group_embed_V = rng.normal(0, 1.0, size=(n_groups, r))

    # Node factors with group bias + small noise
    U = group_embed_U[group_ids] + 0.25 * rng.normal(0, 1.0, size=(C, r))
    V = group_embed_V[group_ids] + 0.25 * rng.normal(0, 1.0, size=(C, r))

    # Similarity logits
    logits = (U @ V.T) / np.sqrt(r)

    # Optional: discourage self-loop dominance (keep diag but not too strong)
    logits = logits - 0.3 * np.eye(C)

    A = row_softmax(logits, axis=1)  # each row sums to 1
    return A


# ----------------------------
# 2) Value heatmap generator
# ----------------------------
def generate_value_segment(C: int, Tseg: int, seed: int = 11) -> np.ndarray:
    """
    Create H_val,s (C x Tseg) with:
      - per-channel baseline
      - mild trends
      - a few local events/bursts
      - different scales across channels (non-stationary magnitude preserved)
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, Tseg)

    # Channel baseline and scale
    base = rng.normal(0.0, 0.6, size=(C, 1))
    scale = rng.uniform(0.6, 1.6, size=(C, 1))

    # Slow drift per channel (trend)
    drift = rng.normal(0.0, 0.35, size=(C, 1)) * (t[None, :] - 0.5)

    # Seasonal-ish small oscillation (not too strong)
    freq = rng.integers(1, 4, size=(C, 1))
    phase = rng.uniform(0, 2*np.pi, size=(C, 1))
    osc = 0.25 * np.sin(2*np.pi*freq * t[None, :] + phase)

    # Local events: a few channels get short bursts
    events = np.zeros((C, Tseg))
    n_event_channels = max(2, C // 6)
    event_ch = rng.choice(C, size=n_event_channels, replace=False)
    for c in event_ch:
        center = rng.uniform(0.25, 0.85)
        width = rng.uniform(0.03, 0.08)
        amp = rng.uniform(0.8, 1.6) * rng.choice([1.0, -1.0], p=[0.75, 0.25])
        bump = amp * np.exp(-0.5 * ((t - center) / width) ** 2)
        events[c] += bump

    # Colored noise (smooth-ish)
    noise = rng.normal(0.0, 0.18, size=(C, Tseg))
    # simple smoothing by convolution
    kernel = np.array([0.2, 0.6, 0.2])
    noise_sm = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), 1, noise)

    H = (base + drift + osc + events) * scale + noise_sm
    return H


# ----------------------------
# 3) Gate generator
# ----------------------------
def generate_gate_values(C: int, seed: int = 23) -> np.ndarray:
    """
    Gate values: mostly near 0, few activated.
    Use a mixture distribution:
      - majority from a Beta concentrated near 0
      - a minority from a Beta shifted to moderate values
    """
    rng = np.random.default_rng(seed)
    n_active = max(2, C // 8)  # few active channels
    active_idx = rng.choice(C, size=n_active, replace=False)

    g = np.zeros(C, dtype=float)

    # Mostly tiny gates (near 0)
    g[:] = rng.beta(a=0.7, b=8.0, size=C) * 0.35  # in [0, ~0.35], skewed to 0

    # A few activated gates (moderate)
    g[active_idx] = 0.25 + rng.beta(a=2.0, b=4.0, size=n_active) * 0.55  # ~[0.25,0.80]

    # Add slight jitter, clamp
    g += rng.normal(0.0, 0.015, size=C)
    g = np.clip(g, 0.0, 1.0)
    return g


# ----------------------------
# Plotters
# ----------------------------
def plot_adjacency_heatmap(A_sp: np.ndarray, out_dir: str):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(A_sp, aspect="equal", interpolation="nearest")
    ax.set_title("Segment-wise Adjacency (top-k + renorm)")
    ax.set_xlabel("target channel j")
    ax.set_ylabel("source channel i")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("weight")

    # subtle grid to help readability (conference-style: light)
    C = A_sp.shape[0]
    ax.set_xticks(np.arange(-.5, C, 1), minor=True)
    ax.set_yticks(np.arange(-.5, C, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.3, alpha=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "adjacency_heatmap.png"), bbox_inches="tight", dpi=500)
    plt.close(fig)


def plot_value_heatmap(H_val: np.ndarray, out_dir: str):
    """
    Use robust normalization for visualization so it looks clean but still realistic.
    """
    # robust clip for better contrast without being "夸张"
    lo, hi = np.percentile(H_val, [2, 98])
    H_clip = np.clip(H_val, lo, hi)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.imshow(H_clip, aspect="auto", interpolation="nearest")
    ax.set_title("Value Activation Snapshot within One Segment")
    ax.set_xlabel("token index (within segment)")
    ax.set_ylabel("channel")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("magnitude (clipped)")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "value_heatmap.png"), bbox_inches="tight", dpi=500)
    plt.close(fig)


def plot_gate_distribution(g: np.ndarray, out_dir: str):
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.hist(g, bins=18, density=True, alpha=0.85)
    ax.set_title("Gate Value Distribution (safe fallback)")
    ax.set_xlabel("gate value")
    ax.set_ylabel("density")

    # Show mean line (simple, clean)
    mean_g = float(np.mean(g))
    ax.axvline(mean_g, linestyle="--", linewidth=1.2, label=f"mean={mean_g:.3f}")
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gate_distribution.png"), bbox_inches="tight", dpi=500)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    set_global_style()
    out_dir = "outputs"
    ensure_dir(out_dir)

    # Reasonable demo sizes
    C = 12         # number of channels/nodes
    r = 3          # low-rank factor dim
    topk = 3       # top-k sparsification per row
    Tseg = 48      # tokens in one segment

    # 1) adjacency: low-rank -> softmax -> top-k -> renorm
    A = generate_structured_lowrank_adjacency(C=C, r=r, seed=7)
    A_sp = topk_sparsify_and_renorm(A, k=topk)

    # 2) value: C x Tseg
    H_val = generate_value_segment(C=C, Tseg=Tseg, seed=11)

    # 3) gate: C
    g = generate_gate_values(C=C, seed=23)

    # Plot
    plot_adjacency_heatmap(A_sp, out_dir)
    plot_value_heatmap(H_val, out_dir)
    plot_gate_distribution(g, out_dir)

    # Print quick stats for sanity (optional)
    print("[Saved] outputs/adjacency_heatmap.png")
    print("[Saved] outputs/value_heatmap.png")
    print("[Saved] outputs/gate_distribution.png")
    print("\nSanity checks:")
    print(f"  A_sp row-sum (min/max): {A_sp.sum(axis=1).min():.4f} / {A_sp.sum(axis=1).max():.4f}")
    print(f"  A_sp sparsity (% zeros): {100.0*np.mean(A_sp == 0.0):.1f}%")
    print(f"  gate mean={g.mean():.3f}, gate>0.3 ratio={np.mean(g>0.3):.2f}")


if __name__ == "__main__":
    main()
