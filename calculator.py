#!/usr/bin/env python3
"""
nn_tools_local.py

Edit the CONFIG block below, then run:
    python nn_tools_local.py

Features:
- Calculator: parameter counts, memory estimate, FLOPs, compare two models (e.g. 1024 -> 2048).
- Visualizer: draw neurons and weight connections with line thickness ~ |weight|,
  blue for positive, red for negative. Saves 'weights_viz.png'.
- Reduction helpers: pooling (contiguous) and k-means clustering to reduce neuron counts
  before visualizing large networks.

Dependencies: numpy, matplotlib. scikit-learn optional for k-means reduction.
"""

from typing import List, Tuple, Optional
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# optional sklearn
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ----------------------------
# CONFIG - edit these values
# ----------------------------
# Calculator settings
RUN_CALC = True
INPUT_SIZE = 784
HIDDEN_SIZES = [1024]    # list of hidden layers, e.g. [1024] or [1024, 512]
OUTPUT_SIZE = 10
BATCH_SIZE = 64
DTYPE_BYTES = 4  # float32

# Visualizer settings
RUN_VISUALIZE = True
W_PATHS = ['w1.npy', 'w2.npy']   # list of .npy files in order of layers (W shape = (out, in))
MAX_LINES = 2000000               # maximum number of connection lines to draw
PICK_TOP = 20000                # if set (int), draw only top-K absolute weights across all edges
NODE_SIZE = 18
FIGSIZE = (12, 8)
SAVE_PNG = True
PNG_PATH = 'weights_viz.png'

# Reduction options: None / 'pool' / 'kmeans'
REDUCTION_METHOD = 'kmeans'   # 'pool', 'kmeans', or None
# Target sizes for reduced layers (list length = number of layers including input & output after reduction)
# If None, automatic targets are chosen (e.g. cap hidden layers to 150 nodes)
REDUCTION_TARGETS = None

# Pooling (contiguous) settings
POOL_AGG = 'mean'  # 'mean' or 'sum'

# KMeans settings
KMEANS_AGG = 'mean'   # 'mean' or 'sum'
KMEANS_RANDOM_STATE = 0

# New: layout tuning (controls vertical spans)
MIN_SPAN_RATIO = 0.28   # minimum fraction of total plot height allocated to any layer
SPAN_SCALE_POW = 0.7    # scaling exponent for span growth; 1.0 = linear, <1 compresses differences

# Per-layer scaling so input/output can be visually compressed/enlarged
INPUT_SPAN_SCALE = 0.50   # multiply input neuron count by this before computing spans
OUTPUT_SPAN_SCALE = 0.60  # multiply output neuron count by this
# ----------------------------

# -----------------------
# Calculator utilities
# -----------------------
def calc_model_stats(input_size: int, hidden_sizes: List[int], output_size: int,
                     dtype_bytes: int = 4, batch_size: int = 64) -> dict:
    sizes = [input_size] + hidden_sizes + [output_size]
    num_layers = len(sizes) - 1

    params = []
    for i in range(num_layers):
        w = sizes[i+1] * sizes[i]
        b = sizes[i+1]
        params.append({'layer': i, 'in': sizes[i], 'out': sizes[i+1], 'w': w, 'b': b, 'total': w + b})

    total_params = sum(p['total'] for p in params)
    weights_bytes = total_params * dtype_bytes
    optimizer_bytes = weights_bytes * 2.0  # e.g. Adam m and v
    activation_elements = sum(s * batch_size for s in sizes)
    activation_bytes = activation_elements * dtype_bytes

    forward_flops = 0
    for i in range(num_layers):
        forward_flops += 2 * sizes[i] * sizes[i+1]  # multiply + add approx

    training_step_flops = forward_flops * 6.0  # rough estimate (forward+backprop+updates)

    return {
        'sizes': sizes,
        'num_layers': num_layers,
        'params_by_layer': params,
        'total_params': total_params,
        'weights_bytes': weights_bytes,
        'optimizer_bytes_estimate': optimizer_bytes,
        'activation_bytes_estimate_batch': activation_bytes,
        'forward_flops_per_sample': forward_flops,
        'training_flops_per_sample': training_step_flops,
        'batch_size': batch_size
    }

def pretty_print_stats(stats: dict):
    sizes = stats['sizes']
    print("Model shape (layer sizes):", sizes)
    print("Num layers:", stats['num_layers'])
    print("Parameters by layer (weights, bias, total):")
    for p in stats['params_by_layer']:
        print(f"  Layer {p['layer']}: {p['in']} -> {p['out']}: w={p['w']:,}, b={p['b']:,}, total={p['total']:,}")
    print("Total params:", f"{stats['total_params']:,}")
    print(f"Weights memory (float32): {stats['weights_bytes'] / (1024**2):.3f} MB")
    print(f"Optimizer buffers (Adam m+v) ~2x weights: {stats['optimizer_bytes_estimate'] / (1024**2):.3f} MB")
    print(f"Approx activations memory for batch {stats['batch_size']}: {stats['activation_bytes_estimate_batch'] / (1024**2):.3f} MB")
    print(f"Forward FLOPs per sample (approx): {stats['forward_flops_per_sample']:,}")
    print(f"Training FLOPs per sample (approx): {stats['training_flops_per_sample']:,}")
    print(f"Training FLOPs per batch (batch={stats['batch_size']}): {stats['training_flops_per_sample'] * stats['batch_size']:,}")
    print()

def compare_models(model_a: dict, model_b: dict) -> None:
    pa = model_a['total_params']
    pb = model_b['total_params']
    print("=== Comparison ===")
    print(f"Params: A={pa:,}, B={pb:,} (ratio B/A = {pb/pa:.3f})")
    fa = model_a['training_flops_per_sample']
    fb = model_b['training_flops_per_sample']
    print(f"Training FLOPs/sample: A={fa:,}, B={fb:,} (ratio B/A = {fb/fa:.3f})")
    mem_a = model_a['weights_bytes'] + model_a['optimizer_bytes_estimate'] + model_a['activation_bytes_estimate_batch']
    mem_b = model_b['weights_bytes'] + model_b['optimizer_bytes_estimate'] + model_b['activation_bytes_estimate_batch']
    print(f"Estimated memory (weights+opt+acts) A={mem_a/(1024**2):.1f} MB, B={mem_b/(1024**2):.1f} MB (ratio B/A = {mem_b/mem_a:.3f})")
    print(f"Estimated relative training time (B/A) ≈ {fb/fa:.3f}")
    print()

# -----------------------
# Reduction helpers
# -----------------------
def pool_reduce_weights(W: np.ndarray, out_pool: int, in_pool: int, agg: str = 'mean') -> np.ndarray:
    out, inn = W.shape
    if out_pool <= 0 or in_pool <= 0:
        raise ValueError("out_pool and in_pool must be > 0")
    out_sizes = np.full(out_pool, out // out_pool, dtype=int)
    out_sizes[:out % out_pool] += 1
    in_sizes = np.full(in_pool, inn // in_pool, dtype=int)
    in_sizes[:inn % in_pool] += 1

    out_idx = np.split(np.arange(out), np.cumsum(out_sizes)[:-1])
    in_idx = np.split(np.arange(inn), np.cumsum(in_sizes)[:-1])

    reduced = np.zeros((out_pool, in_pool), dtype=np.float32)
    for i, rows in enumerate(out_idx):
        for j, cols in enumerate(in_idx):
            block = W[np.ix_(rows, cols)]
            if block.size == 0:
                val = 0.0
            else:
                val = block.mean() if agg == 'mean' else block.sum()
            reduced[i, j] = val
    return reduced

def pool_reduce_weights_list(weights_list: List[np.ndarray], out_pools: List[int], in_pools: List[int], agg: str='mean') -> List[np.ndarray]:
    reduced = []
    for i, W in enumerate(weights_list):
        reduced.append(pool_reduce_weights(W, out_pools[i], in_pools[i], agg=agg))
    return reduced

def kmeans_reduce_weights(W: np.ndarray, target_out: int, target_in: int, agg: str='mean', random_state: int=0) -> np.ndarray:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn not available; install it or use pooling method.")
    out, inn = W.shape

    if target_out < out:
        km_out = KMeans(n_clusters=target_out, random_state=random_state).fit(W)
        labels_out = km_out.labels_
    else:
        labels_out = np.arange(out)

    if target_in < inn:
        km_in = KMeans(n_clusters=target_in, random_state=random_state).fit(W.T)
        labels_in = km_in.labels_
    else:
        labels_in = np.arange(inn)

    reduced = np.zeros((target_out, target_in), dtype=np.float32)
    for i in range(target_out):
        rows = np.where(labels_out == i)[0]
        for j in range(target_in):
            cols = np.where(labels_in == j)[0]
            block = W[np.ix_(rows, cols)]
            if block.size == 0:
                val = 0.0
            else:
                val = block.mean() if agg == 'mean' else block.sum()
            reduced[i, j] = val
    return reduced

def kmeans_reduce_weights_list(weights_list: List[np.ndarray], target_outs: List[int], target_ins: List[int], agg: str='mean', random_state: int=0) -> List[np.ndarray]:
    reduced = []
    for i, W in enumerate(weights_list):
        reduced.append(kmeans_reduce_weights(W, target_outs[i], target_ins[i], agg=agg, random_state=random_state))
    return reduced

def auto_targets_for_reduction(weights_list: List[np.ndarray], cap_hidden: int = 150) -> Tuple[List[int], List[int]]:
    out_pools = []
    in_pools = []
    for W in weights_list:
        out, inn = W.shape
        out_target = min(out, cap_hidden) if out > cap_hidden else out
        in_target = min(inn, cap_hidden) if inn > cap_hidden else inn
        out_pools.append(out_target)
        in_pools.append(in_target)
    return out_pools, in_pools

def reduce_weights_for_visualization(weights_list: List[np.ndarray],
                                     method: Optional[str],
                                     targets: Optional[List[int]] = None,
                                     agg_pool: str = 'mean',
                                     agg_kmeans: str = 'mean') -> List[np.ndarray]:
    if method is None:
        return weights_list

    if targets is None:
        out_pools, in_pools = auto_targets_for_reduction(weights_list, cap_hidden=150)
    else:
        if len(targets) != len(weights_list) + 1:
            raise ValueError("REDUCTION_TARGETS length must be number of matrices + 1 (includes input and final output sizes)")
        out_pools = []
        in_pools = []
        for i, W in enumerate(weights_list):
            in_pools.append(targets[i])
            out_pools.append(targets[i+1])

    if method == 'pool':
        return pool_reduce_weights_list(weights_list, out_pools, in_pools, agg=agg_pool)
    elif method == 'kmeans':
        return kmeans_reduce_weights_list(weights_list, out_pools, in_pools, agg=agg_kmeans, random_state=KMEANS_RANDOM_STATE)
    else:
        raise ValueError("Unknown reduction method: choose None, 'pool' or 'kmeans'")

# -----------------------
# Visualization utilities (updated layout)
# -----------------------
def _layer_positions(sizes: List[int], width: float = 1.0, height: float = 1.0) -> List[List[Tuple[float, float]]]:
    """
    Compute (x,y) positions for neurons in each layer.

    Uses per-layer effective sizes so input/output can be scaled separately from hidden layers:
      effective_n = n * INPUT_SPAN_SCALE  (if first layer)
      effective_n = n * OUTPUT_SPAN_SCALE (if last layer)
      effective_n = n                       (otherwise)
    Then spans are assigned proportional to effective_n ** SPAN_SCALE_POW, with a minimum span MIN_SPAN_RATIO.
    """
    layers_pos = []
    n_layers = len(sizes)
    x_coords = np.linspace(0, width, n_layers)

    # compute effective sizes with special scaling for first/last layers
    eff_sizes = []
    for idx, n in enumerate(sizes):
        if idx == 0:
            eff = max(1.0, n * INPUT_SPAN_SCALE)
        elif idx == (n_layers - 1):
            eff = max(1.0, n * OUTPUT_SPAN_SCALE)
        else:
            eff = max(1.0, n)
        eff_sizes.append(eff)

    max_eff = max(eff_sizes) if len(eff_sizes) > 0 else 1.0

    min_span = MIN_SPAN_RATIO * height
    variable_span = max(0.0, height - min_span)

    for i, n in enumerate(sizes):
        if n <= 0:
            layers_pos.append([])
            continue

        eff = eff_sizes[i]
        norm = float(eff) / float(max_eff) if max_eff > 0 else 1.0
        scale = norm ** SPAN_SCALE_POW

        span = min_span + scale * variable_span
        center = height / 2.0
        top = center + span / 2.0
        bottom = center - span / 2.0

        if n == 1:
            ys = np.array([center])
        else:
            ys = np.linspace(bottom, top, n)

        xs = np.full_like(ys, x_coords[i])
        layers_pos.append(list(zip(xs.tolist(), ys.tolist())))

    return layers_pos

def visualize_weights(weights_list: List[np.ndarray],
                      layer_names: Optional[List[str]] = None,
                      figsize=(12, 6),
                      node_size=20,
                      lw_min=0.2,
                      lw_max=4.0,
                      max_lines: int = 200000,
                      pick_top: Optional[int] = None,
                      title: str = "Network visualization",
                      save_png: bool = True,
                      png_path: str = 'weights_viz.png'):
    """
    Visualize fully connected layers given a list of weight matrices in order.
    weights_list: [W1, W2, ...] where W shape is (out, in)
    - lw_min, lw_max: min and max linewidth
    - max_lines: maximum number of lines to draw (will randomly subsample if too many)
    - pick_top: if set, choose the top-k absolute weights globally to plot (useful for huge matrices)
    """
    sizes = []
    for W in weights_list:
        sizes.append(W.shape[1])
    sizes.append(weights_list[-1].shape[0])

    layer_pos = _layer_positions(sizes, width=1.0, height=1.0)

    edges = []
    for li, W in enumerate(weights_list):
        out, inn = W.shape
        for i_out in range(out):
            for j_in in range(inn):
                w = float(W[i_out, j_in])
                edges.append((li, j_in, li+1, i_out, w))

    if pick_top is not None and pick_top < len(edges):
        edges = sorted(edges, key=lambda e: abs(e[4]), reverse=True)[:pick_top]
    else:
        if len(edges) > max_lines:
            random.seed(42)
            edges = random.sample(edges, max_lines)

    abs_weights = np.array([abs(e[4]) for e in edges], dtype=float)
    if abs_weights.size == 0:
        raise ValueError("No edges to plot (maybe weight matrices are empty?)")

    scaled = np.log1p(abs_weights)
    smin, smax = scaled.min(), scaled.max()
    if (smax - smin) > 1e-12:
        scaled_norm = (scaled - smin) / (smax - smin)
    else:
        scaled_norm = np.ones_like(scaled)

    segments = []
    colors = []
    widths = []
    for idx, e in enumerate(edges):
        li, j_in, li1, i_out, w = e
        if j_in >= len(layer_pos[li]) or i_out >= len(layer_pos[li1]):
            continue
        x0, y0 = layer_pos[li][j_in]
        x1, y1 = layer_pos[li1][i_out]
        segments.append([(x0, y0), (x1, y1)])
        colors.append('blue' if w >= 0 else 'red')
        widths.append(lw_min + (lw_max - lw_min) * scaled_norm[idx])

    fig, ax = plt.subplots(figsize=figsize)
    lc = LineCollection(segments, colors=colors, linewidths=widths, alpha=0.7, zorder=1)
    ax.add_collection(lc)

    for li, layer in enumerate(layer_pos):
        xs = [p[0] for p in layer]
        ys = [p[1] for p in layer]
        ax.scatter(xs, ys, s=node_size, c='k', zorder=2)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.tight_layout()

    if save_png:
        plt.savefig(png_path, dpi=200)
        print(f"Saved visualization to {png_path}")

    plt.show()

# -----------------------
# Main
# -----------------------
def main():
    # Calculator
    if RUN_CALC:
        print("=== Calculator ===")
        stats = calc_model_stats(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE, dtype_bytes=DTYPE_BYTES, batch_size=BATCH_SIZE)
        pretty_print_stats(stats)

        if len(HIDDEN_SIZES) == 1:
            hidden1024 = calc_model_stats(INPUT_SIZE, [HIDDEN_SIZES[0]], OUTPUT_SIZE, dtype_bytes=DTYPE_BYTES, batch_size=BATCH_SIZE)
            hidden2048 = calc_model_stats(INPUT_SIZE, [HIDDEN_SIZES[0]*2], OUTPUT_SIZE, dtype_bytes=DTYPE_BYTES, batch_size=BATCH_SIZE)
            print("If you double the hidden layer size:")
            compare_models(hidden1024, hidden2048)

    # Visualizer
    if RUN_VISUALIZE:
        if not W_PATHS or len(W_PATHS) == 0:
            print("Visualization skipped: set W_PATHS to your .npy weight files.")
            return

        weights_list = []
        for p in W_PATHS:
            print(f"Loading {p} ...")
            W = np.load(p)
            if W.ndim != 2:
                raise ValueError(f"Expected 2D matrix in {p}, got shape {W.shape}")
            weights_list.append(W)
            print("  shape:", W.shape)

        if REDUCTION_METHOD is None:
            reduced = weights_list
        else:
            if REDUCTION_TARGETS is None:
                reduced = reduce_weights_for_visualization(weights_list, REDUCTION_METHOD, targets=None,
                                                           agg_pool=POOL_AGG, agg_kmeans=KMEANS_AGG)
            else:
                reduced = reduce_weights_for_visualization(weights_list, REDUCTION_METHOD, targets=REDUCTION_TARGETS,
                                                           agg_pool=POOL_AGG, agg_kmeans=KMEANS_AGG)

        title = f"Weights visualization ({'reduced ' + REDUCTION_METHOD if REDUCTION_METHOD else 'full'})"
        if REDUCTION_METHOD in ('pool', 'kmeans'):
            sizes = [r.shape for r in reduced]
            print("Reduced weight shapes:", sizes)

        visualize_weights(reduced, figsize=FIGSIZE, node_size=NODE_SIZE,
                          max_lines=MAX_LINES, pick_top=PICK_TOP,
                          title=title, save_png=SAVE_PNG, png_path=PNG_PATH)

if __name__ == '__main__':
    main()
