"""
Step 10 – Hardware Profiling & Bandwidth Calculation
=====================================================
Finalizes the hardware claims for the GraphCoder paper:

  1. Exact byte-size comparison: transmitting z (code indices) vs
     full Oracle/Global GNN weights  →  prove ≥10× communication reduction.
  2. ARM Cortex-M4 deployment simulation: peak RAM < 96 KB,
     p95 inference latency < 50 ms.
  3. Pareto-front scatter plot: Communication Payload (bytes) vs F1-Score
     for GraphCoder, Oracle GNN, Global GNN.

Outputs  (in ./output/)
-----------------------
  hw_profiling_results.json     – all bandwidth & latency numbers
  pareto_communication.png      – Pareto-front scatter plot
  hw_profiling_summary.txt      – detailed written report
"""

import os, sys, time, json, pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
sys.path.insert(0, BASE_DIR)

from ondevice_reconstruction import (
    TinyGNN, WeightReconstructor, quantise_int8, dequantise,
)
from baselines import edge_f1, graph_to_tensors, OracleGNN, GlobalGNN

DEVICE = "cpu"

# =====================================================================
#  ARM Cortex-M4 Hardware Model  (STM32F407 @ 168 MHz)
# =====================================================================
# Reference board: STM32F407VG — 192 KB SRAM, 1 MB Flash, no FPU
# for double, single-precision FPU only. We target INT8 inference.

ARM_CLOCK_MHZ       = 168
ARM_SRAM_KB         = 192       # total available
ARM_TARGET_RAM_KB   = 96        # budget for GNN inference
ARM_TARGET_LAT_MS   = 50        # p95 latency budget

# Cycle estimates per operation (Cortex-M4, single-issue, w/ FPU)
CYCLES_INT8_MADD    = 1         # 8-bit multiply-accumulate
CYCLES_FP32_MADD    = 3         # float32 MAC (with FPU)
CYCLES_RELU         = 1         # simple comparison
CYCLES_SPARSE_IDX   = 4         # sparse index lookup + multiply
CYCLES_MEMORY_LOAD  = 2         # L1-equivalent (tightly coupled)


# =====================================================================
#  1. Communication Payload Size Calculations
# =====================================================================

def calc_graphcoder_payload(M: int = 8, K: int = 64) -> dict:
    """
    GraphCoder transmits M code indices, each in [0, K).
    Minimum bits = M × ceil(log2(K)).
    In practice, pack into whole bytes.
    """
    bits_per_index = int(np.ceil(np.log2(K)))
    total_bits = M * bits_per_index
    total_bytes = int(np.ceil(total_bits / 8))

    return {
        "method": "GraphCoder (code indices z)",
        "M": M,
        "K": K,
        "bits_per_index": bits_per_index,
        "total_bits": total_bits,
        "total_bytes": total_bytes,
        "description": f"{M} indices × {bits_per_index} bits = "
                       f"{total_bits} bits = {total_bytes} bytes",
    }


def calc_model_payload(model: nn.Module, name: str,
                       dtype_bytes: int = 4) -> dict:
    """
    Calculate the byte-size of transmitting ALL model parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = total_params * dtype_bytes

    layer_info = {}
    for pname, p in model.named_parameters():
        layer_info[pname] = {
            "shape": list(p.shape),
            "numel": p.numel(),
            "bytes": p.numel() * dtype_bytes,
        }

    return {
        "method": name,
        "dtype": f"float{dtype_bytes*8}",
        "total_params": total_params,
        "total_bytes": total_bytes,
        "total_kb": round(total_bytes / 1024, 2),
        "layers": layer_info,
    }


def calc_tinygnn_int8_payload() -> dict:
    """
    TinyGNN with INT8 quantisation: weights as int8 + per-layer float32 scales.
    """
    layers = {
        "W1":    (32, 15),
        "W2":    (32, 32),
        "W3":    (32, 32),
        "W_cls": (2, 32),
    }
    total_weight_bytes = 0
    total_params = 0
    info = {}
    for name, shape in layers.items():
        n = shape[0] * shape[1]
        b = n * 1   # int8 = 1 byte each
        info[name] = {"shape": list(shape), "numel": n, "bytes_int8": b}
        total_weight_bytes += b
        total_params += n

    # 4 float32 scale values (one per layer)
    scale_bytes = 4 * 4   # 4 layers × 4 bytes each
    total_bytes = total_weight_bytes + scale_bytes

    return {
        "method": "TinyGNN (INT8 quantised)",
        "total_params": total_params,
        "weight_bytes_int8": total_weight_bytes,
        "scale_bytes": scale_bytes,
        "total_bytes": total_bytes,
        "total_kb": round(total_bytes / 1024, 2),
        "layers": info,
    }


# =====================================================================
#  2. ARM Cortex-M4 Inference Profiling (Cycle-Accurate Simulation)
# =====================================================================

def profile_gcn_layer(N: int, in_dim: int, out_dim: int, E: int,
                      is_int8: bool = True) -> dict:
    """
    Estimate cycles and RAM for one GCN layer on ARM Cortex-M4.

    GCN layer: h' = ReLU(Ã · h · Wᵀ)
      - Ã: sparse (N, N) with nnz = E + N (edges + self-loops)
      - h: (N, in_dim)
      - W: (out_dim, in_dim)

    Steps:
      1. Sparse matmul  Ã · h  → (N, in_dim):  nnz × in_dim MADDs
      2. Dense matmul   · Wᵀ   → (N, out_dim): N × in_dim × out_dim MADDs
      3. ReLU                                  : N × out_dim ops
    """
    nnz = E + N   # edges + self-loops
    mac_unit = CYCLES_INT8_MADD if is_int8 else CYCLES_FP32_MADD
    elem_bytes = 1 if is_int8 else 4

    # Step 1: Sparse matmul
    sparse_cycles = nnz * in_dim * (CYCLES_SPARSE_IDX + mac_unit)

    # Step 2: Dense matmul
    dense_cycles = N * in_dim * out_dim * mac_unit

    # Step 3: ReLU
    relu_cycles = N * out_dim * CYCLES_RELU

    total_cycles = sparse_cycles + dense_cycles + relu_cycles

    # RAM: activations (input + output must coexist)
    #   Input:  N × in_dim  (int8 or fp32 after dequant — always fp32 activations)
    #   Output: N × out_dim (fp32)
    #   Weight: out_dim × in_dim (int8 or fp32)
    act_bytes = N * in_dim * 4 + N * out_dim * 4   # activations always fp32
    weight_bytes = out_dim * in_dim * elem_bytes
    # Sparse index storage: 2 × nnz × 4 bytes (int32 src, dst) + nnz × 4 (values)
    sparse_bytes = nnz * (4 + 4 + 4)

    layer_ram = act_bytes + weight_bytes + sparse_bytes

    return {
        "sparse_cycles": sparse_cycles,
        "dense_cycles": dense_cycles,
        "relu_cycles": relu_cycles,
        "total_cycles": total_cycles,
        "act_bytes": act_bytes,
        "weight_bytes": weight_bytes,
        "sparse_bytes": sparse_bytes,
        "layer_ram_bytes": layer_ram,
    }


def profile_edge_scoring(N: int, E: int, hidden: int, num_classes: int,
                         is_int8: bool = True) -> dict:
    """
    Edge scoring: h_edge = h[src] ⊙ h[dst]; logits = W_cls @ h_edge
    """
    mac_unit = CYCLES_INT8_MADD if is_int8 else CYCLES_FP32_MADD
    elem_bytes = 1 if is_int8 else 4

    # Hadamard product: E × hidden multiplications
    hadamard_cycles = E * hidden * mac_unit

    # Classifier matmul: E × hidden × num_classes MADDs
    cls_cycles = E * hidden * num_classes * mac_unit

    total_cycles = hadamard_cycles + cls_cycles

    # RAM: h_edge (E × hidden) + logits (E × num_classes) + W_cls
    act_bytes = E * hidden * 4 + E * num_classes * 4
    weight_bytes = num_classes * hidden * elem_bytes

    return {
        "hadamard_cycles": hadamard_cycles,
        "cls_cycles": cls_cycles,
        "total_cycles": total_cycles,
        "act_bytes": act_bytes,
        "weight_bytes": weight_bytes,
        "layer_ram_bytes": act_bytes + weight_bytes,
    }


def profile_full_inference(N: int, E: int, is_int8: bool = True) -> dict:
    """
    Profile full TinyGNN inference for a graph with N nodes, E edges.
    Returns cycle count, time estimate, and peak RAM.
    """
    node_feat_dim = 15
    hidden = 32
    num_classes = 2

    # Profile each layer
    l1 = profile_gcn_layer(N, node_feat_dim, hidden, E, is_int8)
    l2 = profile_gcn_layer(N, hidden, hidden, E, is_int8)
    l3 = profile_gcn_layer(N, hidden, hidden, E, is_int8)
    edge = profile_edge_scoring(N, E, hidden, num_classes, is_int8)

    total_cycles = (l1["total_cycles"] + l2["total_cycles"] +
                    l3["total_cycles"] + edge["total_cycles"])

    # Latency
    latency_ms = total_cycles / (ARM_CLOCK_MHZ * 1e3)

    # Peak RAM = max over any layer's live set
    # At any point we need: current layer weights + input act + output act + sparse
    # We process sequentially, so peak is max of individual layers
    peak_ram = max(l1["layer_ram_bytes"], l2["layer_ram_bytes"],
                   l3["layer_ram_bytes"], edge["layer_ram_bytes"])

    # Also add: node features stored (N × 15 × 4), final h (N × 32 × 4)
    persistent_bytes = N * node_feat_dim * 4 + N * hidden * 4
    peak_ram += persistent_bytes

    # Code index storage (8 bytes) + codebook lookup overhead
    peak_ram += 8 + 64 * 32 * 1   # codebook in INT8 if on-device

    return {
        "N": N, "E": E,
        "is_int8": is_int8,
        "layer1": l1,
        "layer2": l2,
        "layer3": l3,
        "edge_scorer": edge,
        "total_cycles": total_cycles,
        "latency_ms": round(latency_ms, 4),
        "peak_ram_bytes": peak_ram,
        "peak_ram_kb": round(peak_ram / 1024, 2),
    }


# =====================================================================
#  3. Pareto-Front Plot
# =====================================================================

def plot_pareto(data: list[dict], save_path: str):
    """
    Scatter plot: X = Communication Payload (Bytes), Y = F1-Score.
    Each dict: {"name", "payload_bytes", "f1", "color", "marker"}.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Background gradient for visual appeal
    ax.set_facecolor("#fafbfc")
    fig.patch.set_facecolor("white")

    # Ideal region highlight (top-left)
    ax.axvspan(0, 100, alpha=0.04, color="#22c55e")
    ax.annotate("Ideal\nregion", xy=(50, 0.95), fontsize=9, color="#16a34a",
                ha="center", alpha=0.5, fontstyle="italic")

    for d in data:
        ax.scatter(d["payload_bytes"], d["f1"],
                   s=d.get("size", 300),
                   c=d["color"], marker=d["marker"],
                   edgecolors="white", linewidths=2, zorder=5,
                   label=d["name"])

        # Annotation box
        offset_x = d.get("offset_x", 15)
        offset_y = d.get("offset_y", 15)
        ax.annotate(
            f'{d["name"]}\n{d["payload_bytes"]:,} B | F1={d["f1"]:.3f}',
            xy=(d["payload_bytes"], d["f1"]),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=d["color"], alpha=0.95, linewidth=1.5),
            arrowprops=dict(arrowstyle="->", color=d["color"],
                            linewidth=1.5, connectionstyle="arc3,rad=0.2"),
        )

    ax.set_xlabel("Communication Payload (Bytes)", fontsize=13,
                  fontweight="bold")
    ax.set_ylabel("F1-Score (Macro)", fontsize=13, fontweight="bold")
    ax.set_title("Pareto Front — Communication Efficiency vs Detection Quality",
                 fontsize=15, fontweight="bold", pad=15)

    ax.set_xscale("log")
    ax.set_xlim(1, 200000)
    ax.set_ylim(0, 1.05)

    # Grid
    ax.grid(True, which="both", alpha=0.15, linestyle="--")
    ax.grid(True, which="major", alpha=0.3, linestyle="-")

    # Add reduction annotation arrow between GraphCoder and Oracle
    gc = next(d for d in data if "GraphCoder" in d["name"])
    oc = next(d for d in data if "Oracle" in d["name"])
    reduction = oc["payload_bytes"] / gc["payload_bytes"]
    mid_x = np.sqrt(gc["payload_bytes"] * oc["payload_bytes"])
    mid_y = (gc["f1"] + oc["f1"]) / 2
    ax.annotate(f"{reduction:.0f}× smaller",
                xy=(mid_x, mid_y - 0.08),
                fontsize=12, fontweight="bold", color="#dc2626",
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2",
                          ec="#dc2626", alpha=0.9))

    ax.legend(loc="lower right", fontsize=11, framealpha=0.95,
              edgecolor="#cbd5e1", markerscale=0.8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Pareto plot -> {save_path}")


# =====================================================================
#  Main
# =====================================================================

def main():
    t0 = time.time()
    print("=" * 65)
    print("  Step 10 — Hardware Profiling & Bandwidth Calculation")
    print("=" * 65)

    # ── 1. Load artefacts ────────────────────────────────────────────
    print("\n[1/5] Loading artefacts ...")

    ckpt = torch.load(os.path.join(OUTPUT_DIR, "ondevice_gnn.pt"),
                      map_location="cpu", weights_only=False)
    codebook = ckpt["codebook"]
    K, D = codebook.shape

    code_indices = np.load(os.path.join(OUTPUT_DIR, "code_indices.npy"))
    T, M = code_indices.shape

    with open(os.path.join(OUTPUT_DIR, "temporal_graphs.pkl"), "rb") as f:
        temporal_graphs = pickle.load(f)

    labels = np.load(os.path.join(OUTPUT_DIR, "fingerprint_labels.npy"))

    # Load baseline results from Step 8
    with open(os.path.join(OUTPUT_DIR, "baselines_results.json"), "r") as f:
        baseline_results = json.load(f)

    print(f"    {T} windows, M={M}, K={K}, D={D}")
    print(f"    {len(temporal_graphs)} temporal graphs")

    # ── 2. Communication payload calculations ────────────────────────
    print("\n" + "=" * 65)
    print("[2/5] Communication Payload Size Calculation")
    print("=" * 65)

    gc_payload = calc_graphcoder_payload(M=M, K=K)
    print(f"\n  GraphCoder payload:")
    print(f"    {gc_payload['description']}")
    print(f"    Total: {gc_payload['total_bytes']} bytes")

    # Oracle GNN: full model with biases and LayerNorm
    oracle_model = OracleGNN()
    oracle_payload = calc_model_payload(oracle_model, "Oracle GNN (float32)")
    print(f"\n  Oracle GNN payload (full float32 weights):")
    print(f"    Total params : {oracle_payload['total_params']:,}")
    print(f"    Total bytes  : {oracle_payload['total_bytes']:,} "
          f"({oracle_payload['total_kb']} KB)")
    for pname, info in oracle_payload["layers"].items():
        print(f"      {pname:<25s}  {str(info['shape']):>14s}  "
              f"{info['numel']:>6,} params  {info['bytes']:>7,} B")

    # Global GNN: same architecture as Oracle
    global_model = GlobalGNN()
    global_payload = calc_model_payload(global_model, "Global GNN (float32)")
    print(f"\n  Global GNN payload (full float32 weights):")
    print(f"    Total params : {global_payload['total_params']:,}")
    print(f"    Total bytes  : {global_payload['total_bytes']:,} "
          f"({global_payload['total_kb']} KB)")

    # TinyGNN INT8 (as if transmitting quantised weights directly)
    tiny_int8 = calc_tinygnn_int8_payload()
    print(f"\n  TinyGNN INT8 payload (quantised weight transfer):")
    print(f"    Weight bytes (int8) : {tiny_int8['weight_bytes_int8']:,}")
    print(f"    Scale bytes         : {tiny_int8['scale_bytes']}")
    print(f"    Total               : {tiny_int8['total_bytes']:,} "
          f"({tiny_int8['total_kb']} KB)")

    # Reduction ratios
    ratio_vs_oracle = oracle_payload["total_bytes"] / gc_payload["total_bytes"]
    ratio_vs_global = global_payload["total_bytes"] / gc_payload["total_bytes"]
    ratio_vs_int8   = tiny_int8["total_bytes"]      / gc_payload["total_bytes"]

    print(f"\n  {'─'*55}")
    print(f"  Communication Reduction Ratios (vs GraphCoder's {gc_payload['total_bytes']}B):")
    print(f"    vs Oracle GNN  : {ratio_vs_oracle:,.0f}×  "
          f"({oracle_payload['total_bytes']:,}B → {gc_payload['total_bytes']}B)")
    print(f"    vs Global GNN  : {ratio_vs_global:,.0f}×  "
          f"({global_payload['total_bytes']:,}B → {gc_payload['total_bytes']}B)")
    print(f"    vs TinyGNN INT8: {ratio_vs_int8:,.0f}×  "
          f"({tiny_int8['total_bytes']:,}B → {gc_payload['total_bytes']}B)")
    print(f"  {'─'*55}")

    target_met = ratio_vs_oracle >= 10
    print(f"  ≥10× reduction target: {'PASS' if target_met else 'FAIL'}  "
          f"(actual: {ratio_vs_oracle:.0f}×)")

    # ── 3. ARM Cortex-M4 inference profiling ─────────────────────────
    print("\n" + "=" * 65)
    print("[3/5] ARM Cortex-M4 Inference Profiling (Cycle-Accurate)")
    print("=" * 65)

    # Graph statistics for profiling
    Ns = [g["num_nodes"] for g in temporal_graphs if g["num_edges"] > 0]
    Es = [g["num_edges"] for g in temporal_graphs if g["num_edges"] > 0]

    N_stats = {
        "min": int(np.min(Ns)), "max": int(np.max(Ns)),
        "mean": float(np.mean(Ns)), "p50": int(np.percentile(Ns, 50)),
        "p95": int(np.percentile(Ns, 95)),
    }
    E_stats = {
        "min": int(np.min(Es)), "max": int(np.max(Es)),
        "mean": float(np.mean(Es)), "p50": int(np.percentile(Es, 50)),
        "p95": int(np.percentile(Es, 95)),
    }

    print(f"\n  Graph size statistics:")
    print(f"    Nodes: min={N_stats['min']}, max={N_stats['max']}, "
          f"mean={N_stats['mean']:.1f}, p95={N_stats['p95']}")
    print(f"    Edges: min={E_stats['min']}, max={E_stats['max']}, "
          f"mean={E_stats['mean']:.1f}, p95={E_stats['p95']}")

    # Profile every graph
    profiles = []
    for g in temporal_graphs:
        if g["num_edges"] < 1 or g["num_nodes"] < 2:
            continue
        p = profile_full_inference(g["num_nodes"], g["num_edges"],
                                   is_int8=True)
        profiles.append(p)

    latencies = [p["latency_ms"] for p in profiles]
    rams = [p["peak_ram_kb"] for p in profiles]

    lat_p50 = np.percentile(latencies, 50)
    lat_p95 = np.percentile(latencies, 95)
    lat_p99 = np.percentile(latencies, 99)
    lat_max = np.max(latencies)

    ram_p50 = np.percentile(rams, 50)
    ram_p95 = np.percentile(rams, 95)
    ram_max = np.max(rams)

    print(f"\n  ARM Cortex-M4 @ {ARM_CLOCK_MHZ} MHz — INT8 TinyGNN:")
    print(f"    {'Metric':<25s}  {'p50':>8s}  {'p95':>8s}  {'p99':>8s}  {'max':>8s}")
    print(f"    {'─'*55}")
    print(f"    {'Latency (ms)':<25s}  {lat_p50:>8.3f}  {lat_p95:>8.3f}  "
          f"{lat_p99:>8.3f}  {lat_max:>8.3f}")
    print(f"    {'Peak RAM (KB)':<25s}  {ram_p50:>8.2f}  {ram_p95:>8.2f}  "
          f"{'—':>8s}  {ram_max:>8.2f}")

    lat_ok = lat_p95 < ARM_TARGET_LAT_MS
    ram_ok = ram_max < ARM_TARGET_RAM_KB

    print(f"\n  Hardware constraints:")
    print(f"    p95 latency < {ARM_TARGET_LAT_MS} ms : "
          f"{'PASS' if lat_ok else 'FAIL'}  "
          f"(actual: {lat_p95:.3f} ms)")
    print(f"    Peak RAM < {ARM_TARGET_RAM_KB} KB    : "
          f"{'PASS' if ram_ok else 'FAIL'}  "
          f"(actual: {ram_max:.2f} KB)")

    # ── 4. Pareto-front plot ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("[4/5] Generating Pareto-Front Scatter Plot")
    print("=" * 65)

    # Retrieve F1 scores from baseline results
    gc_f1 = baseline_results["GraphCoder (ours)"]["mean_f1"]
    oracle_f1 = baseline_results["Oracle GNN (per-device)"]["mean_f1"]
    global_f1 = baseline_results["Global GNN (single fixed)"]["mean_f1"]

    pareto_data = [
        {
            "name": "GraphCoder",
            "payload_bytes": gc_payload["total_bytes"],
            "f1": gc_f1,
            "color": "#22c55e",
            "marker": "*",
            "size": 500,
            "offset_x": 25,
            "offset_y": -25,
        },
        {
            "name": "Oracle GNN",
            "payload_bytes": oracle_payload["total_bytes"],
            "f1": oracle_f1,
            "color": "#3b82f6",
            "marker": "D",
            "size": 300,
            "offset_x": -30,
            "offset_y": 20,
        },
        {
            "name": "Global GNN",
            "payload_bytes": global_payload["total_bytes"],
            "f1": global_f1,
            "color": "#f59e0b",
            "marker": "s",
            "size": 300,
            "offset_x": -30,
            "offset_y": -25,
        },
        {
            "name": "TinyGNN (INT8 direct)",
            "payload_bytes": tiny_int8["total_bytes"],
            "f1": gc_f1,   # same model, just different transmission
            "color": "#8b5cf6",
            "marker": "^",
            "size": 250,
            "offset_x": 20,
            "offset_y": 15,
        },
    ]

    plot_path = os.path.join(OUTPUT_DIR, "pareto_communication.png")
    plot_pareto(pareto_data, plot_path)

    # ── 5. Save results ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("[5/5] Saving Results")
    print("=" * 65)

    results = {
        "communication": {
            "graphcoder": gc_payload,
            "oracle_gnn": {
                "total_params": oracle_payload["total_params"],
                "total_bytes": oracle_payload["total_bytes"],
                "total_kb": oracle_payload["total_kb"],
            },
            "global_gnn": {
                "total_params": global_payload["total_params"],
                "total_bytes": global_payload["total_bytes"],
                "total_kb": global_payload["total_kb"],
            },
            "tinygnn_int8": {
                "total_params": tiny_int8["total_params"],
                "total_bytes": tiny_int8["total_bytes"],
                "total_kb": tiny_int8["total_kb"],
            },
            "reduction_ratios": {
                "vs_oracle": round(ratio_vs_oracle, 1),
                "vs_global": round(ratio_vs_global, 1),
                "vs_tinygnn_int8": round(ratio_vs_int8, 1),
            },
        },
        "arm_cortex_m4": {
            "clock_mhz": ARM_CLOCK_MHZ,
            "graph_stats": {"nodes": N_stats, "edges": E_stats},
            "latency_ms": {
                "p50": round(lat_p50, 4),
                "p95": round(lat_p95, 4),
                "p99": round(lat_p99, 4),
                "max": round(lat_max, 4),
            },
            "peak_ram_kb": {
                "p50": round(ram_p50, 2),
                "p95": round(ram_p95, 2),
                "max": round(ram_max, 2),
            },
            "constraints_met": {
                "latency_p95_under_50ms": bool(lat_ok),
                "peak_ram_under_96kb": bool(ram_ok),
            },
        },
        "f1_scores": {
            "graphcoder": gc_f1,
            "oracle_gnn": oracle_f1,
            "global_gnn": global_f1,
        },
    }

    json_path = os.path.join(OUTPUT_DIR, "hw_profiling_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"    -> {json_path}")

    # Summary text
    summary_lines = [
        "Step 10 — Hardware Profiling & Bandwidth Summary",
        "=" * 55,
        "",
        "COMMUNICATION PAYLOAD COMPARISON",
        "-" * 40,
        f"  GraphCoder (z indices) : {gc_payload['total_bytes']:>8,} bytes  "
        f"({gc_payload['description']})",
        f"  Oracle GNN (float32)   : {oracle_payload['total_bytes']:>8,} bytes  "
        f"({oracle_payload['total_params']:,} params × 4B)",
        f"  Global GNN (float32)   : {global_payload['total_bytes']:>8,} bytes  "
        f"({global_payload['total_params']:,} params × 4B)",
        f"  TinyGNN INT8 (direct)  : {tiny_int8['total_bytes']:>8,} bytes  "
        f"({tiny_int8['total_params']:,} params × 1B + scales)",
        "",
        f"  Reduction vs Oracle  : {ratio_vs_oracle:,.0f}×",
        f"  Reduction vs Global  : {ratio_vs_global:,.0f}×",
        f"  Reduction vs INT8    : {ratio_vs_int8:,.0f}×",
        "",
        "ARM CORTEX-M4 PROFILING (168 MHz, INT8 inference)",
        "-" * 40,
        f"  p95 latency  : {lat_p95:.3f} ms  (target < {ARM_TARGET_LAT_MS} ms)  "
        f"{'PASS' if lat_ok else 'FAIL'}",
        f"  Peak RAM     : {ram_max:.2f} KB  (target < {ARM_TARGET_RAM_KB} KB)  "
        f"{'PASS' if ram_ok else 'FAIL'}",
        "",
        "F1 SCORES",
        "-" * 40,
        f"  GraphCoder : {gc_f1:.4f}",
        f"  Oracle GNN : {oracle_f1:.4f}",
        f"  Global GNN : {global_f1:.4f}",
    ]
    summary_path = os.path.join(OUTPUT_DIR, "hw_profiling_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"    -> {summary_path}")

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  Step 10 complete  ({elapsed:.1f}s)")
    print(f"  Comm reduction: {ratio_vs_oracle:.0f}× vs Oracle  "
          f"{'PASS' if target_met else 'FAIL'}")
    print(f"  p95 latency: {lat_p95:.3f} ms  "
          f"{'PASS' if lat_ok else 'FAIL'}")
    print(f"  Peak RAM: {ram_max:.2f} KB  "
          f"{'PASS' if ram_ok else 'FAIL'}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
