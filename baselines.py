"""
Step 8 – Rigorous Baselines for GraphCoder Comparison
======================================================
Four baselines for edge-level binary classification (normal / attack):

  1. Oracle GNN   — per-graph fully fine-tuned GCN  (upper bound)
  2. Global GNN   — single fixed GCN trained on all graphs  (compressed deploy)
  3. LSTM          — non-graph temporal baseline on node-pair features
  4. TCN           — temporal convolution baseline on node-pair features

All models share the same hidden dimension (32) and edge scoring scheme
for a fair comparison against the GraphCoder pipeline.

Outputs  (in ./output/)
-----------------------
  baselines_results.json        – per-baseline metrics
  baselines_comparison.png      – bar chart comparing all methods
  baselines_summary.txt         – detailed report
"""

import os, sys, time, json, copy, warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
sys.path.insert(0, BASE_DIR)

from ondevice_reconstruction import TinyGNN, WeightReconstructor

DEVICE      = "cpu"
NODE_FEAT   = 15
HIDDEN      = 32
NUM_CLASSES = 2


# =====================================================================
#  Shared GCN normalisation  (reuse TinyGNN static method)
# =====================================================================
gcn_norm = TinyGNN._gcn_norm


# =====================================================================
#  Shared helpers
# =====================================================================

def edge_accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def edge_f1(logits, labels):
    """Compute macro-F1 for binary edge classification."""
    preds = logits.argmax(dim=1)
    f1s = []
    for c in range(NUM_CLASSES):
        tp = ((preds == c) & (labels == c)).sum().float()
        fp = ((preds == c) & (labels != c)).sum().float()
        fn = ((preds != c) & (labels == c)).sum().float()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        f1s.append(f1.item())
    return np.mean(f1s)


def graph_to_tensors(g):
    nf = torch.tensor(g["node_feat"], dtype=torch.float32)
    ei = torch.tensor(g["edge_index"], dtype=torch.long)
    ey = torch.tensor(g["edge_y"], dtype=torch.long)
    return nf, ei, ey


# =====================================================================
#  1.  ORACLE GNN  —  Per-graph fully fine-tuned  (upper bound)
# =====================================================================

class OracleGNN(nn.Module):
    """
    3-layer GCN with LayerNorm, fully fine-tuned per-graph.
    Represents the upper bound: unlimited on-device compute budget.
    """

    def __init__(self, in_dim=NODE_FEAT, hidden=HIDDEN, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.ln3 = nn.LayerNorm(hidden)
        self.cls = nn.Linear(hidden, num_classes)

    def forward(self, node_feat, edge_index):
        N = node_feat.size(0)
        A = gcn_norm(edge_index, N)

        h = F.relu(self.ln1(torch.sparse.mm(A, node_feat) @ self.fc1.weight.t() + self.fc1.bias))
        h = F.relu(self.ln2(torch.sparse.mm(A, h) @ self.fc2.weight.t() + self.fc2.bias))
        h = F.relu(self.ln3(torch.sparse.mm(A, h) @ self.fc3.weight.t() + self.fc3.bias))

        src, dst = edge_index[0], edge_index[1]
        h_edge = h[src] * h[dst]
        logits = self.cls(h_edge)
        return logits


def train_oracle_on_graph(graph, epochs=200, lr=1e-2):
    """Fine-tune a fresh OracleGNN on a single graph's edges."""
    nf, ei, ey = graph_to_tensors(graph)
    if ey.size(0) == 0:
        return None, 0.0, 0.0

    model = OracleGNN()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        logits = model(nf, ei)
        loss = F.cross_entropy(logits, ey)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        logits = model(nf, ei)
        acc = edge_accuracy(logits, ey)
        f1  = edge_f1(logits, ey)
    return model, acc, f1


# =====================================================================
#  2.  GLOBAL GNN  —  Single fixed model for all devices
# =====================================================================

class GlobalGNN(nn.Module):
    """
    Same architecture as Oracle but trained on pooled data from ALL graphs,
    then deployed identically (frozen) everywhere.
    """

    def __init__(self, in_dim=NODE_FEAT, hidden=HIDDEN, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.ln3 = nn.LayerNorm(hidden)
        self.cls = nn.Linear(hidden, num_classes)

    def forward(self, node_feat, edge_index):
        N = node_feat.size(0)
        A = gcn_norm(edge_index, N)

        h = F.relu(self.ln1(torch.sparse.mm(A, node_feat) @ self.fc1.weight.t() + self.fc1.bias))
        h = F.relu(self.ln2(torch.sparse.mm(A, h) @ self.fc2.weight.t() + self.fc2.bias))
        h = F.relu(self.ln3(torch.sparse.mm(A, h) @ self.fc3.weight.t() + self.fc3.bias))

        src, dst = edge_index[0], edge_index[1]
        h_edge = h[src] * h[dst]
        logits = self.cls(h_edge)
        return logits


def train_global_gnn(graphs, epochs=100, lr=1e-3):
    """Train one GlobalGNN on all graphs (mini-epoch = one pass over all graphs)."""
    model = GlobalGNN()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        n = 0
        for g in graphs:
            nf, ei, ey = graph_to_tensors(g)
            if ey.size(0) == 0:
                continue
            logits = model(nf, ei)
            loss = F.cross_entropy(logits, ey)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total_loss += loss.item()
            n += 1
        if ep % 20 == 0 or ep == 1:
            print(f"    Global GNN ep {ep:>3d}/{epochs}  loss={total_loss/max(n,1):.4f}")

    return model


def eval_global_gnn(model, graphs):
    """Evaluate frozen GlobalGNN on each graph individually."""
    model.eval()
    accs, f1s = [], []
    for g in graphs:
        nf, ei, ey = graph_to_tensors(g)
        if ey.size(0) == 0:
            continue
        with torch.no_grad():
            logits = model(nf, ei)
            accs.append(edge_accuracy(logits, ey))
            f1s.append(edge_f1(logits, ey))
    return accs, f1s


# =====================================================================
#  3.  LSTM Baseline  —  Non-graph, operates on node-pair features
# =====================================================================

class EdgeLSTM(nn.Module):
    """
    LSTM over a sequence of node-pair feature vectors.

    For each edge (u, v) we form a feature vector:
        x_edge = [x_u ; x_v ; x_u * x_v]   ∈ R^{3·d}

    Within a graph, edges form a sequence fed to an LSTM.
    """

    def __init__(self, in_dim=NODE_FEAT, hidden=HIDDEN, num_classes=NUM_CLASSES):
        super().__init__()
        self.edge_dim = in_dim * 3   # concat + hadamard
        self.lstm = nn.LSTM(self.edge_dim, hidden, num_layers=1,
                            batch_first=True)
        self.cls = nn.Linear(hidden, num_classes)

    def forward(self, node_feat, edge_index):
        src, dst = edge_index[0], edge_index[1]
        x_src = node_feat[src]
        x_dst = node_feat[dst]
        x_edge = torch.cat([x_src, x_dst, x_src * x_dst], dim=1)  # (E, 3d)

        # Treat as single-batch sequence (1, E, 3d)
        x_seq = x_edge.unsqueeze(0)
        out, _ = self.lstm(x_seq)         # (1, E, hidden)
        out = out.squeeze(0)              # (E, hidden)
        logits = self.cls(out)            # (E, 2)
        return logits


def train_lstm_global(graphs, epochs=100, lr=1e-3):
    model = EdgeLSTM()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, epochs + 1):
        total_loss, n = 0.0, 0
        for g in graphs:
            nf, ei, ey = graph_to_tensors(g)
            if ey.size(0) == 0:
                continue
            logits = model(nf, ei)
            loss = F.cross_entropy(logits, ey)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total_loss += loss.item()
            n += 1
        if ep % 20 == 0 or ep == 1:
            print(f"    LSTM ep {ep:>3d}/{epochs}  loss={total_loss/max(n,1):.4f}")
    return model


def eval_lstm(model, graphs):
    model.eval()
    accs, f1s = [], []
    for g in graphs:
        nf, ei, ey = graph_to_tensors(g)
        if ey.size(0) == 0:
            continue
        with torch.no_grad():
            logits = model(nf, ei)
            accs.append(edge_accuracy(logits, ey))
            f1s.append(edge_f1(logits, ey))
    return accs, f1s


# =====================================================================
#  4.  TCN Baseline  —  Temporal Convolution on node-pair features
# =====================================================================

class TemporalConvBlock(nn.Module):
    """Causal 1D convolution block with residual."""

    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation   # causal padding
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (B, C, L)
        out = self.conv(x)
        # Trim to causal (remove future positions)
        out = out[:, :, :x.size(2)]
        # LayerNorm expects (B, L, C)
        out = self.ln(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)
        return out + x   # residual


class EdgeTCN(nn.Module):
    """
    Temporal convolution network over edge-feature sequences.
    Edge features: [x_u ; x_v ; x_u * x_v]  → 1D causal convolutions.
    """

    def __init__(self, in_dim=NODE_FEAT, hidden=HIDDEN, num_classes=NUM_CLASSES):
        super().__init__()
        self.edge_dim = in_dim * 3
        self.proj = nn.Linear(self.edge_dim, hidden)
        self.ln_in = nn.LayerNorm(hidden)
        self.tcn1 = TemporalConvBlock(hidden, kernel_size=3, dilation=1)
        self.tcn2 = TemporalConvBlock(hidden, kernel_size=3, dilation=2)
        self.cls  = nn.Linear(hidden, num_classes)

    def forward(self, node_feat, edge_index):
        src, dst = edge_index[0], edge_index[1]
        x_src = node_feat[src]
        x_dst = node_feat[dst]
        x_edge = torch.cat([x_src, x_dst, x_src * x_dst], dim=1)  # (E, 3d)

        h = self.proj(x_edge)             # (E, hidden)
        h = self.ln_in(h)                 # stabilise before conv
        # Conv expects (B, C, L)
        h = h.unsqueeze(0).transpose(1, 2)  # (1, hidden, E)
        h = self.tcn1(h)
        h = self.tcn2(h)
        h = h.transpose(1, 2).squeeze(0)   # (E, hidden)
        logits = self.cls(h)               # (E, 2)
        return logits


def train_tcn_global(graphs, epochs=100, lr=1e-3):
    model = EdgeTCN()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, epochs + 1):
        total_loss, n = 0.0, 0
        for g in graphs:
            nf, ei, ey = graph_to_tensors(g)
            if ey.size(0) == 0:
                continue
            logits = model(nf, ei)
            loss = F.cross_entropy(logits, ey)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total_loss += loss.item()
            n += 1
        if ep % 20 == 0 or ep == 1:
            print(f"    TCN  ep {ep:>3d}/{epochs}  loss={total_loss/max(n,1):.4f}")
    return model


def eval_tcn(model, graphs):
    model.eval()
    accs, f1s = [], []
    for g in graphs:
        nf, ei, ey = graph_to_tensors(g)
        if ey.size(0) == 0:
            continue
        with torch.no_grad():
            logits = model(nf, ei)
            accs.append(edge_accuracy(logits, ey))
            f1s.append(edge_f1(logits, ey))
    return accs, f1s


# =====================================================================
#  GraphCoder evaluation  (from Step 6/7 artefacts)
# =====================================================================

def eval_graphcoder(graphs, code_indices, ondevice_ckpt):
    """Evaluate reconstructed TinyGNN (GraphCoder, no LoRA) on all graphs."""
    codebook = ondevice_ckpt["codebook"]
    reconstructor = WeightReconstructor(codebook, node_feat_dim=NODE_FEAT)
    hidden = codebook.shape[1]
    edge_feat_dim = int(graphs[0]["edge_attr"].shape[1])
    gnn = TinyGNN(node_feat_dim=NODE_FEAT, hidden=hidden, num_classes=NUM_CLASSES,
                  edge_feat_dim=edge_feat_dim)
    if "edge_mlp.0.weight" in ondevice_ckpt["hw_model_state_dict"]:
        gnn.load_edge_mlp_state(ondevice_ckpt["hw_model_state_dict"])

    accs, f1s = [], []
    for i, g in enumerate(graphs):
        nf, ei, ey = graph_to_tensors(g)
        edge_attr = torch.tensor(g["edge_attr"], dtype=torch.float32)
        if ey.size(0) == 0:
            continue
        z = torch.tensor(code_indices[i], dtype=torch.long)
        weights = reconstructor(z)
        gnn.load_weights(weights)
        gnn.eval()
        with torch.no_grad():
            logits, _ = gnn(nf, ei, edge_attr)
            accs.append(edge_accuracy(logits, ey))
            f1s.append(edge_f1(logits, ey))
    return accs, f1s


# =====================================================================
#  Comparison Plot
# =====================================================================

def plot_comparison(results: dict, save_path: str):
    """Bar chart comparing all baselines + GraphCoder."""
    methods = list(results.keys())
    accs = [results[m]["mean_acc"] for m in methods]
    f1s  = [results[m]["mean_f1"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, accs, width, label="Edge Accuracy",
                   color="#2563eb", alpha=0.85)
    bars2 = ax.bar(x + width/2, f1s, width, label="Macro-F1",
                   color="#059669", alpha=0.85)

    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Baseline Comparison — Edge-Level Binary Classification",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Comparison plot -> {save_path}")


# =====================================================================
#  Main
# =====================================================================

if __name__ == "__main__":
    import pickle

    t0 = time.time()
    print("=" * 60)
    print("  Step 8 — Rigorous Baselines")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, "temporal_graphs.pkl"), "rb") as f:
        temporal_graphs = pickle.load(f)
    labels = np.load(os.path.join(OUTPUT_DIR, "fingerprint_labels.npy"))
    code_indices = np.load(os.path.join(OUTPUT_DIR, "code_indices.npy"))

    ckpt = torch.load(os.path.join(OUTPUT_DIR, "ondevice_gnn.pt"),
                      map_location="cpu", weights_only=False)

    non_empty = [g for g in temporal_graphs if g["num_edges"] > 0]
    print(f"[INFO] {len(temporal_graphs)} graphs total, "
          f"{len(non_empty)} non-empty")

    results = {}

    # ── 1. Oracle GNN (upper bound) ─────────────────────────────────
    print(f"\n{'─'*60}")
    print("[Baseline 1] Oracle GNN — per-graph fine-tuned (upper bound)")
    print(f"{'─'*60}")

    oracle_accs, oracle_f1s = [], []
    n_oracle = len(non_empty)
    for idx, g in enumerate(temporal_graphs):
        if g["num_edges"] == 0:
            continue
        _, acc, f1 = train_oracle_on_graph(g, epochs=200, lr=1e-2)
        oracle_accs.append(acc)
        oracle_f1s.append(f1)
        if (idx + 1) % 50 == 0:
            print(f"    ... {idx+1}/{len(temporal_graphs)} graphs done")

    results["Oracle GNN\n(per-device)"] = {
        "mean_acc": np.mean(oracle_accs),
        "std_acc":  np.std(oracle_accs),
        "mean_f1":  np.mean(oracle_f1s),
        "std_f1":   np.std(oracle_f1s),
        "params":   sum(p.numel() for p in OracleGNN().parameters()),
    }
    print(f"  Oracle:  acc={np.mean(oracle_accs):.4f} ± {np.std(oracle_accs):.4f}  "
          f"f1={np.mean(oracle_f1s):.4f}")

    # ── 2. Global GNN ───────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("[Baseline 2] Global GNN — single model, all devices")
    print(f"{'─'*60}")

    global_model = train_global_gnn(temporal_graphs, epochs=100, lr=1e-3)
    global_accs, global_f1s = eval_global_gnn(global_model, temporal_graphs)

    results["Global GNN\n(single fixed)"] = {
        "mean_acc": np.mean(global_accs),
        "std_acc":  np.std(global_accs),
        "mean_f1":  np.mean(global_f1s),
        "std_f1":   np.std(global_f1s),
        "params":   sum(p.numel() for p in global_model.parameters()),
    }
    print(f"  Global:  acc={np.mean(global_accs):.4f} ± {np.std(global_accs):.4f}  "
          f"f1={np.mean(global_f1s):.4f}")

    # ── 3. LSTM Baseline ────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("[Baseline 3] LSTM — non-graph temporal baseline")
    print(f"{'─'*60}")

    lstm_model = train_lstm_global(temporal_graphs, epochs=100, lr=1e-3)
    lstm_accs, lstm_f1s = eval_lstm(lstm_model, temporal_graphs)

    results["LSTM\n(non-graph)"] = {
        "mean_acc": np.mean(lstm_accs),
        "std_acc":  np.std(lstm_accs),
        "mean_f1":  np.mean(lstm_f1s),
        "std_f1":   np.std(lstm_f1s),
        "params":   sum(p.numel() for p in lstm_model.parameters()),
    }
    print(f"  LSTM:    acc={np.mean(lstm_accs):.4f} ± {np.std(lstm_accs):.4f}  "
          f"f1={np.mean(lstm_f1s):.4f}")

    # ── 4. TCN Baseline ─────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("[Baseline 4] TCN — temporal convolution baseline")
    print(f"{'─'*60}")

    tcn_model = train_tcn_global(temporal_graphs, epochs=100, lr=1e-3)
    tcn_accs, tcn_f1s = eval_tcn(tcn_model, temporal_graphs)

    results["TCN\n(non-graph)"] = {
        "mean_acc": np.mean(tcn_accs),
        "std_acc":  np.std(tcn_accs),
        "mean_f1":  np.mean(tcn_f1s),
        "std_f1":   np.std(tcn_f1s),
        "params":   sum(p.numel() for p in tcn_model.parameters()),
    }
    print(f"  TCN:     acc={np.mean(tcn_accs):.4f} ± {np.std(tcn_accs):.4f}  "
          f"f1={np.mean(tcn_f1s):.4f}")

    # ── 5. GraphCoder (ours) ─────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("[Ours] GraphCoder — VQ-reconstructed TinyGNN")
    print(f"{'─'*60}")

    gc_accs, gc_f1s = eval_graphcoder(temporal_graphs, code_indices, ckpt)

    results["GraphCoder\n(ours)"] = {
        "mean_acc": np.mean(gc_accs),
        "std_acc":  np.std(gc_accs),
        "mean_f1":  np.mean(gc_f1s),
        "std_f1":   np.std(gc_f1s),
        "params":   TinyGNN().count_params(),
    }
    print(f"  GC:      acc={np.mean(gc_accs):.4f} ± {np.std(gc_accs):.4f}  "
          f"f1={np.mean(gc_f1s):.4f}")

    # ── 6. Summary table ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  BASELINE COMPARISON  (edge-level binary classification)")
    print(f"{'='*60}")
    print(f"{'Method':<22s}  {'Acc':>8s}  {'± std':>7s}  "
          f"{'F1':>7s}  {'± std':>7s}  {'Params':>8s}")
    print("-" * 65)
    for name, m in results.items():
        label = name.replace("\n", " ")
        print(f"{label:<22s}  {m['mean_acc']:>8.4f}  {m['std_acc']:>7.4f}  "
              f"{m['mean_f1']:>7.4f}  {m['std_f1']:>7.4f}  "
              f"{m['params']:>8,}")

    # ── 7. Save artefacts ────────────────────────────────────────────
    # JSON-safe results (no newlines in keys)
    json_results = {}
    for k, v in results.items():
        json_results[k.replace("\n", " ")] = v

    json_path = os.path.join(OUTPUT_DIR, "baselines_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n[INFO] Results JSON -> {json_path}")

    plot_path = os.path.join(OUTPUT_DIR, "baselines_comparison.png")
    plot_comparison(results, plot_path)

    summary_path = os.path.join(OUTPUT_DIR, "baselines_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("  Step 8 — Rigorous Baselines\n")
        f.write("=" * 65 + "\n\n")
        f.write("Task: Edge-level binary classification (normal vs attack)\n")
        f.write(f"Graphs: {len(temporal_graphs)} total, "
                f"{len(non_empty)} non-empty\n\n")

        f.write("Methods\n" + "-" * 40 + "\n")
        f.write("  1. Oracle GNN   — per-graph fully fine-tuned (upper bound)\n")
        f.write("  2. Global GNN   — single fixed model, all devices\n")
        f.write("  3. LSTM          — non-graph, node-pair temporal\n")
        f.write("  4. TCN           — non-graph, temporal convolution\n")
        f.write("  5. GraphCoder   — VQ-reconstructed TinyGNN (ours)\n\n")

        f.write(f"{'Method':<22s}  {'Acc':>8s}  {'std':>7s}  "
                f"{'F1':>7s}  {'std':>7s}  {'Params':>8s}\n")
        f.write("-" * 65 + "\n")
        for name, m in results.items():
            label = name.replace("\n", " ")
            f.write(f"{label:<22s}  {m['mean_acc']:>8.4f}  {m['std_acc']:>7.4f}  "
                    f"{m['mean_f1']:>7.4f}  {m['std_f1']:>7.4f}  "
                    f"{m['params']:>8,}\n")

        f.write("\n\nArchitecture Details\n" + "-" * 40 + "\n")
        f.write(f"  All models: hidden={HIDDEN}, num_classes={NUM_CLASSES}\n")
        f.write(f"  Oracle GNN : 3 GCN layers + LN, trained per graph (200 ep)\n")
        f.write(f"  Global GNN : 3 GCN layers + LN, trained on all (100 ep)\n")
        f.write(f"  LSTM       : 1-layer LSTM on edge features [xu;xv;xu*xv]\n")
        f.write(f"  TCN        : 2 causal conv blocks (k=3, d=1,2) on edge feats\n")
        f.write(f"  GraphCoder : 3 GCN layers, weights from R(z,C)\n")

    print(f"[INFO] Summary     -> {summary_path}")
    print(f"\n[INFO] Step 8 complete in {time.time()-t0:.1f}s")
