"""
Step 6 – On-Device Weight Reconstruction & INT8 Quantisation
=============================================================
Simulates an IoT device receiving a code-index update  z ∈ Z^M  and
reconstructing a tiny GNN from the local codebook  C ∈ R^{K×D}.

Weight Reconstruction  Ŵ = R(z, C)
-----------------------------------
  1.  Look up M code vectors from C:   E = C[z]  →  (M, D) = (8, 32)
  2.  Pair up codes for each GNN layer (rank-1 outer-product factorisation):
        W_l = e_{2l}  ⊗  e_{2l+1}^T      (32 × 32  or  32 × 15)
  3.  The resulting weight matrices directly parametrise the GNN layers.

Tiny GNN Architecture  (3 layers, hidden=32)
---------------------------------------------
  Layer 1  GCN  15 → 32    W₁ = e₀ ⊗ e₁[:15]ᵀ       codes 0-1
  Layer 2  GCN  32 → 32    W₂ = e₂ ⊗ e₃ᵀ             codes 2-3
  Layer 3  GCN  32 → 32    W₃ = e₄ ⊗ e₅ᵀ             codes 4-5
  Classifier   32 → 2      W_cls = stack(e₆, e₇)      codes 6-7
  Edge scoring: h_edge = h_src ⊙ h_dst → W_cls @ h_edge

Post-Training Quantisation
--------------------------
  All reconstructed float32 weights are quantised to INT8:
      scale = max(|W|) / 127
      W_q   = clamp(round(W / scale), -128, 127)

Outputs  (in ./output/)
-----------------------
  ondevice_gnn.pt                 – reconstructed + quantized model
  ondevice_summary.txt            – architecture summary + size audit
  ondevice_weight_heatmaps.png   – reconstructed weight visualisation
"""

import os, sys, time, json

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

from vq_codebook import VQCodebook

DEVICE = "cpu"   # IoT device — always CPU


# =====================================================================
#  1.  Weight Reconstruction  Ŵ = R(z, C)
# =====================================================================

class WeightReconstructor:
    """
    Decoding function  R(z, C)  that lives on every IoT device.

    Given M code indices  z = (z_0, …, z_{M-1})  and codebook  C ∈ R^{K×D}:
      1. Fetch  e_m = C[z_m]  for each position  m.
      2. Build layer weights via rank-2 outer-product factorisation
         (two outer products summed per layer for higher expressiveness).

    Code-to-layer assignment  (M=16, D=48):
      codes  0-3  →  GCN layer 1  W₁ ∈ R^{D×nf}  rank-2: e₀⊗e₁[:nf]ᵀ + e₂⊗e₃[:nf]ᵀ
      codes  4-7  →  GCN layer 2  W₂ ∈ R^{D×D}   rank-2: e₄⊗e₅ᵀ + e₆⊗e₇ᵀ
      codes  8-11 →  GCN layer 3  W₃ ∈ R^{D×D}   rank-2: e₈⊗e₉ᵀ + e₁₀⊗e₁₁ᵀ
      codes 12-15 →  Classifier   W_cls ∈ R^{2×D} stack(e₁₂+e₁₄, e₁₃+e₁₅)
    """

    def __init__(self, codebook: torch.Tensor, node_feat_dim: int = 15):
        """
        Parameters
        ----------
        codebook : Tensor (K, D)
            Local copy of the VQ codebook.
        node_feat_dim : int
            Dimensionality of input node features.
        """
        self.codebook = codebook        # (K, D)
        self.D = codebook.shape[1]      # 48
        self.node_feat_dim = node_feat_dim

    def __call__(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        R(z, C) → dict of named weight tensors.

        Parameters
        ----------
        z : LongTensor (M,)   —   code indices for one time-window.

        Returns
        -------
        weights : dict mapping layer name → Tensor
            W1     : (D, nf)   GCN layer 1
            W2     : (D, D)    GCN layer 2
            W3     : (D, D)    GCN layer 3
            W_cls  : (2, D)    classifier
        """
        e = torch.tanh(self.codebook[z])  # bound code amplitudes for stable rank-2 reconstruction
        scale = max(self.D ** 0.5, 1.0)
        nf = self.node_feat_dim

        # ── Layer 1: rank-2 outer product, D × nf ───────────────────
        W1 = (e[0].unsqueeze(1) * e[1][:nf].unsqueeze(0) +
              e[2].unsqueeze(1) * e[3][:nf].unsqueeze(0)) / scale       # (D, nf)

        # ── Layer 2: rank-2 outer product, D × D ────────────────────
        W2 = (e[4].unsqueeze(1) * e[5].unsqueeze(0) +
              e[6].unsqueeze(1) * e[7].unsqueeze(0)) / scale             # (D, D)

        # ── Layer 3: rank-2 outer product, D × D ────────────────────
        W3 = (e[8].unsqueeze(1) * e[9].unsqueeze(0) +
              e[10].unsqueeze(1) * e[11].unsqueeze(0)) / scale           # (D, D)

        # ── Classifier: sum code-pair per row, 2 × D ────────────────
        W_cls = torch.stack([e[12] + e[14], e[13] + e[15]], dim=0) / 2.0  # (2, D)

        return {"W1": W1, "W2": W2, "W3": W3, "W_cls": W_cls}


# =====================================================================
#  2.  INT8 Post-Training Quantisation
# =====================================================================

def quantise_int8(w_fp32: torch.Tensor):
    """
    Symmetric per-tensor quantisation to 8-bit integers.

    Returns
    -------
    w_q     : Tensor  dtype=int8
    scale   : float   (w_fp32 ≈ w_q * scale)
    """
    amax = w_fp32.abs().max().item()
    if amax < 1e-12:
        return torch.zeros_like(w_fp32, dtype=torch.int8), 0.0
    scale = amax / 127.0
    w_q = (w_fp32 / scale).round().clamp(-128, 127).to(torch.int8)
    return w_q, scale


def dequantise(w_q: torch.Tensor, scale: float) -> torch.Tensor:
    """INT8 → float32 approximate recovery."""
    return w_q.float() * scale


def quantise_weight_dict(weights: dict[str, torch.Tensor]):
    """
    Quantise all weight tensors in a dict.

    Returns
    -------
    q_weights : dict[str, Tensor]   (int8 tensors)
    scales    : dict[str, float]
    """
    q_weights = {}
    scales = {}
    for name, W in weights.items():
        w_q, s = quantise_int8(W)
        q_weights[name] = w_q
        scales[name] = s
    return q_weights, scales


# =====================================================================
#  3.  Tiny GNN  —  On-Device GNN with Reconstructed Weights
# =====================================================================

class TinyGNN(nn.Module):
    """
    Minimal GCN for edge-level binary classification on IoT devices.

    Architecture (3 GCN layers + edge classifier, hidden=32):
    ┌──────────────────────────────────────────────┐
    │  x  (N, 15)  node features                   │
    │     ↓                                         │
    │  GCN Layer 1  x·Ã·W₁ᵀ   (N, 15)→(N, 32)    │
    │     ReLU                                      │
    │     ↓                                         │
    │  GCN Layer 2  h·Ã·W₂ᵀ   (N, 32)→(N, 32)    │
    │     ReLU                                      │
    │     ↓                                         │
    │  GCN Layer 3  h·Ã·W₃ᵀ   (N, 32)→(N, 32)    │
    │     ReLU                                      │
    │     ↓                                         │
    │  Edge scoring:                                │
    │     h_edge = h[src] ⊙ h[dst]   (E, 32)      │
    │     logits = W_cls @ h_edgeᵀ   (E, 2)       │
    └──────────────────────────────────────────────┘

    Where Ã = D^{-½} (A + I) D^{-½}  is the GCN-normalised adjacency.
    """

    def __init__(self, node_feat_dim: int = 15, hidden: int = 48,
                 num_classes: int = 2, edge_feat_dim: int = 0):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden = hidden
        self.num_classes = num_classes
        self.num_layers = 3
        self.edge_feat_dim = edge_feat_dim

        # Placeholder parameters (overwritten by load_weights)
        self.W1 = nn.Parameter(torch.empty(hidden, node_feat_dim))
        self.W2 = nn.Parameter(torch.empty(hidden, hidden))
        self.W3 = nn.Parameter(torch.empty(hidden, hidden))
        self.W_cls = nn.Parameter(torch.empty(num_classes, hidden))
        self.edge_mlp = (nn.Sequential(
            nn.Linear(edge_feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        ) if edge_feat_dim > 0 else None)

        nn.init.kaiming_uniform_(self.W1)
        nn.init.kaiming_uniform_(self.W2)
        nn.init.kaiming_uniform_(self.W3)
        nn.init.kaiming_uniform_(self.W_cls)

    def load_weights(self, weights: dict[str, torch.Tensor]):
        """Load weights produced by R(z, C)."""
        with torch.no_grad():
            self.W1.copy_(weights["W1"])
            self.W2.copy_(weights["W2"])
            self.W3.copy_(weights["W3"])
            self.W_cls.copy_(weights["W_cls"])

    def load_quantised(self, q_weights: dict[str, torch.Tensor],
                       scales: dict[str, float]):
        """Load INT8 quantised weights (dequantised to float32 for inference)."""
        with torch.no_grad():
            self.W1.copy_(dequantise(q_weights["W1"], scales["W1"]))
            self.W2.copy_(dequantise(q_weights["W2"], scales["W2"]))
            self.W3.copy_(dequantise(q_weights["W3"], scales["W3"]))
            self.W_cls.copy_(dequantise(q_weights["W_cls"], scales["W_cls"]))

    def load_edge_mlp_state(self, state_dict: dict[str, torch.Tensor]):
        """Load the shared edge-feature residual head from the training checkpoint."""
        if self.edge_mlp is None:
            return
        with torch.no_grad():
            self.edge_mlp[0].weight.copy_(state_dict["edge_mlp.0.weight"])
            self.edge_mlp[0].bias.copy_(state_dict["edge_mlp.0.bias"])
            self.edge_mlp[2].weight.copy_(state_dict["edge_mlp.2.weight"])
            self.edge_mlp[2].bias.copy_(state_dict["edge_mlp.2.bias"])

    @staticmethod
    def _gcn_norm(edge_index: torch.Tensor, num_nodes: int):
        """
        Compute GCN symmetric normalisation  D^{-½}(A+I)D^{-½}
        as sparse COO tensor.
        """
        src, dst = edge_index[0], edge_index[1]

        # Add self-loops
        self_loops = torch.arange(num_nodes, device=edge_index.device)
        src = torch.cat([src, self_loops])
        dst = torch.cat([dst, self_loops])

        # Degree (with self-loops)
        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))

        # D^{-1/2}
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

        # Edge weights: D^{-½}_i · 1 · D^{-½}_j
        weights = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

        idx = torch.stack([dst, src])   # (2, nnz)  — target, source
        A_norm = torch.sparse_coo_tensor(idx, weights,
                                         size=(num_nodes, num_nodes))
        return A_norm.coalesce()

    def forward(self, node_feat: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor | None = None):
        """
        Parameters
        ----------
        node_feat  : (N, 15)
        edge_index : (2, E)   LongTensor
        edge_attr  : (E, *)   unused by GCN (available for future extensions)

        Returns
        -------
        logits : (E, 2)   edge classification logits
        h      : (N, 32)  final node embeddings
        """
        N = node_feat.size(0)
        A_norm = self._gcn_norm(edge_index, N)

        # ── GCN Layer 1 ─────────────────────────────────────────────
        h = torch.sparse.mm(A_norm, node_feat)  # (N, 15)
        h = h @ self.W1.t()                      # (N, 32)
        h = F.relu(h)

        # ── GCN Layer 2 ─────────────────────────────────────────────
        h = torch.sparse.mm(A_norm, h)           # (N, 32)
        h = h @ self.W2.t()                      # (N, 32)
        h = F.relu(h)

        # ── GCN Layer 3 ─────────────────────────────────────────────
        h = torch.sparse.mm(A_norm, h)           # (N, 32)
        h = h @ self.W3.t()                      # (N, 32)
        h = F.relu(h)

        # ── Edge scoring ─────────────────────────────────────────────
        src, dst = edge_index[0], edge_index[1]
        h_edge = h[src] * h[dst]                  # (E, hidden)  Hadamard
        logits = h_edge @ self.W_cls.t()          # (E, 2)
        if edge_attr is not None and self.edge_mlp is not None:
            logits = logits + self.edge_mlp(edge_attr)

        return logits, h

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def size_bytes(self, dtype_bytes: int = 4) -> int:
        return self.count_params() * dtype_bytes

    def architecture_summary(self) -> str:
        lines = [
            f"Tiny GNN  —  {self.num_layers} GCN layers, hidden={self.hidden}",
            f"  Layer 1  GCN   {self.node_feat_dim:>3d} → {self.hidden}   "
            f"W1 {tuple(self.W1.shape)}",
            f"  Layer 2  GCN   {self.hidden:>3d} → {self.hidden}   "
            f"W2 {tuple(self.W2.shape)}",
            f"  Layer 3  GCN   {self.hidden:>3d} → {self.hidden}   "
            f"W3 {tuple(self.W3.shape)}",
            f"  Classifier     {self.hidden:>3d} → {self.num_classes}    "
            f"W_cls {tuple(self.W_cls.shape)}",
            (f"  Edge residual   {self.edge_feat_dim:>3d} → {self.hidden} → {self.num_classes}"
             if self.edge_mlp is not None else "  Edge residual   disabled"),
            f"  Edge scoring: graph logits + shared edge-feature residual",
            f"  Total parameters : {self.count_params():,}",
            f"  Float32 size     : {self.size_bytes(4):,} bytes "
            f"({self.size_bytes(4)/1024:.2f} KB)",
            f"  INT8 size        : {self.size_bytes(1):,} bytes "
            f"({self.size_bytes(1)/1024:.2f} KB)",
        ]
        return "\n".join(lines)


# =====================================================================
#  4.  Evaluation helpers
# =====================================================================

def evaluate_on_graph(model: TinyGNN, graph: dict):
    """Run the tiny GNN on a single temporal graph snapshot."""
    model.eval()
    with torch.no_grad():
        node_feat  = torch.tensor(graph["node_feat"], dtype=torch.float32)
        edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
        edge_attr  = torch.tensor(graph["edge_attr"], dtype=torch.float32)
        edge_y     = torch.tensor(graph["edge_y"], dtype=torch.long)

        logits, h = model(node_feat, edge_index, edge_attr)
        preds = logits.argmax(dim=1)
        acc = (preds == edge_y).float().mean().item()
    return acc, preds.numpy(), edge_y.numpy()


def plot_weight_heatmaps(weights_fp32: dict, weights_q: dict,
                         scales: dict, save_path: str):
    """Visualise reconstructed weights (float32 vs INT8)."""
    names = ["W1", "W2", "W3", "W_cls"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))

    for i, name in enumerate(names):
        W_fp  = weights_fp32[name].numpy()
        W_int = weights_q[name].numpy().astype(float)

        ax = axes[0, i]
        im = ax.imshow(W_fp, aspect="auto", cmap="RdBu_r")
        ax.set_title(f"{name} (float32)", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1, i]
        im = ax.imshow(W_int, aspect="auto", cmap="RdBu_r")
        ax.set_title(f"{name} (INT8, s={scales[name]:.4f})", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Reconstructed GNN Weights — R(z, C) → INT8 Quantisation",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Weight heatmaps -> {save_path}")


# =====================================================================
#  Main — Full on-device simulation
# =====================================================================

if __name__ == "__main__":
    import pickle

    t0 = time.time()
    print("=" * 60)
    print("  Step 6 — On-Device Weight Reconstruction + INT8")
    print("=" * 60)

    # ── Load artefacts from previous steps ───────────────────────────
    ckpt = torch.load(os.path.join(OUTPUT_DIR, "hw_aware_model.pt"),
                      map_location="cpu", weights_only=False)
    codebook = ckpt["model_state_dict"]["hypernet.vq.embed"]   # (K, D)
    K, D = codebook.shape
    print(f"\n[INFO] Codebook loaded  :  C ∈ R^({K}×{D})")

    code_indices = np.load(os.path.join(OUTPUT_DIR, "code_indices.npy"))
    T, M = code_indices.shape
    print(f"[INFO] Code indices     :  {T} windows × M={M}")

    with open(os.path.join(OUTPUT_DIR, "temporal_graphs.pkl"), "rb") as f:
        temporal_graphs = pickle.load(f)
    print(f"[INFO] Temporal graphs  :  {len(temporal_graphs)} snapshots")
    edge_feat_dim = int(temporal_graphs[0]["edge_attr"].shape[1])

    labels = np.load(os.path.join(OUTPUT_DIR, "fingerprint_labels.npy"))

    # ── 1. Weight Reconstruction  R(z, C) ────────────────────────────
    print(f"\n{'─'*60}")
    print("[STEP 6a] Weight Reconstruction  Ŵ = R(z, C)")
    print(f"{'─'*60}")

    reconstructor = WeightReconstructor(codebook, node_feat_dim=15)

    # Pick a representative sample (first attack window)
    sample_idx = int(np.where(labels == 1)[0][0])
    z_sample = torch.tensor(code_indices[sample_idx], dtype=torch.long)
    print(f"  Sample window : {sample_idx}  (label={'attack' if labels[sample_idx] else 'normal'})")
    print(f"  Code indices z: {z_sample.tolist()}")

    weights_fp32 = reconstructor(z_sample)

    print(f"\n  Reconstructed weight shapes:")
    total_params = 0
    for name, W in weights_fp32.items():
        print(f"    {name:>6s} : {str(tuple(W.shape)):>12s}   "
              f"numel={W.numel():>5d}   "
              f"range=[{W.min():.4f}, {W.max():.4f}]")
        total_params += W.numel()

    print(f"\n  Total reconstructed parameters : {total_params:,}")
    print(f"  Float32 footprint             : {total_params * 4:,} bytes "
          f"({total_params * 4 / 1024:.2f} KB)")
    print(f"  Transmission cost             : {M} indices × 6 bits = "
          f"{M * 6} bits ({M * 6 // 8} bytes)")

    # ── 2. Build Tiny GNN & load weights ─────────────────────────────
    print(f"\n{'─'*60}")
    print("[STEP 6b] Tiny GNN Architecture")
    print(f"{'─'*60}")

    gnn = TinyGNN(node_feat_dim=15, hidden=D, num_classes=2, edge_feat_dim=edge_feat_dim)
    gnn.load_weights(weights_fp32)
    gnn.load_edge_mlp_state(ckpt["model_state_dict"])
    print(f"\n{gnn.architecture_summary()}")

    # Verify constraints
    assert 2 <= gnn.num_layers <= 4, f"Layer count {gnn.num_layers} not in [2,4]"
    assert 32 <= gnn.hidden <= 64, f"Hidden dim {gnn.hidden} not in [32,64]"
    print(f"\n  ✓ Architecture constraints satisfied: "
          f"{gnn.num_layers} layers, {gnn.hidden} hidden units")

    # ── 3. INT8 Post-Training Quantisation ───────────────────────────
    print(f"\n{'─'*60}")
    print("[STEP 6c] INT8 Post-Training Quantisation")
    print(f"{'─'*60}")

    q_weights, scales = quantise_weight_dict(weights_fp32)

    print(f"\n  Quantisation results (per-tensor symmetric):")
    total_int8_bytes = 0
    for name in weights_fp32:
        W_fp = weights_fp32[name]
        W_q  = q_weights[name]
        sc   = scales[name]
        # Quantisation error
        W_deq = dequantise(W_q, sc)
        mse = (W_fp - W_deq).pow(2).mean().item()
        max_err = (W_fp - W_deq).abs().max().item()
        nbytes = W_q.numel()
        total_int8_bytes += nbytes
        print(f"    {name:>6s}  scale={sc:.6f}  "
              f"MSE={mse:.2e}  MaxErr={max_err:.4f}  "
              f"INT8={nbytes:>5d} bytes")

    fp32_bytes = total_params * 4
    print(f"\n  Float32 total : {fp32_bytes:>6,} bytes ({fp32_bytes/1024:.2f} KB)")
    print(f"  INT8 total    : {total_int8_bytes:>6,} bytes ({total_int8_bytes/1024:.2f} KB)")
    print(f"  Compression   : {fp32_bytes / total_int8_bytes:.1f}× reduction")

    # ── 4. Inference with quantised weights ──────────────────────────
    print(f"\n{'─'*60}")
    print("[STEP 6d] On-Device Inference (INT8 weights)")
    print(f"{'─'*60}")

    gnn_q = TinyGNN(node_feat_dim=15, hidden=D, num_classes=2, edge_feat_dim=edge_feat_dim)
    gnn_q.load_quantised(q_weights, scales)
    gnn_q.load_edge_mlp_state(ckpt["model_state_dict"])

    # Evaluate on all graphs
    results_fp32 = []
    results_int8 = []
    for i, g in enumerate(temporal_graphs):
        if g["num_edges"] == 0:
            continue
        # Float32 inference
        gnn.load_weights(reconstructor(torch.tensor(code_indices[i], dtype=torch.long)))
        acc_fp, _, _ = evaluate_on_graph(gnn, g)
        results_fp32.append(acc_fp)

        # INT8 inference
        w = reconstructor(torch.tensor(code_indices[i], dtype=torch.long))
        qw, sc = quantise_weight_dict(w)
        gnn_q.load_quantised(qw, sc)
        acc_q, _, _ = evaluate_on_graph(gnn_q, g)
        results_int8.append(acc_q)

    mean_fp32 = np.mean(results_fp32)
    mean_int8 = np.mean(results_int8)
    agree = np.mean([abs(a-b) < 0.01 for a, b in zip(results_fp32, results_int8)])

    print(f"\n  Evaluated on {len(results_fp32)} graphs (non-empty edges):")
    print(f"    Float32 mean edge accuracy : {mean_fp32:.4f}")
    print(f"    INT8    mean edge accuracy : {mean_int8:.4f}")
    print(f"    FP32↔INT8 agreement rate   : {agree:.4f}")
    print(f"    Accuracy drop from quant   : {mean_fp32 - mean_int8:+.4f}")

    # ── 5. Save artefacts ────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("[STEP 6e] Saving Artefacts")
    print(f"{'─'*60}")

    # Model checkpoint
    save_dict = {
        "codebook": codebook,
        "hw_model_state_dict": {
            k: v.clone() for k, v in ckpt["model_state_dict"].items()
            if k.startswith("edge_mlp.")
        },
        "sample_z": z_sample,
        "weights_fp32": {k: v.clone() for k, v in weights_fp32.items()},
        "q_weights": {k: v.clone() for k, v in q_weights.items()},
        "scales": scales,
        "architecture": {
            "num_layers": gnn.num_layers,
            "hidden": gnn.hidden,
            "node_feat_dim": gnn.node_feat_dim,
            "num_classes": gnn.num_classes,
            "factorisation": "rank-1 outer product per layer pair",
        },
        "metrics": {
            "total_params": total_params,
            "fp32_bytes": fp32_bytes,
            "int8_bytes": total_int8_bytes,
            "compression_ratio": fp32_bytes / total_int8_bytes,
            "mean_acc_fp32": mean_fp32,
            "mean_acc_int8": mean_int8,
            "agreement_rate": agree,
        },
    }
    ckpt_path = os.path.join(OUTPUT_DIR, "ondevice_gnn.pt")
    torch.save(save_dict, ckpt_path)
    print(f"  Model checkpoint  -> {ckpt_path}")

    # Weight heatmaps
    plot_path = os.path.join(OUTPUT_DIR, "ondevice_weight_heatmaps.png")
    plot_weight_heatmaps(weights_fp32, q_weights, scales, plot_path)

    # Summary report
    summary_path = os.path.join(OUTPUT_DIR, "ondevice_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  Step 6 — On-Device Weight Reconstruction + INT8\n")
        f.write("=" * 60 + "\n\n")

        f.write("Weight Reconstruction  R(z, C)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Codebook C      : R^({K} x {D})\n")
        f.write(f"  Code positions M: {M}\n")
        f.write(f"  Code dim D      : {D}\n")
        bits_per_idx = int(np.ceil(np.log2(K)))
        total_bits = M * bits_per_idx
        total_bytes = int(np.ceil(total_bits / 8))
        f.write(f"  Factorisation   : rank-2 outer product per layer (4 codes/layer)\n")
        f.write(f"  Transmission    : {M} indices x {bits_per_idx} bits = "
                f"{total_bits} bits = {total_bytes} bytes\n\n")

        f.write("Code-to-Layer Assignment\n")
        f.write("-" * 40 + "\n")
        f.write(f"  codes  0-3  ->  GCN Layer 1  W1 ({D} x {15}) rank-2\n")
        f.write(f"  codes  4-7  ->  GCN Layer 2  W2 ({D} x {D}) rank-2\n")
        f.write(f"  codes  8-11 ->  GCN Layer 3  W3 ({D} x {D}) rank-2\n")
        f.write(f"  codes 12-15 ->  Classifier   W_cls (2 x {D})\n\n")

        f.write("Tiny GNN Architecture\n")
        f.write("-" * 40 + "\n")
        f.write(gnn.architecture_summary() + "\n\n")

        f.write("INT8 Quantisation\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Method        : Per-tensor symmetric\n")
        f.write(f"  Float32 size  : {fp32_bytes:,} bytes ({fp32_bytes/1024:.2f} KB)\n")
        f.write(f"  INT8 size     : {total_int8_bytes:,} bytes ({total_int8_bytes/1024:.2f} KB)\n")
        f.write(f"  Compression   : {fp32_bytes/total_int8_bytes:.1f}x\n")
        for name in weights_fp32:
            f.write(f"  {name:>6s}  scale={scales[name]:.6f}\n")
        f.write("\n")

        f.write("Inference Results\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Graphs evaluated     : {len(results_fp32)}\n")
        f.write(f"  FP32 mean edge acc   : {mean_fp32:.4f}\n")
        f.write(f"  INT8 mean edge acc   : {mean_int8:.4f}\n")
        f.write(f"  FP32<->INT8 agreement: {agree:.4f}\n")
        f.write(f"  Accuracy drop        : {mean_fp32 - mean_int8:+.4f}\n")

    print(f"  Summary report    -> {summary_path}")
    print(f"\n[INFO] Step 6 complete in {time.time()-t0:.1f}s")
