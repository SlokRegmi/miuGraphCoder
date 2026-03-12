"""
Step 7 – Low-Rank Adapters (LoRA) for On-Device Drift Adaptation
=================================================================
The edge device adapts to local distribution shifts without contacting
the server by training lightweight LoRA bypass matrices.

LoRA bypass per layer
---------------------
  For every reconstructed weight  Ŵ ∈ R^{d_out × d_in}:
      A ∈ R^{d_out × r}     (zero-initialised)
      B ∈ R^{d_in  × r}     (random-initialised)

  Adapted forward pass:
      W' = Ŵ + A·Bᵀ          (rank-r additive perturbation)

  Constraint:  r ≤ 4          (hardcoded maximum rank)

Local training protocol
-----------------------
  • Freeze  Ŵ  (reconstructed base weights)
  • Only  A, B  are learnable  →  very few parameters
  • Train on  ≤ 100  locally labelled edge samples
  • Adam optimiser, no weight decay (tiny param budget)

Outputs (in ./output/)
----------------------
  lora_adapted_gnn.pt           – checkpoint (base + LoRA weights)
  lora_adaptation_curve.png     – loss/accuracy over local epochs
  lora_summary.txt              – architecture + adaptation report
"""

import os, sys, time, json, copy

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
    WeightReconstructor, TinyGNN, quantise_int8, dequantise,
    quantise_weight_dict, evaluate_on_graph,
)

DEVICE = "cpu"

# =====================================================================
#  Hyper-parameters
# =====================================================================
LORA_RANK       = 4           # r ≤ 4 (hardcoded max)
MAX_LOCAL_SAMPLES = 100       # max labelled samples for adaptation
LOCAL_EPOCHS    = 50          # local fine-tuning epochs
LOCAL_LR        = 5e-3        # learning rate for A, B
LOCAL_BS        = 16          # mini-batch size

assert LORA_RANK <= 4, f"LoRA rank must be ≤ 4, got {LORA_RANK}"


# =====================================================================
#  LoRA Layer  —  Low-Rank Adapter for a single weight matrix
# =====================================================================

class LoRALayer(nn.Module):
    """
    Low-rank adapter bypass for a frozen weight matrix Ŵ.

        W' = Ŵ + (1/r)·A·Bᵀ

    Parameters
    ----------
    d_out : int
        Output dimension of the weight matrix.
    d_in : int
        Input dimension of the weight matrix.
    rank : int
        LoRA rank  r ≤ 4.
    """

    def __init__(self, d_out: int, d_in: int, rank: int = LORA_RANK):
        super().__init__()
        assert rank <= 4, f"LoRA rank must be ≤ 4, got {rank}"
        self.rank = rank
        self.scaling = 1.0 / rank       # standard LoRA scaling  α/r

        # A: small random init  (drives the adaptation direction)
        self.A = nn.Parameter(torch.empty(d_out, rank))
        nn.init.normal_(self.A, std=0.01)

        # B: zero-initialised so ΔW starts at zero (no perturbation)
        self.B = nn.Parameter(torch.zeros(d_in, rank))

    def delta(self) -> torch.Tensor:
        """Compute the low-rank perturbation  ΔW = (1/r)·A·Bᵀ."""
        return self.scaling * (self.A @ self.B.t())   # (d_out, d_in)

    def extra_repr(self) -> str:
        return (f"d_out={self.A.shape[0]}, d_in={self.B.shape[0]}, "
                f"rank={self.rank}")


# =====================================================================
#  LoRA-Augmented Tiny GNN
# =====================================================================

class LoRATinyGNN(nn.Module):
    """
    Tiny GNN with LoRA bypass on every layer.

    Architecture:
    ┌────────────────────────────────────────────────────────┐
    │  GCN Layer l:  h' = ReLU( Ã·h · (Ŵ_l + A_l·B_lᵀ)ᵀ )│
    │                                                        │
    │  Ŵ_l : frozen base weights from R(z, C)               │
    │  A_l, B_l : trainable LoRA matrices (rank ≤ 4)        │
    └────────────────────────────────────────────────────────┘

    Forward pass for each layer:
        W'_l = Ŵ_l + A_l · B_lᵀ
        h    = ReLU( Ã · h · W'_lᵀ )
    """

    def __init__(self, base_gnn: TinyGNN, rank: int = LORA_RANK):
        super().__init__()
        assert rank <= 4, f"LoRA rank must be ≤ 4, got {rank}"
        self.rank = rank
        self.node_feat_dim = base_gnn.node_feat_dim
        self.hidden = base_gnn.hidden
        self.num_classes = base_gnn.num_classes
        self.num_layers = base_gnn.num_layers

        # ── Frozen base weights (register as buffers, NOT parameters) ─
        self.register_buffer("W1_base", base_gnn.W1.data.clone())
        self.register_buffer("W2_base", base_gnn.W2.data.clone())
        self.register_buffer("W3_base", base_gnn.W3.data.clone())
        self.register_buffer("W_cls_base", base_gnn.W_cls.data.clone())

        # ── Layer norms (stabilise multi-layer GCN activations) ───────
        self.ln1 = nn.LayerNorm(self.hidden)
        self.ln2 = nn.LayerNorm(self.hidden)
        self.ln3 = nn.LayerNorm(self.hidden)

        # ── LoRA adapters (the ONLY trainable parameters) ─────────────
        self.lora1 = LoRALayer(*self.W1_base.shape, rank=rank)
        self.lora2 = LoRALayer(*self.W2_base.shape, rank=rank)
        self.lora3 = LoRALayer(*self.W3_base.shape, rank=rank)
        self.lora_cls = LoRALayer(*self.W_cls_base.shape, rank=rank)

    def _effective_weight(self, W_base: torch.Tensor,
                          lora: LoRALayer) -> torch.Tensor:
        """W' = Ŵ + A·Bᵀ"""
        return W_base + lora.delta()

    def forward(self, node_feat: torch.Tensor,
                edge_index: torch.Tensor):
        """
        Parameters
        ----------
        node_feat  : (N, 15)
        edge_index : (2, E)

        Returns
        -------
        logits : (E, 2)
        h      : (N, hidden)
        """
        N = node_feat.size(0)
        A_norm = TinyGNN._gcn_norm(edge_index, N)

        # ── GCN Layer 1:  W'₁ = Ŵ₁ + A₁·B₁ᵀ ───────────────────────
        W1_eff = self._effective_weight(self.W1_base, self.lora1)
        h = torch.sparse.mm(A_norm, node_feat)
        h = h @ W1_eff.t()
        h = self.ln1(h)
        h = F.relu(h)

        # ── GCN Layer 2:  W'₂ = Ŵ₂ + A₂·B₂ᵀ ───────────────────────
        W2_eff = self._effective_weight(self.W2_base, self.lora2)
        h = torch.sparse.mm(A_norm, h)
        h = h @ W2_eff.t()
        h = self.ln2(h)
        h = F.relu(h)

        # ── GCN Layer 3:  W'₃ = Ŵ₃ + A₃·B₃ᵀ ───────────────────────
        W3_eff = self._effective_weight(self.W3_base, self.lora3)
        h = torch.sparse.mm(A_norm, h)
        h = h @ W3_eff.t()
        h = self.ln3(h)
        h = F.relu(h)

        # ── Edge scoring:  W'_cls = Ŵ_cls + A_cls·B_clsᵀ ───────────
        W_cls_eff = self._effective_weight(self.W_cls_base, self.lora_cls)
        src, dst = edge_index[0], edge_index[1]
        h_edge = h[src] * h[dst]
        logits = h_edge @ W_cls_eff.t()

        return logits, h

    def count_lora_params(self) -> int:
        """Count only the trainable LoRA parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_base_params(self) -> int:
        """Count frozen base-weight elements."""
        return sum(b.numel() for b in [self.W1_base, self.W2_base,
                                        self.W3_base, self.W_cls_base])

    def lora_summary(self) -> str:
        layers = [
            ("W1",    self.W1_base.shape,    self.lora1),
            ("W2",    self.W2_base.shape,    self.lora2),
            ("W3",    self.W3_base.shape,    self.lora3),
            ("W_cls", self.W_cls_base.shape, self.lora_cls),
        ]
        lines = [f"LoRA-Augmented Tiny GNN  (rank r={self.rank})"]
        lines.append(f"{'Layer':>8s}  {'Base shape':>14s}  "
                     f"{'A shape':>12s}  {'B shape':>12s}  {'LoRA params':>12s}")
        lines.append("-" * 68)
        total_lora = 0
        for name, shape, lora in layers:
            a_shape = tuple(lora.A.shape)
            b_shape = tuple(lora.B.shape)
            n = lora.A.numel() + lora.B.numel()
            total_lora += n
            lines.append(f"{name:>8s}  {str(shape):>14s}  "
                         f"{str(a_shape):>12s}  {str(b_shape):>12s}  "
                         f"{n:>12,}")
        lines.append("-" * 68)
        base_n = self.count_base_params()
        lines.append(f"  Base params (frozen) : {base_n:,}")
        lines.append(f"  LoRA params (train)  : {total_lora:,}")
        lines.append(f"  LoRA overhead        : {total_lora / base_n * 100:.2f}%")
        lines.append(f"  LoRA memory (float32): {total_lora * 4:,} bytes "
                     f"({total_lora * 4 / 1024:.2f} KB)")
        return "\n".join(lines)


# =====================================================================
#  Local Drift-Adaptation Training
# =====================================================================

def collect_local_samples(graphs: list[dict], max_samples: int = MAX_LOCAL_SAMPLES):
    """
    Simulate collecting labelled edge samples from recent local traffic.
    Gathers edges from the most recent graphs until we hit max_samples.

    Returns
    -------
    local_graphs : list[dict]
        Subset of graphs whose edges sum to ≤ max_samples.
    total_edges  : int
    """
    local_graphs = []
    total_edges = 0
    # Walk backwards (most recent windows first)
    for g in reversed(graphs):
        if g["num_edges"] == 0:
            continue
        n_edges = g["edge_y"].shape[0]
        if total_edges + n_edges > max_samples:
            # Take a partial slice of this graph to stay within budget
            if total_edges == 0:
                # Need at least some data — take what fits
                local_graphs.append(g)
                total_edges += n_edges
            break
        local_graphs.append(g)
        total_edges += n_edges
        if total_edges >= max_samples:
            break
    local_graphs.reverse()
    return local_graphs, total_edges


def train_lora_local(model: LoRATinyGNN,
                     local_graphs: list[dict],
                     epochs: int = LOCAL_EPOCHS,
                     lr: float = LOCAL_LR):
    """
    On-device local training loop.
    Updates ONLY the LoRA parameters (A, B); base weights Ŵ stay frozen.

    Parameters
    ----------
    model : LoRATinyGNN
    local_graphs : list of graph dicts with edge labels
    epochs : int
    lr : float

    Returns
    -------
    log : list[dict]   per-epoch metrics
    """
    model.train()

    # Only optimise LoRA parameters
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.Adam(lora_params, lr=lr)

    log = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for g in local_graphs:
            node_feat  = torch.tensor(g["node_feat"], dtype=torch.float32)
            edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
            edge_y     = torch.tensor(g["edge_y"], dtype=torch.long)

            logits, _ = model(node_feat, edge_index)
            loss = F.cross_entropy(logits, edge_y)

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimiser.step()

            epoch_loss += loss.item() * edge_y.size(0)
            epoch_correct += (logits.argmax(dim=1) == edge_y).sum().item()
            epoch_total += edge_y.size(0)

        avg_loss = epoch_loss / max(epoch_total, 1)
        accuracy = epoch_correct / max(epoch_total, 1)
        log.append({"epoch": epoch, "loss": avg_loss, "accuracy": accuracy})

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"    Ep {epoch:>3d}/{epochs}  "
                  f"loss={avg_loss:.4f}  acc={accuracy:.4f}")

    return log


# =====================================================================
#  Evaluation  (before / after LoRA)
# =====================================================================

def evaluate_model_on_graphs(model, graphs: list[dict]):
    """Evaluate any GNN-like model on a list of graphs. Returns per-graph accs."""
    model.eval()
    accs = []
    for g in graphs:
        if g["num_edges"] == 0:
            continue
        with torch.no_grad():
            nf = torch.tensor(g["node_feat"], dtype=torch.float32)
            ei = torch.tensor(g["edge_index"], dtype=torch.long)
            ey = torch.tensor(g["edge_y"], dtype=torch.long)
            logits, _ = model(nf, ei)
            preds = logits.argmax(dim=1)
            acc = (preds == ey).float().mean().item()
        accs.append(acc)
    return accs


# =====================================================================
#  Adaptation Curve Plot
# =====================================================================

def plot_adaptation_curve(log: list[dict], save_path: str):
    """Plot loss and accuracy over local LoRA training epochs."""
    epochs = [e["epoch"] for e in log]
    losses = [e["loss"] for e in log]
    accs   = [e["accuracy"] for e in log]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    c1, c2 = "#2563eb", "#dc2626"

    ax1.plot(epochs, losses, color=c1, linewidth=2.0, label="CE Loss")
    ax1.set_xlabel("Local Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12, color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.plot(epochs, accs, color=c2, linewidth=2.0, label="Accuracy")
    ax2.set_ylabel("Accuracy", fontsize=12, color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right",
               fontsize=10, framealpha=0.9)

    ax1.set_title("Local LoRA Drift Adaptation — On-Device Training",
                  fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Adaptation curve -> {save_path}")


# =====================================================================
#  Main
# =====================================================================

if __name__ == "__main__":
    import pickle

    t0 = time.time()
    print("=" * 60)
    print("  Step 7 — LoRA Drift Adaptation (On-Device)")
    print("=" * 60)

    # ── 1. Load artefacts ────────────────────────────────────────────
    ckpt = torch.load(os.path.join(OUTPUT_DIR, "ondevice_gnn.pt"),
                      map_location="cpu", weights_only=False)
    codebook = ckpt["codebook"]
    K, D = codebook.shape

    code_indices = np.load(os.path.join(OUTPUT_DIR, "code_indices.npy"))
    T, M = code_indices.shape

    with open(os.path.join(OUTPUT_DIR, "temporal_graphs.pkl"), "rb") as f:
        temporal_graphs = pickle.load(f)

    labels = np.load(os.path.join(OUTPUT_DIR, "fingerprint_labels.npy"))
    print(f"[INFO] Loaded {T} windows, {len(temporal_graphs)} graphs")

    # ── 2. Reconstruct base GNN for a representative window ─────────
    reconstructor = WeightReconstructor(codebook, node_feat_dim=15)

    # Use last attack window as the "current deployment" context
    attack_idxs = np.where(labels == 1)[0]
    deploy_idx = int(attack_idxs[-1])
    z_deploy = torch.tensor(code_indices[deploy_idx], dtype=torch.long)
    print(f"[INFO] Deployment window : {deploy_idx} "
          f"(label={'attack' if labels[deploy_idx] else 'normal'})")
    print(f"       Code indices z    : {z_deploy.tolist()}")

    weights_base = reconstructor(z_deploy)

    base_gnn = TinyGNN(node_feat_dim=15, hidden=32, num_classes=2)
    base_gnn.load_weights(weights_base)

    # ── 3. Build LoRA-augmented model ────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"[STEP 7a] LoRA Augmentation  (rank r={LORA_RANK})")
    print(f"{'─'*60}")

    lora_gnn = LoRATinyGNN(base_gnn, rank=LORA_RANK)
    print(f"\n{lora_gnn.lora_summary()}")

    # Verify base weights are frozen
    frozen_params = [n for n, p in lora_gnn.named_parameters()
                     if not p.requires_grad]
    trainable_params = [n for n, p in lora_gnn.named_parameters()
                        if p.requires_grad]
    print(f"\n  Frozen parameters  : {len(frozen_params)} "
          f"(should be 0 — base weights are buffers)")
    print(f"  Trainable params   : {len(trainable_params)}")
    for n in trainable_params:
        p = dict(lora_gnn.named_parameters())[n]
        print(f"    {n:>20s}  {str(tuple(p.shape)):>12s}  "
              f"numel={p.numel()}")

    # ── 4. Collect local labelled samples ────────────────────────────
    print(f"\n{'─'*60}")
    print(f"[STEP 7b] Collect Local Samples (max {MAX_LOCAL_SAMPLES})")
    print(f"{'─'*60}")

    local_graphs, total_edges = collect_local_samples(
        temporal_graphs, max_samples=MAX_LOCAL_SAMPLES)
    print(f"  Collected {len(local_graphs)} graph(s), "
          f"{total_edges} edge samples")

    # Count class balance in local data
    local_labels = np.concatenate([g["edge_y"] for g in local_graphs])
    unique, counts = np.unique(local_labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    class {u} : {c} edges  "
              f"({'normal' if u == 0 else 'attack'})")

    # ── 5. Evaluate BEFORE adaptation ────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"[STEP 7c] Baseline Evaluation (before LoRA training)")
    print(f"{'─'*60}")

    accs_before = evaluate_model_on_graphs(lora_gnn, local_graphs)
    mean_before = np.mean(accs_before)
    print(f"  Mean edge accuracy (before) : {mean_before:.4f}")

    # Also evaluate on ALL graphs
    all_accs_before = evaluate_model_on_graphs(lora_gnn, temporal_graphs)
    print(f"  Mean edge accuracy (all graphs, before) : {np.mean(all_accs_before):.4f}")

    # ── 6. Local LoRA training ───────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"[STEP 7d] Local LoRA Training ({LOCAL_EPOCHS} epochs, "
          f"lr={LOCAL_LR})")
    print(f"{'─'*60}")

    log = train_lora_local(lora_gnn, local_graphs,
                           epochs=LOCAL_EPOCHS, lr=LOCAL_LR)

    # ── 7. Evaluate AFTER adaptation ─────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"[STEP 7e] Post-Adaptation Evaluation")
    print(f"{'─'*60}")

    accs_after = evaluate_model_on_graphs(lora_gnn, local_graphs)
    mean_after = np.mean(accs_after)
    print(f"  Mean edge accuracy (local, after)  : {mean_after:.4f}")

    all_accs_after = evaluate_model_on_graphs(lora_gnn, temporal_graphs)
    mean_all_after = np.mean(all_accs_after)
    print(f"  Mean edge accuracy (all, after)    : {mean_all_after:.4f}")

    improvement = mean_after - mean_before
    print(f"\n  Local improvement  : {improvement:+.4f}")
    print(f"  Global drift       : {mean_all_after - np.mean(all_accs_before):+.4f}")

    # ── 8. Inspect LoRA deltas ───────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"[STEP 7f] LoRA Delta Analysis")
    print(f"{'─'*60}")

    for name, lora in [("W1", lora_gnn.lora1), ("W2", lora_gnn.lora2),
                        ("W3", lora_gnn.lora3), ("W_cls", lora_gnn.lora_cls)]:
        delta = lora.delta().detach()
        base = getattr(lora_gnn, f"{name}_base")
        ratio = delta.norm() / base.norm()
        print(f"  {name:>6s}  ||delta||={delta.norm():.4f}  "
              f"||base||={base.norm():.4f}  "
              f"ratio={ratio:.4f}  "
              f"rank={lora.rank}")

    # ── 9. Save artefacts ────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"[STEP 7g] Saving Artefacts")
    print(f"{'─'*60}")

    # Checkpoint
    save_dict = {
        "codebook": codebook,
        "deploy_z": z_deploy,
        "deploy_idx": deploy_idx,
        "base_weights": {k: v.clone() for k, v in weights_base.items()},
        "lora_state_dict": lora_gnn.state_dict(),
        "lora_config": {
            "rank": LORA_RANK,
            "local_epochs": LOCAL_EPOCHS,
            "local_lr": LOCAL_LR,
            "max_local_samples": MAX_LOCAL_SAMPLES,
            "actual_samples": total_edges,
        },
        "metrics": {
            "mean_acc_before": mean_before,
            "mean_acc_after": mean_after,
            "improvement": improvement,
            "mean_all_before": float(np.mean(all_accs_before)),
            "mean_all_after": mean_all_after,
        },
    }
    ckpt_path = os.path.join(OUTPUT_DIR, "lora_adapted_gnn.pt")
    torch.save(save_dict, ckpt_path)
    print(f"  Checkpoint -> {ckpt_path}")

    # Adaptation curve plot
    plot_path = os.path.join(OUTPUT_DIR, "lora_adaptation_curve.png")
    plot_adaptation_curve(log, plot_path)

    # Summary report
    summary_path = os.path.join(OUTPUT_DIR, "lora_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  Step 7 — LoRA Drift Adaptation (On-Device)\n")
        f.write("=" * 60 + "\n\n")

        f.write("LoRA Configuration\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Rank r             : {LORA_RANK} (max 4)\n")
        f.write(f"  Local epochs       : {LOCAL_EPOCHS}\n")
        f.write(f"  Learning rate      : {LOCAL_LR}\n")
        f.write(f"  Max local samples  : {MAX_LOCAL_SAMPLES}\n")
        f.write(f"  Actual samples     : {total_edges}\n\n")

        f.write("Architecture\n")
        f.write("-" * 40 + "\n")
        f.write(lora_gnn.lora_summary() + "\n\n")

        f.write("Adaptation Results\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Local accuracy (before)  : {mean_before:.4f}\n")
        f.write(f"  Local accuracy (after)   : {mean_after:.4f}\n")
        f.write(f"  Improvement              : {improvement:+.4f}\n")
        f.write(f"  Global accuracy (before) : {np.mean(all_accs_before):.4f}\n")
        f.write(f"  Global accuracy (after)  : {mean_all_after:.4f}\n\n")

        f.write("LoRA Delta Norms\n")
        f.write("-" * 40 + "\n")
        for name, lora in [("W1", lora_gnn.lora1), ("W2", lora_gnn.lora2),
                            ("W3", lora_gnn.lora3), ("W_cls", lora_gnn.lora_cls)]:
            delta = lora.delta().detach()
            base = getattr(lora_gnn, f"{name}_base")
            ratio = delta.norm() / base.norm()
            f.write(f"  {name:>6s}  ||delta||/||base|| = {ratio:.4f}\n")

        f.write(f"\nTraining Log (sampled)\n")
        f.write(f"{'Ep':>5s}  {'Loss':>8s}  {'Acc':>8s}\n")
        for e in log:
            if e["epoch"] % 10 == 0 or e["epoch"] == 1 or e["epoch"] == LOCAL_EPOCHS:
                f.write(f"{e['epoch']:>5d}  {e['loss']:>8.4f}  "
                        f"{e['accuracy']:>8.4f}\n")

    print(f"  Summary    -> {summary_path}")
    print(f"\n[INFO] Step 7 complete in {time.time()-t0:.1f}s")
