"""
Step 5 – Hardware-Aware Composite Loss
=======================================
    L = L_task  +  L_recon  +  L_vq  +  λ₁·L_lat  +  λ₂·L_mem

Components
----------
  L_task  : Cross-Entropy for binary event classification (normal / attack)
  L_recon : MSE fingerprint reconstruction (self-supervised auxiliary)
  L_vq    : Codebook + commitment loss (from VQ-VAE, Step 4)
  L_lat   : Differentiable latency surrogate – penalises exceeding target
  L_mem   : Differentiable memory surrogate – penalises exceeding target

Latency surrogate
    Per-layer base times come from a lookup table profiled for an
    ARM Cortex-M4 @ 168 MHz.  Differentiation is achieved by scaling
    each layer's contribution by its normalised Frobenius weight norm:
        lat  =  Σ_l  t_l · (‖W_l‖_F / √(fan_in · fan_out))
    Weight decay / pruning pressure → smaller norms → lower predicted
    latency → loss decreases.

Memory surrogate
    Param memory : Σ_l  mean(|W_l|) · numel(W_l) · 4   (bytes, differentiable)
    Activation   : analytical constant from forward-pass buffer sizes
    Penalty fires only when total exceeds the target threshold.

Outputs (in ./output/)
----------------------
  hw_aware_model.pt             – trained model checkpoint
  hw_aware_training_log.json    – per-epoch loss & hardware metrics
  training_convergence.png      – convergence plot with dual Y-axes
  hw_aware_summary.txt          – training summary report
"""

import os, sys, time, json, warnings
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

from vq_codebook import VQCodebook
from hypernetwork import (
    Hypernetwork, FINGERPRINT_DIM, HIDDEN_DIM,
    NUM_CODES_M, CODE_EMB_DIM, CODEBOOK_SIZE_K,
    COMMITMENT_COST, DROPOUT,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================================
#  Training hyper-parameters
# =====================================================================
EPOCHS          = 200
BATCH_SIZE      = 32
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
LAMBDA_LAT      = 0.10      # λ₁  latency penalty weight
LAMBDA_MEM      = 0.01      # λ₂  memory penalty weight
LAT_TARGET_MS   = 2.0       # target: ≤ 2 ms per inference on MCU
MEM_TARGET_KB   = 900.0     # target: ≤ 900 KB total footprint
NUM_CLASSES     = 2


# =====================================================================
#  Latency Lookup Table  (LUT)
# =====================================================================

class LatencyLUT:
    """
    Lookup table of estimated execution times (ms) for layer operations
    on the target MCU (ARM Cortex-M4 @ 168 MHz).

    Profiling assumptions:
        - ~10 clock cycles per MAC   (int8 inference with CMSIS-NN)
        - GELU  ≈ 5 cycles / element (polynomial approximation)
        - LayerNorm ≈ 20 cycles / element (mean + var + normalise)
    """

    CYCLES_PER_MAC  = 10
    CLOCK_MHZ       = 168       # ARM Cortex-M4

    def __init__(self):
        self._table: dict[tuple, float] = {}
        self._populate()

    def _populate(self):
        """Pre-populate profiled latencies for known layer configs."""
        #          (type,       in,  out) →  ms
        entries = [
            # ── Hypernetwork encoder ──────────────────────────────
            ("layernorm",  43,  43,   0.005),
            ("linear",     43, 256,   0.061),   # 11 008 MACs
            ("gelu",      256, 256,   0.008),
            ("linear",    256, 256,   0.390),   # 65 536 MACs
            ("linear",    256, 256,   0.390),   # encoder → M·D
            # ── Hypernetwork decoder (auxiliary) ──────────────────
            ("linear",    256, 256,   0.390),
            ("linear",    256,  43,   0.066),
            # ── Classification head ───────────────────────────────
            ("linear",    256,  64,   0.098),   # 16 384 MACs
            ("gelu",       64,  64,   0.002),
            ("linear",     64,   2,   0.008),   #    128 MACs
        ]
        for typ, din, dout, ms in entries:
            self._table[(typ, din, dout)] = ms

    def lookup(self, layer_type: str, dim_in: int, dim_out: int) -> float:
        """Return profiled latency or fall back to a MAC-proportional estimate."""
        key = (layer_type, dim_in, dim_out)
        if key in self._table:
            return self._table[key]
        if layer_type == "linear":
            macs = dim_in * dim_out
            return (macs * self.CYCLES_PER_MAC) / (self.CLOCK_MHZ * 1e3)
        return 0.001

    def total_baseline_ms(self) -> float:
        """Sum of all profiled entries (static baseline at init)."""
        return sum(self._table.values())

    def as_dict(self) -> dict:
        return {f"{k[0]}({k[1]}→{k[2]})": f"{v:.4f} ms" for k, v in self._table.items()}


# =====================================================================
#  Classifier  (Hypernetwork + classification head)
# =====================================================================

class HypernetClassifier(nn.Module):
    """
    Extends the Hypernetwork with a classification head for binary
    event classification  (normal = 0,  attack = 1).

        f_t → Encoder → VQ → z_q (B, M·D) → Classifier → logits (B, 2)
                                             → Decoder   → f_hat  (B, d_in)
    """

    def __init__(self, hypernet: Hypernetwork, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.hypernet = hypernet
        self.classifier = nn.Sequential(
            nn.Linear(NUM_CODES_M * CODE_EMB_DIM, 64),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, num_classes),
        )

    # ---- helpers used by surrogates ----
    @property
    def M(self):
        return self.hypernet.M

    @property
    def D(self):
        return self.hypernet.D

    def forward(self, f_t: torch.Tensor):
        """
        Returns
        -------
        logits  : (B, num_classes)
        f_hat   : (B, d_in)        reconstructed fingerprint
        z_q     : (B, M, D)        quantised codes
        indices : (B, M)           discrete code indices
        vq_loss : scalar
        diag    : dict
        """
        B = f_t.size(0)
        z_e = self.hypernet.encoder(f_t).view(B, self.M, self.D)
        z_q, indices, vq_loss, diag = self.hypernet.vq(z_e)

        flat_q = z_q.view(B, -1)
        logits = self.classifier(flat_q)
        f_hat  = self.hypernet.decoder(flat_q)

        return logits, f_hat, z_q, indices, vq_loss, diag


# =====================================================================
#  Differentiable Hardware Surrogates
# =====================================================================

def latency_surrogate(model: HypernetClassifier,
                      lut: LatencyLUT) -> torch.Tensor:
    """
    Differentiable latency estimate (ms).

    For each Linear layer the contribution is:
        lat_l  =  base_ms_l  ×  (‖W_l‖_F / √(fan_in · fan_out))

    At initialisation the normalised Frobenius norm ≈ 1, so latency ≈
    baseline.  Weight decay / pruning shrinks norms → latency decreases.
    """
    dev = next(model.parameters()).device
    total = torch.tensor(0.0, device=dev, requires_grad=True)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            din  = module.in_features
            dout = module.out_features
            base = lut.lookup("linear", din, dout)
            w_scale = module.weight.norm(p="fro") / (din * dout) ** 0.5
            total = total + base * w_scale

    return total                                              # scalar (ms)


def memory_surrogate(model: HypernetClassifier,
                     batch_size: int = 1) -> torch.Tensor:
    """
    Differentiable memory estimate (KB).

    1. Parameter memory (differentiable):
         Σ_l  mean(|W_l|) × numel(W_l) × 4   bytes
       At init ≈ total_param_bytes; shrinks under L1 pressure.

    2. Activation memory (analytical constant):
         Sum of forward-pass intermediate buffer sizes × 4 bytes.
    """
    dev = next(model.parameters()).device

    # ── differentiable parameter footprint ───────────────────────────
    param_mem = torch.tensor(0.0, device=dev, requires_grad=True)
    for p in model.parameters():
        param_mem = param_mem + p.abs().mean() * p.numel() * 4.0   # bytes

    # ── activation footprint (constant, per batch_size=1) ────────────
    # Encoder:  LN(43) + h1(256) + h2(256) + proj(256)   =  811 floats
    # VQ:       z_q (256)                                  =  256 floats
    # Classifier: h(64) + logits(2)                        =   66 floats
    # Decoder: h(256) + f_hat(43)                          =  299 floats
    act_floats   = 43 + 256 + 256 + 256 + 256 + 64 + 2 + 256 + 43
    act_bytes    = batch_size * act_floats * 4
    act_const    = torch.tensor(float(act_bytes), device=dev)

    total_kb = (param_mem + act_const) / 1024.0
    return total_kb                                           # scalar (KB)


# =====================================================================
#  Composite Loss Module
# =====================================================================

class HardwareAwareLoss(nn.Module):
    """
    L  =  L_task  +  L_recon  +  L_vq
        + λ₁·max(0, lat − lat_target)²
        + λ₂·max(0, mem − mem_target)²
    """

    def __init__(self, lut: LatencyLUT,
                 class_weights: torch.Tensor | None = None,
                 lambda_lat: float  = LAMBDA_LAT,
                 lambda_mem: float  = LAMBDA_MEM,
                 lat_target: float  = LAT_TARGET_MS,
                 mem_target: float  = MEM_TARGET_KB):
        super().__init__()
        self.lut        = lut
        self.lambda_lat = lambda_lat
        self.lambda_mem = lambda_mem
        self.lat_target = lat_target
        self.mem_target = mem_target
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, model, logits, labels, f_hat, f_t, vq_loss, batch_size):
        # ── Task loss (cross-entropy) ────────────────────────────────
        L_task = F.cross_entropy(logits, labels, weight=self.class_weights)

        # ── Reconstruction auxiliary loss ────────────────────────────
        L_recon = F.mse_loss(f_hat, f_t)

        # ── Hardware surrogates ──────────────────────────────────────
        lat_pred = latency_surrogate(model, self.lut)
        mem_pred = memory_surrogate(model, batch_size)

        # Penalty: only fires when exceeding target
        L_lat = F.relu(lat_pred - self.lat_target) ** 2
        L_mem = F.relu(mem_pred - self.mem_target) ** 2

        # ── Composite ────────────────────────────────────────────────
        L = (L_task
             + L_recon
             + vq_loss
             + self.lambda_lat * L_lat
             + self.lambda_mem * L_mem)

        metrics = {
            "L_total":  L.item(),
            "L_task":   L_task.item(),
            "L_recon":  L_recon.item(),
            "L_vq":     vq_loss.item(),
            "L_lat":    (self.lambda_lat * L_lat).item(),
            "L_mem":    (self.lambda_mem * L_mem).item(),
            "lat_ms":   lat_pred.item(),
            "mem_kb":   mem_pred.item(),
        }
        return L, metrics


# =====================================================================
#  Training loop
# =====================================================================

def train(model: HypernetClassifier,
          loss_fn: HardwareAwareLoss,
          X: np.ndarray,
          y: np.ndarray,
          epochs: int   = EPOCHS,
          bs: int       = BATCH_SIZE,
          lr: float     = LR,
          device: str   = DEVICE):

    model.to(device)
    loss_fn.to(device)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=False)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    log = []
    model.train()

    for epoch in range(1, epochs + 1):
        epoch_m = {k: 0.0 for k in ["L_total", "L_task", "L_recon",
                                      "L_vq", "L_lat", "L_mem",
                                      "lat_ms", "mem_kb"]}
        n_correct = 0
        n_total   = 0
        n_batches = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits, f_hat, z_q, indices, vq_loss, diag = model(batch_x)

            L, metrics = loss_fn(model, logits, batch_y, f_hat,
                                 batch_x, vq_loss, batch_x.size(0))

            optimiser.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            for k in epoch_m:
                epoch_m[k] += metrics[k]
            n_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            n_total   += batch_y.size(0)
            n_batches += 1

        scheduler.step()

        # Average over batches
        for k in epoch_m:
            epoch_m[k] /= n_batches
        epoch_m["accuracy"] = n_correct / n_total
        epoch_m["epoch"]    = epoch

        log.append(epoch_m)

        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Ep {epoch:>4d}/{epochs}  "
                  f"L={epoch_m['L_total']:.4f}  "
                  f"CE={epoch_m['L_task']:.4f}  "
                  f"acc={epoch_m['accuracy']:.3f}  "
                  f"lat={epoch_m['lat_ms']:.3f}ms  "
                  f"mem={epoch_m['mem_kb']:.1f}KB")

    return log


# =====================================================================
#  Convergence Plot
# =====================================================================

def plot_convergence(log: list[dict], save_path: str):
    """
    Training convergence plot:
      - Left Y-axis : total loss L (+ component breakdown)
      - Right Y-axis 1 : predicted latency (ms) with target threshold
      - Right Y-axis 2 : predicted memory (KB) with target threshold
    """
    epochs   = [e["epoch"]   for e in log]
    L_total  = [e["L_total"] for e in log]
    L_task   = [e["L_task"]  for e in log]
    L_recon  = [e["L_recon"] for e in log]
    lat_ms   = [e["lat_ms"]  for e in log]
    mem_kb   = [e["mem_kb"]  for e in log]
    acc      = [e["accuracy"] for e in log]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # ── Left axis: losses ────────────────────────────────────────────
    c1, c2, c3 = "#2563eb", "#7c3aed", "#059669"
    ax1.plot(epochs, L_total, color=c1, linewidth=2.0, label="L total")
    ax1.plot(epochs, L_task,  color=c2, linewidth=1.2, alpha=0.7,
             linestyle="--", label="L_task (CE)")
    ax1.plot(epochs, L_recon, color=c3, linewidth=1.2, alpha=0.7,
             linestyle=":", label="L_recon (MSE)")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12, color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_ylim(bottom=0)

    # ── Right axis 1: latency ────────────────────────────────────────
    ax2 = ax1.twinx()
    c4 = "#dc2626"
    ax2.plot(epochs, lat_ms, color=c4, linewidth=1.5, label="Latency (ms)")
    ax2.axhline(y=LAT_TARGET_MS, color=c4, linewidth=1.0, linestyle="--",
                alpha=0.5, label=f"Lat target ({LAT_TARGET_MS} ms)")
    ax2.set_ylabel("Latency (ms)", fontsize=12, color=c4)
    ax2.tick_params(axis="y", labelcolor=c4)

    # ── Right axis 2: memory (offset) ───────────────────────────────
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.12))
    c5 = "#ea580c"
    ax3.plot(epochs, mem_kb, color=c5, linewidth=1.5, label="Memory (KB)")
    ax3.axhline(y=MEM_TARGET_KB, color=c5, linewidth=1.0, linestyle="--",
                alpha=0.5, label=f"Mem target ({MEM_TARGET_KB} KB)")
    ax3.set_ylabel("Memory (KB)", fontsize=12, color=c5)
    ax3.tick_params(axis="y", labelcolor=c5)

    # ── Legend ────────────────────────────────────────────────────────
    lines  = ax1.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    lines3 = ax3.get_legend_handles_labels()
    all_handles = lines[0] + lines2[0] + lines3[0]
    all_labels  = lines[1] + lines2[1] + lines3[1]
    ax1.legend(all_handles, all_labels, loc="upper right", fontsize=9,
               framealpha=0.9)

    ax1.set_title("Training Convergence — Hardware-Aware Composite Loss",
                  fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Convergence plot -> {save_path}")


# =====================================================================
#  Main
# =====================================================================

if __name__ == "__main__":
    t0 = time.time()

    # ── 1. Load data ─────────────────────────────────────────────────
    fp_path  = os.path.join(OUTPUT_DIR, "fingerprints.npy")
    lbl_path = os.path.join(OUTPUT_DIR, "fingerprint_labels.npy")
    ckpt_path = os.path.join(OUTPUT_DIR, "hypernetwork_state.pt")

    print("[INFO] Loading fingerprints & labels …")
    fingerprints = np.load(fp_path).astype(np.float32)
    fingerprints = np.nan_to_num(fingerprints, nan=0.0, posinf=0.0, neginf=0.0)
    labels = np.load(lbl_path).astype(np.int64)
    T, d_in = fingerprints.shape
    print(f"       {T} samples, dim={d_in}, classes={np.unique(labels)}")

    # Standardise
    fp_mean = fingerprints.mean(axis=0)
    fp_std  = fingerprints.std(axis=0)
    fp_std[fp_std < 1e-8] = 1.0
    X_norm = (fingerprints - fp_mean) / fp_std

    # ── 2. Build model: load pretrained hypernetwork + add classifier ─
    print("[INFO] Loading pretrained hypernetwork …")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = checkpoint["config"]

    hypernet = Hypernetwork(
        d_in=cfg["d_in"], hidden=cfg["hidden"],
        M=cfg["M"], D=cfg["D"], K=cfg["K"],
        commitment=cfg["commitment"], dropout=cfg["dropout"],
    )
    hypernet.load_state_dict(
        {k: v for k, v in checkpoint["model_state_dict"].items()},
        strict=True,
    )
    print(f"       Hypernetwork loaded  ({sum(p.numel() for p in hypernet.parameters()):,} params)")

    model = HypernetClassifier(hypernet, num_classes=NUM_CLASSES)
    total_params = sum(p.numel() for p in model.parameters())
    cls_params   = sum(p.numel() for p in model.classifier.parameters())
    print(f"       + Classifier head    ({cls_params:,} params)")
    print(f"       = Total              ({total_params:,} params)")
    print(f"       Model size           ≈ {total_params * 4 / 1024:.1f} KB (float32)")

    # ── 3. Loss & lookup table ───────────────────────────────────────
    lut = LatencyLUT()
    print(f"\n[INFO] Latency LUT baseline : {lut.total_baseline_ms():.3f} ms")
    for k, v in lut.as_dict().items():
        print(f"       {k:>25s}  {v}")

    # Class weights for imbalanced data
    counts = np.bincount(labels)
    weights = torch.tensor(T / (NUM_CLASSES * counts), dtype=torch.float32)
    print(f"\n[INFO] Class weights : {weights.tolist()}")

    loss_fn = HardwareAwareLoss(
        lut=lut,
        class_weights=weights,
        lambda_lat=LAMBDA_LAT,
        lambda_mem=LAMBDA_MEM,
        lat_target=LAT_TARGET_MS,
        mem_target=MEM_TARGET_KB,
    )

    print(f"       λ_lat={LAMBDA_LAT}  target={LAT_TARGET_MS} ms")
    print(f"       λ_mem={LAMBDA_MEM}  target={MEM_TARGET_KB} KB")

    # ── 4. Train ─────────────────────────────────────────────────────
    print(f"\n[INFO] Training for {EPOCHS} epochs on {DEVICE} …")
    log = train(model, loss_fn, X_norm, labels)

    # ── 5. Final evaluation ──────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_norm, dtype=torch.float32).to(DEVICE)
        y_t = torch.tensor(labels, dtype=torch.long).to(DEVICE)
        logits, f_hat, z_q, indices, vq_loss, diag = model(X_t)
        preds = logits.argmax(dim=1)
        acc = (preds == y_t).float().mean().item()
        lat_final = latency_surrogate(model, lut).item()
        mem_final = memory_surrogate(model).item()

    print(f"\n{'='*60}")
    print(f"  Final Results")
    print(f"{'='*60}")
    print(f"  Accuracy          : {acc:.4f}  ({(preds == y_t).sum().item()}/{T})")
    print(f"  Latency (pred)    : {lat_final:.3f} ms   (target ≤ {LAT_TARGET_MS} ms)")
    print(f"  Memory  (pred)    : {mem_final:.1f} KB    (target ≤ {MEM_TARGET_KB} KB)")
    print(f"  Lat within target : {'YES' if lat_final <= LAT_TARGET_MS else 'NO'}")
    print(f"  Mem within target : {'YES' if mem_final <= MEM_TARGET_KB else 'NO'}")

    # ── 6. Save artefacts ────────────────────────────────────────────
    # Checkpoint
    save_path = os.path.join(OUTPUT_DIR, "hw_aware_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "fp_mean": fp_mean,
        "fp_std":  fp_std,
        "config": {
            **cfg,
            "num_classes": NUM_CLASSES,
            "lambda_lat":  LAMBDA_LAT,
            "lambda_mem":  LAMBDA_MEM,
            "lat_target":  LAT_TARGET_MS,
            "mem_target":  MEM_TARGET_KB,
        },
        "final_accuracy":  acc,
        "final_latency_ms": lat_final,
        "final_memory_kb":  mem_final,
    }, save_path)
    print(f"\n[INFO] Model saved         -> {save_path}")

    # Training log
    log_path = os.path.join(OUTPUT_DIR, "hw_aware_training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"[INFO] Training log        -> {log_path}")

    # Convergence plot
    plot_path = os.path.join(OUTPUT_DIR, "training_convergence.png")
    plot_convergence(log, plot_path)

    # Summary report
    summary_path = os.path.join(OUTPUT_DIR, "hw_aware_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  Step 5 - Hardware-Aware Composite Loss\n")
        f.write("=" * 60 + "\n\n")
        f.write("Composite Loss:\n")
        f.write("  L = L_task + L_recon + L_vq + lambda1*L_lat + lambda2*L_mem\n\n")
        f.write(f"Hyperparameters:\n")
        f.write(f"  Epochs           : {EPOCHS}\n")
        f.write(f"  Batch size       : {BATCH_SIZE}\n")
        f.write(f"  Learning rate    : {LR}\n")
        f.write(f"  Weight decay     : {WEIGHT_DECAY}\n")
        f.write(f"  lambda_lat       : {LAMBDA_LAT}\n")
        f.write(f"  lambda_mem       : {LAMBDA_MEM}\n")
        f.write(f"  Lat target       : {LAT_TARGET_MS} ms\n")
        f.write(f"  Mem target       : {MEM_TARGET_KB} KB\n")
        f.write(f"  Num classes      : {NUM_CLASSES}\n\n")
        f.write(f"Model:\n")
        f.write(f"  Total params     : {total_params:,}\n")
        f.write(f"  Classifier params: {cls_params:,}\n")
        f.write(f"  Model size       : {total_params * 4 / 1024:.1f} KB\n\n")
        f.write(f"Latency LUT (ARM Cortex-M4 @ {LatencyLUT.CLOCK_MHZ} MHz):\n")
        for k, v in lut.as_dict().items():
            f.write(f"  {k:>28s}  {v}\n")
        f.write(f"  {'Baseline total':>28s}  {lut.total_baseline_ms():.4f} ms\n\n")
        f.write(f"Final Results:\n")
        f.write(f"  Accuracy         : {acc:.4f}\n")
        f.write(f"  Latency (pred)   : {lat_final:.3f} ms  (target <= {LAT_TARGET_MS})\n")
        f.write(f"  Memory  (pred)   : {mem_final:.1f} KB   (target <= {MEM_TARGET_KB})\n")
        f.write(f"  Lat OK           : {'YES' if lat_final <= LAT_TARGET_MS else 'NO'}\n")
        f.write(f"  Mem OK           : {'YES' if mem_final <= MEM_TARGET_KB else 'NO'}\n\n")
        f.write(f"Training log (sampled):\n")
        f.write(f"{'Ep':>5s}  {'L_total':>8s}  {'L_task':>8s}  {'Acc':>6s}  "
                f"{'Lat ms':>7s}  {'Mem KB':>8s}\n")
        for e in log:
            if e["epoch"] % 20 == 0 or e["epoch"] == 1 or e["epoch"] == EPOCHS:
                f.write(f"{e['epoch']:>5d}  {e['L_total']:>8.4f}  {e['L_task']:>8.4f}  "
                        f"{e['accuracy']:>6.3f}  {e['lat_ms']:>7.3f}  "
                        f"{e['mem_kb']:>8.1f}\n")

    print(f"[INFO] Summary             -> {summary_path}")
    print(f"\n[INFO] Step 5 complete in {time.time()-t0:.1f}s")
