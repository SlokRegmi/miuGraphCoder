"""
Step 4 – Codebook Verification & Diagnostics
=============================================
Loads the trained hypernetwork + VQ codebook from Step 3/4 and runs a
battery of checks:

1. STE gradient flow verification
2. EMA codebook update verification
3. Codebook utilization analysis
4. Dead-code revival test
5. Reconstruction quality check
6. Codebook embedding visualisation (PCA)

Outputs  (in ./output/)
-----------------------
codebook_diagnostics.txt     – full diagnostic report
codebook_embeddings_pca.png  – PCA scatter of codebook vectors
codebook_usage_hist.png      – histogram of code utilization
"""

import os, sys, pickle, warnings

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
sys.path.insert(0, BASE_DIR)

from vq_codebook import VQCodebook
from hypernetwork import Hypernetwork, DEVICE

# =====================================================================
#  Load model
# =====================================================================
print("[INFO] Loading trained hypernetwork …")
ckpt = torch.load(os.path.join(OUTPUT_DIR, "hypernetwork_state.pt"),
                   map_location=DEVICE, weights_only=False)
cfg = ckpt["config"]
model = Hypernetwork(
    d_in=cfg["d_in"], hidden=cfg["hidden"],
    M=cfg["M"], D=cfg["D"], K=cfg["K"],
    commitment=cfg["commitment"], dropout=cfg["dropout"],
)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE)
fp_mean = ckpt["fp_mean"]
fp_std  = ckpt["fp_std"]

# Load data
fingerprints = np.load(os.path.join(OUTPUT_DIR, "fingerprints.npy")).astype(np.float32)
fingerprints = np.nan_to_num(fingerprints, nan=0.0, posinf=0.0, neginf=0.0)
X_norm = (fingerprints - fp_mean) / fp_std
labels = np.load(os.path.join(OUTPUT_DIR, "fingerprint_labels.npy"))

report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(msg)

log("=" * 65)
log("  Step 4 – VQ Codebook Diagnostics")
log("=" * 65)

# =====================================================================
#  1. STE gradient flow verification
# =====================================================================
log("\n--- 1. Straight-Through Estimator (STE) Gradient Flow ---")

model.train()
x = torch.tensor(X_norm[:16], dtype=torch.float32, device=DEVICE)
x.requires_grad_(True)

f_hat, z_q, indices, vq_loss, diag = model(x)
recon_loss = F.mse_loss(f_hat, x)
total_loss = recon_loss + vq_loss
total_loss.backward()

# Check gradients reached the encoder through the STE
encoder_grad_ok = all(
    p.grad is not None and p.grad.abs().sum() > 0
    for p in model.encoder.parameters() if p.requires_grad
)
# Check gradients reached the decoder
decoder_grad_ok = all(
    p.grad is not None and p.grad.abs().sum() > 0
    for p in model.decoder.parameters() if p.requires_grad
)
# Input gradient exists (STE passes grad from z_q back to z_e → back to x)
input_grad_ok = x.grad is not None and x.grad.abs().sum() > 0

log(f"  Encoder gradients flow through STE : {'PASS' if encoder_grad_ok else 'FAIL'}")
log(f"  Decoder gradients present          : {'PASS' if decoder_grad_ok else 'FAIL'}")
log(f"  Input gradients via STE            : {'PASS' if input_grad_ok else 'FAIL'}")

# Show gradient magnitudes
for name, p in model.encoder.named_parameters():
    if p.grad is not None:
        log(f"    encoder.{name:20s}  grad_norm={p.grad.norm():.6f}")
        break  # just show first

log(f"  VQ loss from forward: {vq_loss.item():.6f}")
log(f"  Diagnostics: perplexity={diag['perplexity']:.2f}, "
    f"utilization={diag['utilization']:.2%}")

model.zero_grad()

# =====================================================================
#  2. EMA codebook update verification
# =====================================================================
log("\n--- 2. EMA Codebook Update ---")

vq = model.vq
codebook_before = vq.get_codebook().clone()

# Run one forward in training mode to trigger EMA update
model.train()
with torch.enable_grad():
    f_hat, z_q, indices, vq_loss, diag = model(x)
    (F.mse_loss(f_hat, x) + vq_loss).backward()

codebook_after = vq.get_codebook()
delta = (codebook_after - codebook_before).abs()
log(f"  Codebook changed after EMA update  : {'YES' if delta.sum() > 0 else 'NO'}")
log(f"  Mean |Δ embed|                     : {delta.mean():.8f}")
log(f"  Max  |Δ embed|                     : {delta.max():.8f}")
log(f"  EMA decay (γ)                      : {vq.ema_decay}")
model.zero_grad()

# =====================================================================
#  3. Codebook utilization analysis
# =====================================================================
log("\n--- 3. Codebook Utilization ---")

model.eval()
with torch.no_grad():
    X_all = torch.tensor(X_norm, dtype=torch.float32, device=DEVICE)
    all_indices = model.encode(X_all).cpu().numpy()   # (T, M)

unique_codes = np.unique(all_indices)
code_counts  = np.bincount(all_indices.flatten(), minlength=cfg["K"])

log(f"  Codebook size (K)        : {cfg['K']}")
log(f"  Code dimension (D)       : {cfg['D']}")
log(f"  Code sequence length (M) : {cfg['M']}")
log(f"  Unique codes used        : {len(unique_codes)}/{cfg['K']}  "
    f"({100*len(unique_codes)/cfg['K']:.1f}%)")
log(f"  Most  used code          : index {code_counts.argmax()} "
    f"(count={code_counts.max()})")
log(f"  Least used code (>0)     : index {code_counts[code_counts > 0].argmin()} "
    f"(count={code_counts[code_counts > 0].min()})")
log(f"  Unused codes             : {(code_counts == 0).sum()}")
log(f"  Code count std           : {code_counts[code_counts > 0].std():.2f}")

# EMA cluster sizes
stats = vq.get_usage_stats()
log(f"  EMA active codes         : {stats['active_codes']}/{stats['total_codes']}")
log(f"  EMA mean cluster size    : {stats['mean_cluster_size']:.4f}")
log(f"  EMA std cluster size     : {stats['std_cluster_size']:.4f}")
log(f"  Last perplexity          : {stats['perplexity']:.2f}")

# =====================================================================
#  4. Dead-code revival test
# =====================================================================
log("\n--- 4. Dead-Code Revival ---")

# Create a fresh VQ, kill some codes, verify revival
test_vq = VQCodebook(num_codes=16, code_dim=8, dead_threshold=2.0)
# Force init
test_data = torch.randn(32, 4, 8)
test_vq.train()
_ = test_vq(test_data)

# Manually kill some codes
test_vq.cluster_size.data[0:4] = 0.0  # mark 4 codes as dead

n_revived = test_vq._revive_dead_codes(test_data.reshape(-1, 8))
log(f"  Killed 4 codes (cluster_size=0)")
log(f"  Codes revived by revival  : {n_revived}")
log(f"  Cluster sizes after revival: all >= 1.0? "
    f"{'PASS' if (test_vq.cluster_size >= 1.0).all() else 'FAIL'}")

# =====================================================================
#  5. Reconstruction quality
# =====================================================================
log("\n--- 5. Reconstruction Quality ---")

model.eval()
with torch.no_grad():
    f_hat_all, _, _, _, _ = model(X_all)
    recon_mse = F.mse_loss(f_hat_all, X_all).item()
    per_dim_mse = ((f_hat_all - X_all) ** 2).mean(dim=0).cpu().numpy()

log(f"  Overall reconstruction MSE : {recon_mse:.6f}")
log(f"  Per-dim MSE range          : [{per_dim_mse.min():.6f}, {per_dim_mse.max():.6f}]")
log(f"  Mean per-dim MSE           : {per_dim_mse.mean():.6f}")

# Information preserved: correlation between original and reconstructed
from scipy.stats import pearsonr
orig = X_all.cpu().numpy().flatten()
recon = f_hat_all.cpu().numpy().flatten()
corr, _ = pearsonr(orig, recon)
log(f"  Pearson correlation (orig ↔ recon) : {corr:.4f}")

# =====================================================================
#  6. Visualisations
# =====================================================================
log("\n--- 6. Generating Visualisations ---")

# 6a. PCA of codebook embeddings
codebook_np = vq.get_codebook().cpu().numpy()  # (K, D)
pca = PCA(n_components=2)
C_2d = pca.fit_transform(codebook_np)

fig, ax = plt.subplots(figsize=(7, 6))
sizes = code_counts / code_counts.max() * 300 + 20  # scale marker size by usage
sc = ax.scatter(C_2d[:, 0], C_2d[:, 1], s=sizes, c=code_counts,
                cmap="viridis", alpha=0.8, edgecolors="k", linewidths=0.5)
for i in range(cfg["K"]):
    ax.annotate(str(i), (C_2d[i, 0], C_2d[i, 1]), fontsize=6,
                ha="center", va="center")
plt.colorbar(sc, label="Usage count")
ax.set_title(f"Codebook Embeddings (PCA)  —  K={cfg['K']}, D={cfg['D']}")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.grid(True, alpha=0.3)
fig.tight_layout()
pca_path = os.path.join(OUTPUT_DIR, "codebook_embeddings_pca.png")
fig.savefig(pca_path, dpi=200)
plt.close(fig)
log(f"  Saved: {pca_path}")

# 6b. Code usage histogram
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(cfg["K"]), code_counts, color="#1f77b4", edgecolor="k", linewidth=0.3)
ax.axhline(code_counts[code_counts > 0].mean(), color="red", ls="--",
           label=f"Mean = {code_counts[code_counts>0].mean():.1f}")
ax.set_xlabel("Code index $k$")
ax.set_ylabel("Assignment count")
ax.set_title(f"VQ Codebook Usage Distribution  —  "
             f"{len(unique_codes)}/{cfg['K']} codes active")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
hist_path = os.path.join(OUTPUT_DIR, "codebook_usage_hist.png")
fig.savefig(hist_path, dpi=200)
plt.close(fig)
log(f"  Saved: {hist_path}")

# =====================================================================
#  Save report
# =====================================================================
log("\n" + "=" * 65)
report_path = os.path.join(OUTPUT_DIR, "codebook_diagnostics.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"\n[INFO] Full diagnostic report → {report_path}")
print("[INFO] Done.")
