"""
Step 3 – Server-Side Hypernetwork  H(f_t) -> z
================================================
An MLP hypernetwork that lives on the simulated cloud server.

Architecture
------------
                  f_t  (R^43)
                    |
              ┌─────┴─────┐
              │  LayerNorm │
              │  Linear    │  43  -> 256
              │  GELU      │
              │  Dropout   │
              ├────────────┤
              │  Linear    │  256 -> 256
              │  GELU      │
              │  Dropout   │
              ├────────────┤
              │  Linear    │  256 -> M·D  (reshape to M x D)
              └─────┬──────┘
                    |
           ┌────────┴────────┐
           │ Vector Quantizer│  M continuous vectors  ->  M code indices
           └────────┬────────┘
                    |
                z  (Z^M)   discrete code sequence

Where:
    M  = number of code positions (sequence length, default 8)
    D  = codebook embedding dimension (default 32)
    K  = codebook size (number of codes, default 64)

The VQ layer uses straight-through estimation (STE) so gradients flow
through the discrete bottleneck during training.

Outputs  (in ./output/)
-----------------------
hypernetwork_state.pt        – trained model weights
code_indices.npy             – (T, M)  discrete codes for every window
hypernetwork_summary.txt     – architecture + training log
"""

import os, sys, pickle, time, warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
sys.path.insert(0, BASE_DIR)

from vq_codebook import VQCodebook          # Step 4 standalone module

# =====================================================================
#  Hyper-parameters
# =====================================================================
FINGERPRINT_DIM = 43          # d_in   (from Step 2)
HIDDEN_DIM      = 384         # MLP hidden width  (↑ from 256 for richer mapping)
NUM_CODES_M     = 16          # M – code sequence length (↑ from 8; enables rank-2 recon)
CODE_EMB_DIM    = 48          # D – embedding dim per code position  (↑ from 32)
CODEBOOK_SIZE_K = 64          # K – number of discrete codes  (6 bits/index → 16×6=12 B)
COMMITMENT_COST = 0.25        # beta  for VQ commitment loss
DROPOUT         = 0.1

# Training
EPOCHS     = 400              # ↑ from 300 to compensate for larger model
BATCH_SIZE = 32
LR         = 3e-4
WEIGHT_DECAY = 1e-5
DEAD_CODE_THRESHOLD = 2       # revive codes used fewer times than this
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================================
#  MLP Hypernetwork  H(f_t) -> z
#  (uses VQCodebook from vq_codebook.py – Step 4)
# =====================================================================

class Hypernetwork(nn.Module):
    """
    MLP that maps a fingerprint f_t (R^d_in) to a discrete code sequence
    z (Z^M) via a vector-quantized bottleneck.

    The full forward pass:
        f_t  ->  MLP  ->  (B, M, D)  ->  VQ  ->  z (B, M)  indices
    """

    def __init__(self, d_in: int = FINGERPRINT_DIM,
                 hidden: int = HIDDEN_DIM,
                 M: int = NUM_CODES_M,
                 D: int = CODE_EMB_DIM,
                 K: int = CODEBOOK_SIZE_K,
                 commitment: float = COMMITMENT_COST,
                 dropout: float = DROPOUT):
        super().__init__()
        self.M = M
        self.D = D

        self.encoder = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, M * D),         # project to M code slots
        )

        self.vq = VQCodebook(
            num_codes=K, code_dim=D,
            commitment=commitment,
            dead_threshold=DEAD_CODE_THRESHOLD,
        )

        # Decoder: reconstruct f_t from quantized codes  (for training signal)
        self.decoder = nn.Sequential(
            nn.Linear(M * D, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_in),
        )

    def forward(self, f_t: torch.Tensor):
        """
        f_t : (B, d_in)
        Returns:
            f_hat   : (B, d_in)   reconstructed fingerprint
            z_q     : (B, M, D)   quantised continuous codes
            indices : (B, M)      discrete code indices  <-- this is z = H(f_t)
            vq_loss : scalar
        """
        B = f_t.size(0)
        z_e = self.encoder(f_t).view(B, self.M, self.D)    # (B, M, D)
        z_q, indices, vq_loss, diag = self.vq(z_e)         # quantise
        f_hat = self.decoder(z_q.view(B, -1))               # reconstruct
        return f_hat, z_q, indices, vq_loss, diag

    @torch.no_grad()
    def encode(self, f_t: torch.Tensor):
        """Inference-only: return discrete code indices z = H(f_t)."""
        B = f_t.size(0)
        z_e = self.encoder(f_t).view(B, self.M, self.D)
        _, indices, _, _ = self.vq(z_e)
        return indices


# =====================================================================
#  Training loop  (self-supervised: reconstruction of f_t)
# =====================================================================

def train_hypernetwork(model: Hypernetwork, X: np.ndarray,
                       epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
                       lr: float = LR, device: str = DEVICE):
    """Train on fingerprint reconstruction + VQ loss."""

    model.to(device)
    X_t = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         drop_last=False)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    log = []
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_recon = 0.0
        epoch_vq    = 0.0
        n_batches   = 0

        for (batch,) in loader:
            batch = batch.to(device)
            f_hat, z_q, indices, vq_loss, diag = model(batch)

            recon_loss = F.mse_loss(f_hat, batch)
            loss = recon_loss + vq_loss

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            epoch_recon += recon_loss.item()
            epoch_vq    += vq_loss.item()
            n_batches   += 1

        scheduler.step()
        avg_recon = epoch_recon / n_batches
        avg_vq    = epoch_vq / n_batches

        log.append({"epoch": epoch, "recon": avg_recon, "vq": avg_vq,
                     "total": avg_recon + avg_vq})

        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            cb_stats = model.vq.get_usage_stats()
            print(f"  Epoch {epoch:>4d}/{epochs}  "
                  f"recon={avg_recon:.6f}  vq={avg_vq:.6f}  "
                  f"total={avg_recon + avg_vq:.6f}  "
                  f"perp={cb_stats['perplexity']:.1f}  "
                  f"util={cb_stats['active_codes']}/{cb_stats['total_codes']}")

    return log


# =====================================================================
#  Main
# =====================================================================

if __name__ == "__main__":
    t0 = time.time()

    # ── Load fingerprints from Step 2 ─────────────────────────────────
    fp_path = os.path.join(OUTPUT_DIR, "fingerprints.npy")
    print(f"[INFO] Loading fingerprints from {fp_path}")
    fingerprints = np.load(fp_path).astype(np.float32)
    fingerprints = np.nan_to_num(fingerprints, nan=0.0, posinf=0.0, neginf=0.0)
    T, d_in = fingerprints.shape
    print(f"[INFO] {T} fingerprints, dim={d_in}")

    # Standardise (zero-mean, unit-var) for stable training
    fp_mean = fingerprints.mean(axis=0)
    fp_std  = fingerprints.std(axis=0)
    fp_std[fp_std < 1e-8] = 1.0
    X_norm = (fingerprints - fp_mean) / fp_std

    # ── Build model ───────────────────────────────────────────────────
    model = Hypernetwork(d_in=d_in)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Hypernetwork  params={n_params:,}")
    print(f"       Input  :  f_t in R^{d_in}")
    print(f"       Output :  z in Z^{NUM_CODES_M}  "
          f"(codebook K={CODEBOOK_SIZE_K}, D={CODE_EMB_DIM})")
    print(model)
    print(f"       VQ codebook: {model.vq}")

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\n[INFO] Training for {EPOCHS} epochs on {DEVICE} …")
    log = train_hypernetwork(model, X_norm)

    # ── Encode all fingerprints -> discrete code indices ──────────────
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_norm, dtype=torch.float32).to(DEVICE)
        code_indices = model.encode(X_t).cpu().numpy()       # (T, M)

    print(f"\n[INFO] Code indices shape : {code_indices.shape}")
    print(f"[INFO] Unique codes used  : {len(np.unique(code_indices))}/{CODEBOOK_SIZE_K}")
    print(f"[INFO] Sample z[0]        : {code_indices[0]}")

    # ── Save artefacts ────────────────────────────────────────────────
    state_path = os.path.join(OUTPUT_DIR, "hypernetwork_state.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "fp_mean": fp_mean,
        "fp_std": fp_std,
        "config": {
            "d_in": d_in,
            "hidden": HIDDEN_DIM,
            "M": NUM_CODES_M,
            "D": CODE_EMB_DIM,
            "K": CODEBOOK_SIZE_K,
            "commitment": COMMITMENT_COST,
            "dropout": DROPOUT,
        },
    }, state_path)
    print(f"[INFO] Model saved -> {state_path}")

    idx_path = os.path.join(OUTPUT_DIR, "code_indices.npy")
    np.save(idx_path, code_indices)
    print(f"[INFO] Code indices saved -> {idx_path}")

    # ── Summary text file ─────────────────────────────────────────────
    summary_path = os.path.join(OUTPUT_DIR, "hypernetwork_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  Server-Side Hypernetwork  H(f_t) -> z\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Fingerprint dim  (d_in)  : {d_in}\n")
        f.write(f"Hidden dim               : {HIDDEN_DIM}\n")
        f.write(f"Code sequence length (M) : {NUM_CODES_M}\n")
        f.write(f"Code embedding dim   (D) : {CODE_EMB_DIM}\n")
        f.write(f"Codebook size        (K) : {CODEBOOK_SIZE_K}\n")
        f.write(f"Commitment cost (beta)   : {COMMITMENT_COST}\n")
        f.write(f"Total parameters         : {n_params:,}\n")
        f.write(f"Epochs                   : {EPOCHS}\n")
        f.write(f"Device                   : {DEVICE}\n\n")
        f.write("Architecture:\n")
        f.write(str(model) + "\n\n")
        f.write("Training log (sampled):\n")
        f.write(f"{'Epoch':>6s}  {'Recon':>10s}  {'VQ':>10s}  {'Total':>10s}\n")
        for entry in log:
            if entry["epoch"] % 20 == 0 or entry["epoch"] == 1:
                f.write(f"{entry['epoch']:>6d}  {entry['recon']:>10.6f}  "
                        f"{entry['vq']:>10.6f}  {entry['total']:>10.6f}\n")
        f.write(f"\nFinal loss: {log[-1]['total']:.6f}\n")
        f.write(f"Unique codes used: {len(np.unique(code_indices))}/{CODEBOOK_SIZE_K}\n")
        f.write(f"Code indices shape: {code_indices.shape}\n")
    print(f"[INFO] Summary -> {summary_path}")

    # ── Console summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Hypernetwork  H(f_t) -> z    Summary")
    print(f"{'='*60}")
    print(f"  Input          : f_t in R^{d_in}  (spectral-statistical fingerprint)")
    print(f"  Output         : z in Z^{NUM_CODES_M}  (discrete code indices)")
    print(f"  Codebook       : K={CODEBOOK_SIZE_K} codes, D={CODE_EMB_DIM} dims each")
    print(f"  Parameters     : {n_params:,}")
    print(f"  Final loss     : {log[-1]['total']:.6f}")
    print(f"  Codes utilised : {len(np.unique(code_indices))}/{CODEBOOK_SIZE_K}")
    print(f"  Wall time      : {time.time()-t0:.1f}s")
    print(f"{'='*60}")
    print("[INFO] Done.")
