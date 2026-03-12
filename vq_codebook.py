"""
Step 4 – Vector-Quantized Codebook  C = {e_1, ..., e_K}
=========================================================
A standalone, reusable VQ codebook module that serves as the dictionary
of neural-network weight fragments for the hypernetwork.

Key components
--------------
1. **Learnable embedding matrix** (codebook)
      C ∈ R^{K x D}   where K = codebook size, D = code dimension

2. **Routing mechanism** with Straight-Through Estimator (STE)
      Forward:  z_q = embed[argmin_k ||z_e - e_k||]    (non-differentiable)
      Backward: grad flows as if z_q ≡ z_e              (STE bypass)

3. **EMA codebook updates** with Laplace smoothing
      Prevents codebook collapse by updating embeddings as exponential
      moving averages of assigned encoder outputs.

4. **Anti-collapse safeguards**
      - K-means initialisation on first forward pass
      - Dead-code revival (replace unused codes with perturbed encoder outputs)
      - Usage tracking & diagnostics per forward call

This module is imported by hypernetwork.py and can be used independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQCodebook(nn.Module):
    """
    Vector-Quantized Codebook  C = {e_1, ..., e_K}

    Parameters
    ----------
    num_codes : int
        K – number of codebook entries.
    code_dim : int
        D – dimensionality of each code embedding.
    commitment : float
        β – weight on the commitment loss  β·||z_e - sg[z_q]||².
    ema_decay : float
        γ – decay rate for the exponential moving average (0.99 typical).
    eps : float
        ε – Laplace-smoothing term for EMA cluster sizes.
    dead_threshold : float
        Codes with EMA usage below this are revived each forward pass.
    kmeans_iters : int
        Number of k-means iterations for lazy codebook initialisation.
    """

    def __init__(
        self,
        num_codes: int = 64,
        code_dim: int = 32,
        commitment: float = 0.25,
        ema_decay: float = 0.99,
        eps: float = 1e-5,
        dead_threshold: float = 2.0,
        kmeans_iters: int = 10,
    ):
        super().__init__()
        self.K = num_codes
        self.D = code_dim
        self.commitment = commitment
        self.ema_decay = ema_decay
        self.eps = eps
        self.dead_threshold = dead_threshold
        self.kmeans_iters = kmeans_iters
        self._initialized = False

        # ── Codebook embedding matrix  C ∈ R^{K x D} ─────────────────
        # Stored as buffer (updated via EMA, not gradient descent)
        embed = torch.randn(num_codes, code_dim)
        self.register_buffer("embed", embed)

        # EMA accumulators
        self.register_buffer("cluster_size", torch.ones(num_codes))
        self.register_buffer("embed_avg", embed.clone())

        # Diagnostics (populated every forward)
        self.register_buffer("_last_perplexity", torch.tensor(0.0))
        self.register_buffer("_last_utilization", torch.tensor(0.0))

    # -----------------------------------------------------------------
    #  K-means initialisation  (called once, on first training forward)
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _kmeans_init(self, flat: torch.Tensor):
        """Initialise codebook via mini k-means on encoder outputs."""
        N, D = flat.shape
        K = min(self.K, N)  # can't have more centroids than data points

        # Random subset as initial centroids
        perm = torch.randperm(N, device=flat.device)[:K]
        centroids = flat[perm].clone()

        for _ in range(self.kmeans_iters):
            # Assign each point to nearest centroid
            dist = torch.cdist(flat, centroids, p=2)  # (N, K)
            assign = dist.argmin(dim=1)

            # Update centroids
            for k in range(K):
                mask = assign == k
                if mask.any():
                    centroids[k] = flat[mask].mean(dim=0)

        self.embed.data[:K] = centroids
        self.embed_avg.data[:K] = centroids.clone()
        self.cluster_size.data.fill_(1.0)
        self._initialized = True

    # -----------------------------------------------------------------
    #  Dead-code revival
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _revive_dead_codes(self, flat: torch.Tensor):
        """
        Replace codebook entries with EMA cluster size below `dead_threshold`
        with randomly sampled encoder outputs + small Gaussian noise.
        Returns number of codes revived.
        """
        dead_mask = self.cluster_size < self.dead_threshold
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return 0

        # Sample replacements from current encoder outputs
        idx = torch.randint(0, flat.size(0), (n_dead,), device=flat.device)
        noise = torch.randn_like(flat[idx]) * 0.01
        self.embed.data[dead_mask] = flat[idx] + noise
        self.embed_avg.data[dead_mask] = (flat[idx] + noise).clone()
        self.cluster_size.data[dead_mask] = 1.0
        return n_dead

    # -----------------------------------------------------------------
    #  Core forward:  routing  +  STE  +  EMA
    # -----------------------------------------------------------------

    def forward(self, z_e: torch.Tensor):
        """
        Routing mechanism:  z_e -> z_q  with discrete index selection.

        Parameters
        ----------
        z_e : Tensor (B, M, D)
            Continuous encoder outputs (one per code position).

        Returns
        -------
        z_q_st : Tensor (B, M, D)
            Quantised codes with STE gradient bypass.
        indices : LongTensor (B, M)
            Discrete code indices  (the "z" in  z = H(f_t)).
        vq_loss : Tensor (scalar)
            Combined codebook + commitment loss.
        diagnostics : dict
            Per-call health metrics (perplexity, utilization, etc.).
        """
        B, M, D = z_e.shape
        flat = z_e.reshape(-1, D)                           # (N, D)  N = B*M

        # ── Lazy k-means initialisation ──────────────────────────────
        if self.training and not self._initialized:
            self._kmeans_init(flat)

        # ── 1. Routing: nearest-neighbour lookup ─────────────────────
        #    index_k = argmin_k  ||z_e - e_k||²
        #    This is NON-DIFFERENTIABLE – handled by STE below.
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)            # ||z_e||²
            - 2.0 * flat @ self.embed.t()                   # -2·z_e·C^T
            + self.embed.pow(2).sum(dim=1, keepdim=True).t() # ||e_k||²
        )                                                    # (N, K)

        indices = dist.argmin(dim=1)                         # (N,)
        z_q = self.embed[indices].view(B, M, D)              # quantised

        # ── 2. EMA codebook update (training only) ───────────────────
        #    Running averages of cluster sizes and embedding sums.
        #    Uses Laplace smoothing to prevent divide-by-zero.
        if self.training:
            one_hot = F.one_hot(indices, self.K).float()     # (N, K)

            # Update cluster sizes:  n_k ← γ·n_k + (1-γ)·|{i: q(i)=k}|
            self.cluster_size.data.mul_(self.ema_decay).add_(
                one_hot.sum(dim=0), alpha=1.0 - self.ema_decay
            )

            # Update embedding sums:  s_k ← γ·s_k + (1-γ)·Σ_{i:q(i)=k} z_e^i
            embed_sum = one_hot.t() @ flat.detach()          # (K, D)
            self.embed_avg.data.mul_(self.ema_decay).add_(
                embed_sum, alpha=1.0 - self.ema_decay
            )

            # Laplace-smoothed normalisation → new embeddings
            #   e_k = s_k / n_k   (with smoothing to avoid collapse)
            n = self.cluster_size.sum()
            smoothed_size = (
                (self.cluster_size + self.eps)
                / (n + self.K * self.eps) * n
            )
            self.embed.data.copy_(
                self.embed_avg / smoothed_size.unsqueeze(1)
            )

            # ── Dead-code revival ────────────────────────────────────
            self._revive_dead_codes(flat.detach())

        # ── 3. Loss computation ──────────────────────────────────────
        #
        # codebook_loss  =  ||z_q - sg[z_e]||²
        #     Pulls codebook entries toward encoder outputs.
        #     (Not needed when using EMA, but kept for hybrid stability.)
        #
        # commitment_loss = ||z_e - sg[z_q]||²
        #     Prevents encoder from drifting far from codebook.
        #
        codebook_loss   = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.commitment * commitment_loss

        # ── 4. Straight-Through Estimator (STE) ─────────────────────
        #
        # The argmin routing is non-differentiable.  We use the STE trick:
        #
        #   Forward:   z_q_st  =  z_q                (use discrete codes)
        #   Backward:  ∂L/∂z_e =  ∂L/∂z_q_st        (copy gradients through)
        #
        # Implemented as:  z_q_st = z_e + (z_q - z_e).detach()
        # so that the VALUE is z_q but the GRADIENT acts on z_e.
        #
        z_q_st = z_e + (z_q - z_e).detach()

        # ── 5. Diagnostics ───────────────────────────────────────────
        with torch.no_grad():
            # Codebook perplexity: exp(H(p)) where p is the usage distribution
            avg_probs = one_hot.mean(dim=0) if self.training else \
                F.one_hot(indices, self.K).float().mean(dim=0)
            avg_probs = avg_probs + 1e-10
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs)))

            # Utilization: fraction of codes used in this batch
            utilization = (avg_probs > 1e-8).float().sum() / self.K

            self._last_perplexity.fill_(perplexity.item())
            self._last_utilization.fill_(utilization.item())

        diagnostics = {
            "perplexity":       perplexity.item(),
            "utilization":      utilization.item(),
            "codebook_loss":    codebook_loss.item(),
            "commitment_loss":  commitment_loss.item(),
            "vq_loss":          vq_loss.item(),
            "mean_code_dist":   dist.min(dim=1).values.mean().item(),
            "cluster_size_std": self.cluster_size.std().item(),
        }

        return z_q_st, indices.view(B, M), vq_loss, diagnostics

    # -----------------------------------------------------------------
    #  Convenience methods
    # -----------------------------------------------------------------

    @torch.no_grad()
    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Given discrete code indices, return the corresponding embeddings.
            indices : (*, M) -> embeddings : (*, M, D)
        """
        return self.embed[indices]

    @torch.no_grad()
    def get_codebook(self) -> torch.Tensor:
        """Return the full codebook matrix C ∈ R^{K x D}."""
        return self.embed.clone()

    @torch.no_grad()
    def get_usage_stats(self) -> dict:
        """Return current EMA cluster sizes and derived statistics."""
        cs = self.cluster_size.clone()
        # Use a minimal threshold for 'active': any code with EMA > eps
        active_threshold = self.eps * 10
        return {
            "cluster_sizes": cs,
            "active_codes": int((cs > active_threshold).sum().item()),
            "total_codes": self.K,
            "utilization": float((cs > active_threshold).sum().item()) / self.K,
            "perplexity": self._last_perplexity.item(),
            "mean_cluster_size": cs.mean().item(),
            "std_cluster_size": cs.std().item(),
        }

    def extra_repr(self) -> str:
        return (
            f"K={self.K}, D={self.D}, commitment={self.commitment}, "
            f"ema_decay={self.ema_decay}, dead_threshold={self.dead_threshold}"
        )
