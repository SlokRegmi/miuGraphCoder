"""
Step 2 – Spectral-Statistical Fingerprinting
=============================================
Takes the temporal graph sequence {G_t} produced in Step 1 and compresses
each snapshot into a compact fingerprint vector  f_t  (< 512 bytes).

Fingerprint composition
-----------------------
A.  **Degree statistics** (8 scalars)
    mean / std / skew / kurtosis  of  in-degree and out-degree

B.  **Centrality statistics** (6 scalars)
    mean / std / max  of  betweenness and closeness centrality

C.  **Edge weight / traffic distribution** (12 scalars)
    For each of the 12 edge flow features:  mean across all edges in G_t

D.  **Node churn** (3 scalars)
    |V_t \\ V_{t-1}|  /  |V_t|          (join ratio)
    |V_{t-1} \\ V_t|  /  |V_{t-1}|    (leave ratio)
    Jaccard(V_t, V_{t-1})

E.  **Spectral profile** (10 scalars)
    Leading 10 eigenvalues of the symmetric normalised Laplacian
    (zero-padded when |V| < 10)

F.  **Graph-level scalars** (4 scalars)
    |V_t|,  |E_t|,  density,  attack_edge_ratio

Total:  8 + 6 + 12 + 3 + 10 + 4  =  **43  float32 values  =  172 bytes  < 512**

Outputs  (in ./output/)
-----------------------
fingerprints.npy           – (T, 43) float32 array
fingerprint_labels.npy     – (T,)    per-window majority label (0/1)
fingerprint_meta.pkl       – feature names + per-window metadata
tsne_fingerprints.png      – t-SNE scatter plot coloured by network state
"""

import os, pickle, struct, time, warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.stats import skew, kurtosis
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PKL_PATH   = os.path.join(OUTPUT_DIR, "temporal_graphs.pkl")

N_EIGEN    = 10          # leading Laplacian eigenvalues to keep
MAX_BYTES  = 512         # hard constraint on serialised fingerprint size

# ── Feature-name registry (for interpretability) ─────────────────────
FEAT_NAMES = (
    # A – degree statistics (8)
    ["in_deg_mean", "in_deg_std", "in_deg_skew", "in_deg_kurt",
     "out_deg_mean", "out_deg_std", "out_deg_skew", "out_deg_kurt"]
    # B – centrality statistics (6)
    + ["betw_mean", "betw_std", "betw_max",
       "close_mean", "close_std", "close_max"]
    # C – edge traffic distribution (12)
    + ["edge_pkts", "edge_bytes", "edge_dur",
       "edge_spkts", "edge_dpkts", "edge_sbytes", "edge_dbytes",
       "edge_rate", "edge_srate", "edge_drate",
       "edge_mean", "edge_stddev"]
    # D – node churn (3)
    + ["churn_join", "churn_leave", "churn_jaccard"]
    # E – spectral profile (10)
    + [f"lambda_{i}" for i in range(N_EIGEN)]
    # F – graph scalars (4)
    + ["num_nodes", "num_edges", "density", "attack_ratio"]
)
FINGERPRINT_DIM = len(FEAT_NAMES)          # 43


# =====================================================================
#  Core fingerprinting function
# =====================================================================

def compute_fingerprint(G_t: dict, prev_node_set: set | None = None) -> np.ndarray:
    """Return a float32 vector of length FINGERPRINT_DIM for one graph snapshot."""

    n_nodes    = G_t["num_nodes"]
    n_edges    = G_t["num_edges"]
    edge_index = G_t["edge_index"]          # (2, E)
    edge_attr  = G_t["edge_attr"]           # (E, 12)
    edge_y     = G_t["edge_y"]              # (E,)
    node_list  = G_t["node_list"]

    fp = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
    pos = 0   # cursor into fp

    # ----- A. Degree statistics (8) -----------------------------------
    in_deg  = np.zeros(n_nodes, dtype=np.float32)
    out_deg = np.zeros(n_nodes, dtype=np.float32)
    if n_edges > 0:
        np.add.at(out_deg, edge_index[0], 1)
        np.add.at(in_deg,  edge_index[1], 1)

    for deg_arr in [in_deg, out_deg]:
        fp[pos]     = deg_arr.mean()
        fp[pos + 1] = deg_arr.std()
        fp[pos + 2] = float(skew(deg_arr))   if n_nodes > 2 else 0.0
        fp[pos + 3] = float(kurtosis(deg_arr)) if n_nodes > 2 else 0.0
        pos += 4

    # ----- B. Centrality statistics (6) -------------------------------
    G_nx = nx.DiGraph()
    G_nx.add_nodes_from(range(n_nodes))
    if n_edges > 0:
        edges_with_weight = []
        total_bytes = edge_attr[:, 1]   # "bytes" is column 1
        for i in range(n_edges):
            w = float(total_bytes[i]) if total_bytes[i] > 0 else 1.0
            edges_with_weight.append((int(edge_index[0, i]),
                                      int(edge_index[1, i]),
                                      w))
        G_nx.add_weighted_edges_from(edges_with_weight)

    # Betweenness (on undirected view for stability)
    G_und = G_nx.to_undirected()
    betw = np.array(list(nx.betweenness_centrality(G_und).values()), dtype=np.float32)
    fp[pos]     = betw.mean();  fp[pos+1] = betw.std();  fp[pos+2] = betw.max()
    pos += 3

    # Closeness
    close = np.array(list(nx.closeness_centrality(G_und).values()), dtype=np.float32)
    fp[pos]     = close.mean(); fp[pos+1] = close.std(); fp[pos+2] = close.max()
    pos += 3

    # ----- C. Edge weight / traffic distribution (12) -----------------
    if n_edges > 0:
        fp[pos:pos+edge_attr.shape[1]] = edge_attr.mean(axis=0)
    pos += 12

    # ----- D. Node churn (3) ------------------------------------------
    curr_set = set(node_list)
    if prev_node_set is not None and len(prev_node_set) > 0:
        joined  = curr_set - prev_node_set
        left    = prev_node_set - curr_set
        union   = curr_set | prev_node_set
        fp[pos]     = len(joined)  / max(len(curr_set), 1)
        fp[pos + 1] = len(left)    / max(len(prev_node_set), 1)
        fp[pos + 2] = len(curr_set & prev_node_set) / max(len(union), 1)
    else:
        fp[pos] = 1.0      # first window: all nodes are "new"
        fp[pos + 1] = 0.0
        fp[pos + 2] = 0.0
    pos += 3

    # ----- E. Spectral profile – leading Laplacian eigenvalues (10) ---
    if n_nodes >= 2 and n_edges > 0:
        # Build symmetric adjacency (undirected)
        row = np.concatenate([edge_index[0], edge_index[1]])
        col = np.concatenate([edge_index[1], edge_index[0]])
        data = np.ones(len(row), dtype=np.float32)
        A = sp.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
        # Remove duplicate entries (sum → clip to 1 for unweighted)
        A.data = np.minimum(A.data, 1.0)

        deg_vec = np.array(A.sum(axis=1)).flatten()
        deg_vec[deg_vec == 0] = 1.0
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg_vec))
        L_norm = sp.eye(n_nodes) - D_inv_sqrt @ A @ D_inv_sqrt

        k = min(N_EIGEN, n_nodes - 1)
        try:
            eigvals = eigsh(L_norm.astype(np.float64), k=k,
                            which="SM", return_eigenvectors=False)
            eigvals = np.sort(np.real(eigvals))
        except Exception:
            eigvals = np.zeros(k, dtype=np.float64)

        fp[pos:pos+k] = eigvals[:k].astype(np.float32)
    pos += N_EIGEN

    # ----- F. Graph-level scalars (4) ---------------------------------
    max_possible_edges = n_nodes * (n_nodes - 1) if n_nodes > 1 else 1
    density = n_edges / max_possible_edges
    atk_ratio = edge_y.sum() / max(n_edges, 1)

    fp[pos]     = n_nodes
    fp[pos + 1] = n_edges
    fp[pos + 2] = density
    fp[pos + 3] = atk_ratio
    pos += 4

    return fp


# =====================================================================
#  512-byte constraint check
# =====================================================================

def assert_fingerprint_size(fp: np.ndarray) -> None:
    """Raise AssertionError if the serialised fingerprint exceeds 512 bytes."""
    raw = fp.astype(np.float32).tobytes()
    assert len(raw) <= MAX_BYTES, (
        f"Fingerprint serialises to {len(raw)} bytes, exceeds {MAX_BYTES}-byte limit!"
    )


# =====================================================================
#  Main
# =====================================================================

if __name__ == "__main__":
    t0 = time.time()

    # ── Load temporal graphs from Step 1 ──────────────────────────────
    print(f"[INFO] Loading temporal graphs from {PKL_PATH} …")
    with open(PKL_PATH, "rb") as f:
        temporal_graphs = pickle.load(f)
    T = len(temporal_graphs)
    print(f"[INFO] Loaded {T} graph snapshots")

    # ── Compute fingerprints ──────────────────────────────────────────
    print(f"[INFO] Computing {FINGERPRINT_DIM}-dim fingerprints …")
    fingerprints = np.zeros((T, FINGERPRINT_DIM), dtype=np.float32)
    labels       = np.zeros(T, dtype=np.int64)
    meta         = []

    prev_nodes = None
    for i, G_t in enumerate(temporal_graphs):
        fp = compute_fingerprint(G_t, prev_node_set=prev_nodes)
        assert_fingerprint_size(fp)                       # ← 512-byte gate
        fingerprints[i] = fp
        # Window label: 1 if majority of edges are attack
        atk_edges  = int(G_t["edge_y"].sum())
        norm_edges = int((G_t["edge_y"] == 0).sum())
        labels[i]  = 1 if atk_edges > norm_edges else 0
        prev_nodes = set(G_t["node_list"])
        meta.append({
            "window_id":    G_t["window_id"],
            "t_start":      G_t["t_start"],
            "t_end":        G_t["t_end"],
            "num_nodes":    G_t["num_nodes"],
            "num_edges":    G_t["num_edges"],
            "attack_ratio": float(atk_edges / max(atk_edges + norm_edges, 1)),
        })
        if (i + 1) % 50 == 0 or i + 1 == T:
            print(f"  … {i+1}/{T}")

    # ── Byte-size report ──────────────────────────────────────────────
    raw_bytes = fingerprints[0].tobytes()
    print(f"\n[CHECK] Fingerprint dimension : {FINGERPRINT_DIM}")
    print(f"[CHECK] Serialised size       : {len(raw_bytes)} bytes  "
          f"(limit {MAX_BYTES})")
    print(f"[CHECK] Constraint satisfied  : {len(raw_bytes) <= MAX_BYTES}  ✓")

    # ── Save artefacts ────────────────────────────────────────────────
    np.save(os.path.join(OUTPUT_DIR, "fingerprints.npy"), fingerprints)
    np.save(os.path.join(OUTPUT_DIR, "fingerprint_labels.npy"), labels)
    with open(os.path.join(OUTPUT_DIR, "fingerprint_meta.pkl"), "wb") as f:
        pickle.dump({"feat_names": FEAT_NAMES, "meta": meta}, f)
    print(f"[INFO] Saved fingerprints.npy  shape={fingerprints.shape}")

    # ── t-SNE Visualisation ──────────────────────────────────────────
    print("[INFO] Running t-SNE (perplexity=30) …")
    # Replace any NaN / inf that slipped through (e.g. skew of constant array)
    fingerprints = np.nan_to_num(fingerprints, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(fingerprints)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    perp = min(30, T - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                max_iter=1000, learning_rate="auto", init="pca")
    X_2d = tsne.fit_transform(X_scaled)

    # Colour map: 0 = Normal (blue), 1 = Attack-dominant (red)
    fig, ax = plt.subplots(figsize=(8, 6))
    colours = np.where(labels == 0, "#1f77b4", "#d62728")
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                         c=colours, s=40, alpha=0.75, edgecolors="k",
                         linewidths=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', label='Normal-dominant window',
               markerfacecolor='#1f77b4', markersize=9, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Attack-dominant window',
               markerfacecolor='#d62728', markersize=9, markeredgecolor='k'),
    ]
    ax.legend(handles=legend_elems, loc="best", fontsize=10)

    ax.set_title("t-SNE of Spectral-Statistical Fingerprints  "
                 f"($f_t \\in \\mathbb{{R}}^{{{FINGERPRINT_DIM}}}$, "
                 f"{len(raw_bytes)} bytes)", fontsize=12)
    ax.set_xlabel("t-SNE dim 1", fontsize=11)
    ax.set_ylabel("t-SNE dim 2", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, "tsne_fingerprints.png")
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] t-SNE plot saved → {png_path}")

    # ── Summary ──────────────────────────────────────────────────────
    n_atk = int(labels.sum())
    n_nrm = T - n_atk
    print(f"\n{'='*55}")
    print(f"  Fingerprint Summary")
    print(f"{'='*55}")
    print(f"  Windows          : {T}")
    print(f"  Normal windows   : {n_nrm}  ({100*n_nrm/T:.1f}%)")
    print(f"  Attack windows   : {n_atk}  ({100*n_atk/T:.1f}%)")
    print(f"  Fingerprint dim  : {FINGERPRINT_DIM}")
    print(f"  Serialised bytes : {len(raw_bytes)}  (< {MAX_BYTES})")
    print(f"  Feature groups   :")
    print(f"    A  Degree stats       : 8")
    print(f"    B  Centrality stats   : 6")
    print(f"    C  Traffic distrib.   : 12")
    print(f"    D  Node churn         : 3")
    print(f"    E  Spectral profile   : {N_EIGEN}")
    print(f"    F  Graph scalars      : 4")
    print(f"{'='*55}")
    print(f"[INFO] Total wall time: {time.time()-t0:.1f}s")
    print("[INFO] Done.")
