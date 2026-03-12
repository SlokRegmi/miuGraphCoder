"""
Step 1 – Data Acquisition & Temporal Graph Construction  (vectorised)
=====================================================================
Reads the BoT-IoT dataset and builds a temporal graph sequence
    G_t = (V_t, E_t)
using non-overlapping 5-minute sliding windows.

* V_t  – unique IP endpoints active in window t
* E_t  – directed communication flows (src_ip -> dst_ip) in window t

Every operation is fully vectorised with pandas / numpy – no Python-level
row iteration – so it scales to the full 73 M-row dataset.

Outputs  (in ./output/)
-----------------------
temporal_graphs.pkl      – list of per-window graph dicts
per_window_statistics.csv
temporal_statistics.csv  – paper-ready summary table
temporal_statistics.tex  – LaTeX table for the paper
"""

import os, glob, pickle, warnings, time
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
WINDOW_SEC  = 5 * 60          # 5-minute windows
CHUNK_SIZE  = 1_000_000       # rows per read
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Flow-level features to keep as edge attributes
FLOW_FEATS = [
    "pkts", "bytes", "dur",
    "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "srate", "drate",
    "mean", "stddev",
]

csv_files = sorted(
    glob.glob(os.path.join(DATASET_DIR, "data_[0-9]*.csv")),
    key=lambda f: int(os.path.basename(f).split("_")[1].split(".")[0]),
)
print(f"[INFO] Found {len(csv_files)} CSV shards")

# ── 1. Global time range (fast) ──────────────────────────────────────
print("[INFO] Pass 1 / 2 – scanning time range ...")
t0 = time.time()
g_min, g_max = np.inf, -np.inf
for fp in csv_files:
    for ch in pd.read_csv(fp, usecols=["stime"], chunksize=CHUNK_SIZE):
        lo, hi = ch["stime"].min(), ch["stime"].max()
        if lo < g_min: g_min = lo
        if hi > g_max: g_max = hi
n_windows = int(np.ceil((g_max - g_min) / WINDOW_SEC))
print(f"       {pd.Timestamp(g_min, unit='s')} -> {pd.Timestamp(g_max, unit='s')}")
print(f"       {(g_max-g_min)/3600:.1f} h  =>  {n_windows} windows   ({time.time()-t0:.1f}s)")

# ── 2. Main pass – vectorised aggregation ────────────────────────────
#
# Strategy:  for each chunk we
#   a) assign window id  w = floor((stime - g_min) / WINDOW_SEC)
#   b) groupby (w, saddr, daddr) and aggregate flow features
#   c) collect the chunk-level aggregates into a list
#
# After all files are read we do the final reduction and build graphs.
# ──────────────────────────────────────────────────────────────────────

print("[INFO] Pass 2 / 2 – aggregating edges per window ...")
t1 = time.time()

# Accumulator: list of (small) DataFrames – we concat at the end
acc_parts = []
total_flows = 0

needed_cols = ["stime", "saddr", "daddr", "attack"] + FLOW_FEATS

for fi, fp in enumerate(csv_files):
    for ch in pd.read_csv(fp, chunksize=CHUNK_SIZE):
        ch.columns = [c.strip() for c in ch.columns]

        # Keep only needed columns that exist
        avail = [c for c in needed_cols if c in ch.columns]
        ch = ch[avail]
        ch = ch.dropna(subset=["stime", "saddr", "daddr"])

        total_flows += len(ch)

        # Numeric coercion
        for c in FLOW_FEATS:
            if c in ch.columns:
                ch[c] = pd.to_numeric(ch[c], errors="coerce").fillna(0.0)

        ch["attack"] = pd.to_numeric(ch["attack"], errors="coerce").fillna(0).astype(int)

        # Window assignment
        ch["w"] = ((ch["stime"] - g_min) / WINDOW_SEC).astype(int).clip(upper=n_windows - 1)

        # Group by (window, src, dst) and aggregate
        agg_dict = {c: "mean" for c in FLOW_FEATS if c in ch.columns}
        agg_dict["attack"] = "sum"           # number of attack flows
        agg_dict["stime"]  = "count"         # number of flows on that edge

        grouped = ch.groupby(["w", "saddr", "daddr"], sort=False).agg(agg_dict)
        grouped.rename(columns={"stime": "flow_count"}, inplace=True)
        grouped.reset_index(inplace=True)
        acc_parts.append(grouped)

    if (fi + 1) % 10 == 0 or fi + 1 == len(csv_files):
        print(f"  ... file {fi+1}/{len(csv_files)}  "
              f"({total_flows:,} flows,  {time.time()-t1:.0f}s)")

# ── 3. Final reduction ───────────────────────────────────────────────
print("[INFO] Merging partial aggregates ...")
edges_df = pd.concat(acc_parts, ignore_index=True)
del acc_parts  # free memory

# Re-aggregate across chunk boundaries
feat_cols = [c for c in FLOW_FEATS if c in edges_df.columns]
agg2 = {c: "mean" for c in feat_cols}
agg2["attack"]     = "sum"
agg2["flow_count"] = "sum"

edges_df = edges_df.groupby(["w", "saddr", "daddr"], sort=False).agg(agg2).reset_index()
edges_df["flow_count"] = edges_df["flow_count"].astype(int)
edges_df["attack"]     = edges_df["attack"].astype(int)

print(f"[INFO] Total unique (window, src, dst) edges: {len(edges_df):,}")

# ── 4. Build temporal graph objects ──────────────────────────────────
print("[INFO] Building graph objects ...")

feat_dim = len(feat_cols)
NODE_FEAT_DIM = 3 + feat_dim        # in_deg, out_deg, total_deg + flow agg

temporal_graphs = []
stats_rows      = []

for w, grp in edges_df.groupby("w", sort=True):
    nodes = sorted(set(grp["saddr"]).union(grp["daddr"]))
    ip2idx = {ip: i for i, ip in enumerate(nodes)}
    n_nodes = len(nodes)
    n_edges = len(grp)

    src_idx = grp["saddr"].map(ip2idx).values.astype(np.int64)
    dst_idx = grp["daddr"].map(ip2idx).values.astype(np.int64)
    edge_index = np.stack([src_idx, dst_idx])  # (2, E)

    edge_attr = grp[feat_cols].values.astype(np.float32)  # (E, d_e)
    edge_y = (grp["attack"] > 0).astype(np.int64).values   # 1 if any attack flow

    # Node features --------------------------------------------------
    in_deg  = np.zeros(n_nodes, dtype=np.float32)
    out_deg = np.zeros(n_nodes, dtype=np.float32)
    node_flow = np.zeros((n_nodes, feat_dim), dtype=np.float32)
    node_cnt  = np.zeros(n_nodes, dtype=np.float32)

    np.add.at(out_deg, src_idx, 1)
    np.add.at(in_deg,  dst_idx, 1)
    np.add.at(node_flow, src_idx, edge_attr)
    np.add.at(node_flow, dst_idx, edge_attr)
    np.add.at(node_cnt,  src_idx, 1)
    np.add.at(node_cnt,  dst_idx, 1)

    node_cnt[node_cnt == 0] = 1
    node_flow /= node_cnt[:, None]
    total_deg = in_deg + out_deg
    node_feat = np.column_stack([in_deg, out_deg, total_deg, node_flow])

    n_flows = int(grp["flow_count"].sum())
    t_start = g_min + int(w) * WINDOW_SEC
    t_end   = t_start + WINDOW_SEC

    temporal_graphs.append({
        "window_id":  int(w),
        "t_start":    t_start,
        "t_end":      t_end,
        "num_nodes":  n_nodes,
        "num_edges":  n_edges,
        "node_list":  nodes,
        "ip2idx":     ip2idx,
        "node_feat":  node_feat,
        "edge_index": edge_index,
        "edge_attr":  edge_attr,
        "edge_y":     edge_y,
        "num_flows":  n_flows,
    })

    stats_rows.append({
        "window_id":     int(w),
        "t_start":       pd.Timestamp(t_start, unit="s"),
        "t_end":         pd.Timestamp(t_end,   unit="s"),
        "num_nodes":     n_nodes,
        "num_edges":     n_edges,
        "num_flows":     n_flows,
        "node_feat_dim": node_feat.shape[1],
        "edge_feat_dim": edge_attr.shape[1],
        "attack_edges":  int(edge_y.sum()),
        "normal_edges":  int((edge_y == 0).sum()),
    })

# ── 5. Persist ────────────────────────────────────────────────────────
pkl_path = os.path.join(OUTPUT_DIR, "temporal_graphs.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(temporal_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"[INFO] Saved {len(temporal_graphs)} temporal graphs -> {pkl_path}")

# ── 6. Statistics table ──────────────────────────────────────────────
stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(os.path.join(OUTPUT_DIR, "per_window_statistics.csv"), index=False)

summary = {
    "Total time windows ($|\\mathcal{T}|$)":  len(stats_df),
    "Window size ($\\Delta t$)":                f"{WINDOW_SEC // 60} min",
    "Total flows processed":                   f"{total_flows:,}",
    "Avg. active nodes / window":              f"{stats_df['num_nodes'].mean():.2f}",
    "Std. active nodes / window":              f"{stats_df['num_nodes'].std():.2f}",
    "Min / Max active nodes":                  f"{stats_df['num_nodes'].min()} / {stats_df['num_nodes'].max()}",
    "Avg. edges / window":                     f"{stats_df['num_edges'].mean():.2f}",
    "Std. edges / window":                     f"{stats_df['num_edges'].std():.2f}",
    "Min / Max edges":                         f"{stats_df['num_edges'].min()} / {stats_df['num_edges'].max()}",
    "Avg. flows / window":                     f"{stats_df['num_flows'].mean():.2f}",
    "Node feature dimension ($d_v$)":          int(stats_df["node_feat_dim"].iloc[0]),
    "Edge feature dimension ($d_e$)":          int(stats_df["edge_feat_dim"].iloc[0]),
    "Avg. attack edges / window":              f"{stats_df['attack_edges'].mean():.2f}",
    "Avg. normal edges / window":              f"{stats_df['normal_edges'].mean():.2f}",
    "Time span":                               f"{pd.Timestamp(g_min, unit='s')} -- "
                                                f"{pd.Timestamp(g_max, unit='s')}",
}

summary_df = pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])
summary_df.to_csv(os.path.join(OUTPUT_DIR, "temporal_statistics.csv"), index=False)

# Pretty-print
print("\n" + "=" * 65)
print("   Table 1: Dataset Temporal Statistics  (BoT-IoT)")
print("=" * 65)
for _, row in summary_df.iterrows():
    print(f"  {row['Metric']:45s}  {row['Value']}")
print("=" * 65)

# LaTeX version
tex_path = os.path.join(OUTPUT_DIR, "temporal_statistics.tex")
with open(tex_path, "w") as f:
    f.write("\\begin{table}[ht]\n\\centering\n")
    f.write("\\caption{Dataset Temporal Statistics (BoT-IoT)}\n")
    f.write("\\label{tab:temporal_stats}\n")
    f.write("\\begin{tabular}{l r}\n\\toprule\n")
    f.write("Metric & Value \\\\\n\\midrule\n")
    for _, r in summary_df.iterrows():
        f.write(f"{r['Metric']} & {r['Value']} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

print(f"\n[INFO] LaTeX table -> {tex_path}")
print(f"[INFO] Total wall time: {time.time()-t0:.0f}s")
print("[INFO] Done.")
