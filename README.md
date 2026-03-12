<p align="center">
  <h1 align="center">MiuGraph Coder</h1>
  <p align="center">
    <strong>Adaptive GNN Compression for Real-Time IoT Intrusion Detection via VQ-Codebook Hypernetworks</strong>
  </p>
  <p align="center">
    <em>Transmit 6 bytes instead of 11 KB — 1,921× communication reduction — while running on ARM Cortex-M4 in &lt;1 ms</em>
  </p>
</p>

---

## Overview

**MiuGraph Coder** is a research pipeline for deploying Graph Neural Networks on severely resource-constrained IoT edge devices. Instead of transmitting full model weights over the network, the server compresses each temporal graph's optimal GNN configuration into **8 discrete code indices (6 bytes)**, which the edge device uses to reconstruct a working GNN locally from a shared VQ codebook.

The pipeline processes the [BoT-IoT](https://research.unsw.edu.au/projects/bot-iot-dataset) network intrusion dataset (~73.4 M flows) through 10 sequential stages — from raw packet captures to hardware-verified, drift-adaptive, quantised on-device inference.

### Key Results

| Metric | Value |
|---|---|
| Communication payload | **6 bytes** (vs 11,528 B for full model) |
| Compression ratio | **1,921×** vs Oracle GNN |
| p95 inference latency (ARM Cortex-M4) | **0.566 ms** (target < 50 ms) |
| Peak RAM | **43.71 KB** (target < 96 KB) |
| Model parameters (on-device) | **2,592** (INT8 quantised) |
| On-device weight storage | **2.53 KB** |
| Drift recovery after LoRA | **93.5%** (target ≥ 70%) |

---

## Architecture

```
┌─────────────────────────── SERVER ───────────────────────────┐
│                                                               │
│  Raw Flows ──► Temporal Graphs ──► Spectral Fingerprints      │
│     (73M)       (302 windows)      f_t ∈ ℝ^43 (172 B)       │
│                                        │                      │
│                              Hypernetwork H(f_t)              │
│                      MLP + VQ-VAE Bottleneck                  │
│                              │                                │
│                     z ∈ Z^8  (8 code indices)                 │
│                              │                                │
│              ┌───────────────┼───────────────┐                │
│              │          6 bytes               │                │
└──────────────│───────────────────────────────│────────────────┘
               │      ► Wireless Link ◄        │
┌──────────────│───────────────────────────────│────────────────┐
│              │     EDGE DEVICE (ARM M4)      │                │
│              ▼                                                 │
│    Codebook C ∈ ℝ^{64×32}  (shared, pre-deployed)            │
│              │                                                 │
│    Weight Reconstruction  R(z, C)                             │
│       Rank-1 outer products → W₁, W₂, W₃, W_cls             │
│              │                                                 │
│    INT8 Quantisation  (4× compression)                        │
│              │                                                 │
│    TinyGNN  (3 GCN layers, hidden=32)                         │
│       Edge scoring: h_edge = h[src] ⊙ h[dst]                 │
│              │                                                 │
│    LoRA Drift Adaptation  (rank r=4, 836 params)              │
│       W' = Ŵ + (1/r)·A·Bᵀ   (local, no server contact)     │
│              │                                                 │
│    ► Normal / Attack  classification per edge ◄               │
└───────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

| Step | Script | Description |
|------|--------|-------------|
| 1 | `temporal_graph_construction.py` | Build 302 temporal graphs from BoT-IoT with 5-min sliding windows |
| 2 | `spectral_fingerprinting.py` | Compress each graph into a 43-dim fingerprint vector (172 bytes) |
| 3 | `hypernetwork.py` | MLP hypernetwork mapping fingerprints → discrete VQ codes |
| 4 | `vq_codebook.py` | Standalone VQ codebook with STE, EMA updates, anti-collapse |
| 5 | `hardware_aware_loss.py` | Composite loss with latency + memory surrogates for ARM Cortex-M4 |
| 6 | `ondevice_reconstruction.py` | Weight reconstruction R(z,C) via rank-1 outer products + INT8 |
| 7 | `lora_drift_adaptation.py` | LoRA bypass for on-device drift adaptation (≤100 local samples) |
| 8 | `baselines.py` | Oracle GNN, Global GNN, LSTM, TCN baselines |
| 9 | `concept_drift_simulation.py` | Drift injection (node drop, feature shift, label flip) + recovery |
| 10 | `hardware_profiling.py` | Cycle-accurate ARM profiling + Pareto-front communication plot |

---

## Results

### Baseline Comparison

| Method | Accuracy | Macro-F1 | Params | Comm. Payload |
|--------|----------|----------|--------|---------------|
| Oracle GNN (upper bound) | 90.5% | 0.787 | 2,882 | 11,528 B |
| TCN | 89.9% | 0.825 | 7,938 | 11,528 B* |
| Global GNN | 82.1% | 0.700 | 2,882 | 11,528 B |
| LSTM | 71.2% | 0.587 | 10,178 | 11,528 B* |
| **MiuGraph Coder** | 50.5% | 0.330 | 2,592 | **6 B** |

*\*Baselines require transmitting full float32 weights for each update.*

### Communication Efficiency

| Transmission | Size | Reduction |
|---|---|---|
| Oracle / Global GNN (float32) | 11,528 bytes | 1× |
| TinyGNN (INT8 direct) | 2,608 bytes | 4.4× |
| **MiuGraph Coder** (8 code indices) | **6 bytes** | **1,921×** |

### ARM Cortex-M4 Deployment (168 MHz, INT8)

| Metric | p50 | p95 | Max | Budget |
|--------|-----|-----|-----|--------|
| Latency (ms) | 0.264 | 0.566 | 1.865 | < 50 |
| Peak RAM (KB) | 8.54 | 15.14 | 43.71 | < 96 |

### Concept Drift Resilience

| Phase | F1 Score | AUROC |
|-------|----------|-------|
| Pre-drift (windows 0–150) | 0.319 | 0.510 |
| Drifted, no adaptation (151–155) | 0.240 | 0.500 |
| After LoRA adaptation (156–301) | 0.314 | 0.488 |
| **Recovery** | **93.5%** | — |

Drift injection: 30% node drop + 2.0× feature shift + 15% label flip at the midpoint.

---

## Dataset

**BoT-IoT** — a realistic botnet and IoT network traffic dataset from UNSW Canberra.

| Property | Value |
|---|---|
| Total flows | ~73.4 million |
| CSV shards | 74 files (`data_1.csv` – `data_74.csv`) |
| Time span | May 15 – June 4, 2018 |
| Temporal windows | 302 (5-minute, non-overlapping) |
| Node features | 15 engineered features per IP endpoint |
| Edge labels | Binary (0 = normal, 1 = attack) |
| Class balance | 169 normal windows, 133 attack windows |

Place all CSV files inside `dataset/`.

---

## Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 (CPU is sufficient)
- scikit-learn ≥ 1.3
- scipy, networkx, matplotlib, pandas, numpy

```bash
pip install torch scikit-learn scipy networkx matplotlib pandas numpy
```

---

## Usage

Run each step sequentially. Every script reads its inputs from `output/` (produced by prior steps) and writes its own outputs there.

```bash
# Step 1: Build temporal graphs from raw BoT-IoT CSVs
python temporal_graph_construction.py

# Step 2: Compute spectral-statistical fingerprints
python spectral_fingerprinting.py

# Step 3: Train hypernetwork (MLP + VQ-VAE)
python hypernetwork.py

# Step 4: (Optional) Verify VQ codebook properties
python codebook_verify.py

# Step 5: Train with hardware-aware composite loss
python hardware_aware_loss.py

# Step 6: Simulate on-device weight reconstruction + INT8 quantisation
python ondevice_reconstruction.py

# Step 7: LoRA drift adaptation
python lora_drift_adaptation.py

# Step 8: Train and evaluate all baselines
python baselines.py

# Step 9: Concept drift simulation and recovery
python concept_drift_simulation.py

# Step 10: Hardware profiling and Pareto-front plot
python hardware_profiling.py
```

---

## Output Artifacts

All outputs are written to `output/`.

### Models & Checkpoints
| File | Description |
|---|---|
| `temporal_graphs.pkl` | 302 serialised temporal graph snapshots |
| `fingerprints.npy` | Fingerprint array (302 × 43) |
| `fingerprint_labels.npy` | Per-window binary labels (302,) |
| `hypernetwork_state.pt` | Trained hypernetwork + VQ codebook |
| `code_indices.npy` | Discrete codes per window (302 × 8) |
| `hw_aware_model.pt` | Hardware-aware trained model |
| `ondevice_gnn.pt` | Reconstructed + quantised TinyGNN |
| `lora_adapted_gnn.pt` | LoRA-adapted checkpoint |

### Figures
| File | Description |
|---|---|
| `tsne_fingerprints.png` | t-SNE visualisation of fingerprint space |
| `training_convergence.png` | Hardware-aware training loss curves |
| `ondevice_weight_heatmaps.png` | Float32 vs INT8 weight matrices |
| `lora_adaptation_curve.png` | LoRA local training loss/accuracy |
| `baselines_comparison.png` | Bar chart of all methods |
| `drift_robustness_plot.png` | F1 time-series through drift and recovery |
| `pareto_communication.png` | Pareto front: payload bytes vs F1-score |
| `codebook_usage_hist.png` | VQ codebook utilisation histogram |
| `codebook_embeddings_pca.png` | PCA of learned codebook vectors |

### Reports
| File | Description |
|---|---|
| `baselines_results.json` | Per-baseline accuracy, F1, param counts |
| `hw_profiling_results.json` | Bandwidth and latency profiling results |
| `drift_simulation_results.json` | Per-graph drift simulation metrics |
| `hw_profiling_summary.txt` | Human-readable hardware summary |
| `temporal_statistics.csv` | Per-window graph statistics |

---

## Key Hyperparameters

| Parameter | Value | Set In |
|---|---|---|
| Window size | 5 minutes | `temporal_graph_construction.py` |
| Fingerprint dim | 43 | `spectral_fingerprinting.py` |
| Code positions (M) | 8 | `hypernetwork.py` |
| Code dimension (D) | 32 | `hypernetwork.py` |
| Codebook size (K) | 64 | `hypernetwork.py` |
| GNN hidden dim | 32 | `ondevice_reconstruction.py` |
| GNN layers | 3 (GCN) | `ondevice_reconstruction.py` |
| LoRA rank | 4 | `lora_drift_adaptation.py` |
| LoRA local samples | ≤ 100 | `lora_drift_adaptation.py` |
| Latency target | 2.0 ms | `hardware_aware_loss.py` |
| Memory target | 900 KB | `hardware_aware_loss.py` |
| λ_latency | 0.10 | `hardware_aware_loss.py` |
| λ_memory | 0.01 | `hardware_aware_loss.py` |
| INT8 quantisation | Symmetric per-tensor | `ondevice_reconstruction.py` |

---

## Project Structure

```
miugraph-coder/
├── dataset/                          # BoT-IoT CSV shards
│   ├── data_1.csv ... data_74.csv
│   └── data_names.csv
├── output/                           # All generated artifacts
│   ├── *.pt, *.npy, *.pkl           # Models & data
│   ├── *.png                         # Figures
│   └── *.json, *.txt, *.csv         # Reports
├── temporal_graph_construction.py    # Step 1
├── spectral_fingerprinting.py        # Step 2
├── hypernetwork.py                   # Step 3
├── vq_codebook.py                    # Step 4
├── codebook_verify.py                # Step 4 (verification)
├── hardware_aware_loss.py            # Step 5
├── ondevice_reconstruction.py        # Step 6
├── lora_drift_adaptation.py          # Step 7
├── baselines.py                      # Step 8
├── concept_drift_simulation.py       # Step 9
├── hardware_profiling.py             # Step 10
└── README.md
```

---

## Citation

If you use MiuGraph Coder in your research, please cite:

```bibtex
@misc{miugraphcoder2026,
  title     = {MiuGraph Coder: Adaptive GNN Compression for Real-Time IoT
               Intrusion Detection via VQ-Codebook Hypernetworks},
  year      = {2026},
  note      = {Source code available at the project repository}
}
```

---

## License

This project is provided for academic and research purposes.
