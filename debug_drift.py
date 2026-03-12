import os, sys, copy, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
from ondevice_reconstruction import TinyGNN, WeightReconstructor
from lora_drift_adaptation import LoRATinyGNN
from concept_drift_simulation import inject_drift

with open('output/temporal_graphs.pkl', 'rb') as f:
    tg = pickle.load(f)

ckpt = torch.load('output/ondevice_gnn.pt', map_location='cpu', weights_only=False)
codebook = ckpt['codebook']
K, D = codebook.shape
edge_mlp_shared = nn.Sequential(nn.Linear(12, D), nn.GELU(), nn.Linear(D, 2))
hw_state = ckpt['hw_model_state_dict']
with torch.no_grad():
    edge_mlp_shared[0].weight.copy_(hw_state['edge_mlp.0.weight'])
    edge_mlp_shared[0].bias.copy_(hw_state['edge_mlp.0.bias'])
    edge_mlp_shared[2].weight.copy_(hw_state['edge_mlp.2.weight'])
    edge_mlp_shared[2].bias.copy_(hw_state['edge_mlp.2.bias'])

code_indices = np.load('output/code_indices.npy')
reconstructor = WeightReconstructor(codebook, node_feat_dim=15)

drift_start = 151
adapt_graphs = []
for j in range(drift_start, drift_start+5):
    dg = inject_drift(tg[j], 0.30, 2.0, 0.0, np.random.RandomState(j))
    if dg['num_edges'] >= 1:
        adapt_graphs.append(dg)

print(f'Adapt graphs: {len(adapt_graphs)}, edges: {[g["num_edges"] for g in adapt_graphs]}')

ea_samples = [g['edge_attr'] for g in adapt_graphs if g.get('edge_attr') is not None]
if ea_samples:
    all_ea = np.concatenate(ea_samples)
    print(f'edge_attr range: [{all_ea.min():.3f}, {all_ea.max():.3f}], mean={all_ea.mean():.3f}')

z = torch.tensor(code_indices[drift_start], dtype=torch.long)
weights = reconstructor(z)
base_gnn = TinyGNN(node_feat_dim=15, hidden=D, num_classes=2, edge_feat_dim=12)
base_gnn.load_weights(weights)
lora_m = LoRATinyGNN(base_gnn, rank=4, edge_mlp=edge_mlp_shared)

g = adapt_graphs[0]
nf = torch.tensor(g['node_feat'], dtype=torch.float32)
ei = torch.tensor(g['edge_index'], dtype=torch.long)
ey = torch.tensor(g['edge_y'], dtype=torch.long)
ea = torch.tensor(g['edge_attr'], dtype=torch.float32) if g.get('edge_attr') is not None else None

print(f'nf range: [{nf.min():.3f}, {nf.max():.3f}]')
print(f'ey: {ey.unique(return_counts=True)}')

lora_m.train()
logits, _ = lora_m(nf, ei, ea)
loss = F.cross_entropy(logits, ey)
print(f'Forward logits range: [{logits.min():.3f}, {logits.max():.3f}], loss={loss:.4f}')
loss.backward()

for n, p in lora_m.named_parameters():
    if p.grad is not None:
        print(f'  {n}: grad_norm={p.grad.norm():.4f}')
    else:
        print(f'  {n}: NO GRAD')

# Check trainable params
print('\nTrainable params:')
for n, p in lora_m.named_parameters():
    if p.requires_grad:
        print(f'  {n}: shape={tuple(p.shape)}')
