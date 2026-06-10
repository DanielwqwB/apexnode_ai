"""
SentryMesh — VigilantPath ST-GNN Model
Spatial-Temporal Graph Neural Network for multi-hazard severity prediction.

Architecture:
    Input node features  →  2× GCN spatial layers
                         →  GRU temporal encoder  (per-node)
                         →  MLP classifier head
                         →  Severity score [0,1] + binary event label
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ST-GNN MODEL
# ══════════════════════════════════════════════════════════════════════════════

class SpatialEncoder(nn.Module):
    """Two-layer GCN for spatial message passing."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.conv1   = GCNConv(in_dim,  hidden)
        self.conv2   = GCNConv(hidden,  out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return x


class TemporalEncoder(nn.Module):
    """Per-node GRU for temporal modelling."""

    def __init__(self, in_dim: int, hidden: int, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=num_layers,
                          batch_first=True, dropout=0.2)

    def forward(self, x):
        """x: (B*N, T, F)  →  out: (B*N, H)"""
        _, h = self.gru(x)
        return h[-1]   # last layer hidden state


class VigilantPathEngine(nn.Module):
    """
    Full ST-GNN:
        1. Spatial GCN  (graph convolution over sensor mesh)
        2. Temporal GRU (track hazard evolution per node)
        3. MLP head     (severity score + binary label)
    """

    def __init__(self,
                 node_feat_dim: int,
                 time_window: int  = 6,
                 gcn_hidden: int   = 64,
                 gcn_out: int      = 32,
                 gru_hidden: int   = 64,
                 mlp_hidden: int   = 32,
                 num_hazard_types: int = 3,
                 dropout: float    = 0.3):
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.time_window   = time_window

        # Spatial
        self.spatial = SpatialEncoder(
            in_dim  = node_feat_dim * time_window,
            hidden  = gcn_hidden,
            out_dim = gcn_out,
            dropout = dropout
        )

        # Temporal — receives (N, T, F) properly shaped sequence
        self.temporal = TemporalEncoder(
            in_dim     = node_feat_dim,
            hidden     = gru_hidden,
            num_layers = 2
        )

        # Fusion
        fusion_dim = gcn_out + gru_hidden
        self.fusion_norm = nn.LayerNorm(fusion_dim)

        # Multi-task head
        self.severity_head = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
            nn.Sigmoid()           # output ∈ [0,1]
        )
        self.event_head = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
            # raw logit — use BCEWithLogitsLoss in trainer
        )
        # Issue #11 fix: hazard_embed was never called in forward(); removed to avoid confusion.

    def forward(self, data):
        """
        data.x          : (N, T*F)  — time-flattened node features
        data.edge_index : (2, E)
        """
        x_flat     = data.x           # (N, T*F)
        edge_index = data.edge_index
        N, TF      = x_flat.shape
        T          = self.time_window
        F          = self.node_feat_dim

        # ── Spatial encoding ──
        sp_out = self.spatial(x_flat, edge_index)   # (N, gcn_out)

        # ── Temporal encoding ──
        # Bug #3 fix: reshape from (N, T*F) → (N, T, F) so the GRU sees
        # T proper time-steps of dimension F, not a single step of T*F.
        x_seq  = x_flat.view(N, T, F)               # (N, T, F)
        tm_out = self.temporal(x_seq)                # (N, gru_hidden)

        # ── Fusion ──
        fused  = torch.cat([sp_out, tm_out], dim=-1) # (N, fusion_dim)
        fused  = self.fusion_norm(fused)

        severity = self.severity_head(fused).squeeze(-1)   # (N,)
        event    = self.event_head(fused).squeeze(-1)      # (N,) logits

        return severity, event


# ══════════════════════════════════════════════════════════════════════════════
# 3.  LOSS — weighted for class imbalance
# ══════════════════════════════════════════════════════════════════════════════

class SentryMeshLoss(nn.Module):
    """Combined severity MSE + binary event BCE (with positive-class weight)."""

    def __init__(self, pos_weight: float = 10.0, alpha: float = 0.4):
        super().__init__()
        self.alpha      = alpha   # weight of severity loss vs event loss
        self.pos_weight = pos_weight

    def forward(self, severity_pred, event_logits, labels):
        pos_w  = torch.tensor([self.pos_weight], device=labels.device)
        bce    = F.binary_cross_entropy_with_logits(
                    event_logits, labels, pos_weight=pos_w)
        mse    = F.mse_loss(severity_pred, labels)
        return self.alpha * mse + (1 - self.alpha) * bce


# ══════════════════════════════════════════════════════════════════════════════
# 4.  QUICK SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    FEAT_DIM   = 6
    TIME_WIN   = 6
    N_NODES    = 50
    N_EDGES    = 120

    x   = torch.randn(N_NODES, FEAT_DIM * TIME_WIN)
    ei  = torch.randint(0, N_NODES, (2, N_EDGES))
    y   = torch.randint(0, 2, (N_NODES,)).float()

    data = Data(x=x, edge_index=ei, y=y, num_nodes=N_NODES)
    model = VigilantPathEngine(node_feat_dim=FEAT_DIM, time_window=TIME_WIN)
    model.eval()
    with torch.no_grad():
        sev, evt = model(data)

    print("Severity output:", sev.shape, "  min/max:", sev.min().item(), sev.max().item())
    print("Event logits   :", evt.shape)
    print("Model params   :", sum(p.numel() for p in model.parameters()))
    print("✓ Model forward pass OK")
