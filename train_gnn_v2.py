"""
SentryMesh — VigilantPath ST-GNN  (v2, fixed data framing)

The original GNN sat at ROC-AUC ~0.5 because its dataset was broken:
  - no true negatives (every row was an actual event; label = severe vs mild)
  - empty temporal windows (a node only had features in the snapshot where it had
    an event, so the GRU's input window was almost all zeros)

This v2 reframes the GNN onto the SAME leakage-free occurrence data the
susceptibility models use, so the task is real:

  Nodes      = flood + landslide events (positives) AND random no-event points
               (negatives), each with antecedent rainfall + terrain features.
  Spatial    = GCN over a haversine k-NN graph of node locations (risk propagates
               between nearby places).
  Temporal   = GRU reads the antecedent-rainfall ESCALATION sequence
               [rain_30d -> rain_7d -> rain_3d -> rain_1d] — a genuine build-up
               toward the event, not an empty window.
  Setting    = transductive node classification with train/val/test masks.

Reuses VigilantPathEngine + SentryMeshLoss from model.py. CPU is fine.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Import torch BEFORE numpy/sklearn — on this Windows env, loading numpy's OpenMP
# first breaks torch's c10.dll init (WinError 1114).
import torch
from model import VigilantPathEngine, SentryMeshLoss

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, classification_report,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

CKPT = Path("checkpoints")
CKPT.mkdir(exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Temporal channel: antecedent-rainfall escalation toward the event (far -> near)
RAIN_SEQ = ["rain_30d", "rain_7d", "rain_3d", "rain_1d"]
# Static per-node context, repeated at every time-step
STATIC = ["elev", "slope_deg", "relief", "rp10_risk", "rp100_risk",
          "rain_api", "rain_max3", "month_sin", "month_cos", "lat", "lon"]
T = len(RAIN_SEQ)
F = 1 + len(STATIC)            # per-timestep feature dim


def load_nodes():
    f = pd.read_parquet("data/flood_susceptibility.parquet"); f["hazard"] = 1
    l = pd.read_parquet("data/susceptibility.parquet");        l["hazard"] = 2
    df = pd.concat([f, l], ignore_index=True)
    for c in RAIN_SEQ + STATIC:
        if c not in df.columns:
            df[c] = 0.0
    df[RAIN_SEQ + STATIC] = df[RAIN_SEQ + STATIC].fillna(0.0)
    df["label"] = df["label"].fillna(0).astype(np.float32)
    return df.reset_index(drop=True)


def build_graph(df, radius_deg=1.5, k=12):
    coords = np.radians(df[["lat", "lon"]].to_numpy(float))
    tree = BallTree(coords, metric="haversine")
    kk = min(k + 1, len(df))
    dist, idx = tree.query(coords, k=kk)
    src = np.repeat(np.arange(len(df)), kk)
    dst = idx.ravel()
    keep = (src != dst) & (dist.ravel() <= np.radians(radius_deg))
    src, dst = src[keep], dst[keep]
    # undirected
    ei = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    return torch.tensor(ei, dtype=torch.long)


def assemble_x(df, scaler, fit_mask=None):
    raw = df[RAIN_SEQ + STATIC].to_numpy(np.float32)
    if fit_mask is not None:
        scaler.fit(raw[fit_mask])
    raw = scaler.transform(raw).astype(np.float32)
    rain = raw[:, :T]                      # (N, T)
    stat = raw[:, T:]                      # (N, |STATIC|)
    N = len(df)
    x = np.zeros((N, T * F), np.float32)
    for i in range(T):
        x[:, i * F] = rain[:, i]           # this step's rainfall window
        x[:, i * F + 1:(i + 1) * F] = stat  # static context repeated
    return x


def split_masks(y, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_tr, n_val = int(.70 * len(y)), int(.15 * len(y))
    tr, val, te = idx[:n_tr], idx[n_tr:n_tr + n_val], idx[n_tr + n_val:]
    m = lambda s: torch.tensor(np.isin(np.arange(len(y)), s))
    return m(tr), m(val), m(te)


def best_threshold(y, p):
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    df = load_nodes()
    print(f"nodes: {len(df)}  positives: {int(df.label.sum())}  device: {DEV}")

    tr_m, val_m, te_m = split_masks(df["label"].to_numpy())
    scaler = StandardScaler()
    x = assemble_x(df, scaler, fit_mask=tr_m.numpy())
    ei = build_graph(df)
    print(f"edges: {ei.shape[1]}  | per-step F={F}  T={T}")

    from torch_geometric.data import Data
    data = Data(x=torch.tensor(x), edge_index=ei,
                y=torch.tensor(df["label"].to_numpy(np.float32))).to(DEV)
    tr_m, val_m, te_m = tr_m.to(DEV), val_m.to(DEV), te_m.to(DEV)

    model = VigilantPathEngine(node_feat_dim=F, time_window=T, gcn_hidden=128,
                               gcn_out=64, gru_hidden=128, mlp_hidden=64,
                               dropout=0.3).to(DEV)
    pos_w = float((df.label == 0).sum() / max((df.label == 1).sum(), 1))
    crit = SentryMeshLoss(alpha=0.3, focal_alpha=0.75, focal_gamma=2.0).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2)

    y_np = data.y.cpu().numpy()
    best_auc, best_state, stale = -1.0, None, 0
    for epoch in range(1, 201):
        model.train(); opt.zero_grad()
        sev, evt = model(data)
        loss = crit(sev, evt, data.y, mask=tr_m)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(epoch)

        model.eval()
        with torch.no_grad():
            _, evt = model(data)
            p = torch.sigmoid(evt).cpu().numpy()
        vm = val_m.cpu().numpy()
        val_auc = roc_auc_score(y_np[vm], p[vm])
        if val_auc > best_auc:
            best_auc, stale = val_auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 20 == 0:
            print(f"  epoch {epoch:03d} | loss {loss.item():.4f} | val_auc {val_auc:.4f}")
        if stale >= 30:
            print(f"  early stop @ {epoch}"); break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        _, evt = model(data)
        p = torch.sigmoid(evt).cpu().numpy()
    vm, tm = val_m.cpu().numpy(), te_m.cpu().numpy()
    thr = best_threshold(y_np[vm], p[vm])
    pred = (p[tm] >= thr).astype(int)
    m = {"threshold": thr,
         "f1": f1_score(y_np[tm], pred, zero_division=0),
         "precision": precision_score(y_np[tm], pred, zero_division=0),
         "recall": recall_score(y_np[tm], pred, zero_division=0),
         "roc_auc": roc_auc_score(y_np[tm], p[tm]),
         "avg_prec": average_precision_score(y_np[tm], p[tm])}
    print("\nST-GNN v2 — Test Results")
    for k, v in m.items():
        print(f"  {k:10s}: {v:.4f}")
    print("\n", classification_report(y_np[tm], pred,
          target_names=["No event", "Event"], zero_division=0))

    torch.save({"model": model.state_dict(), "metrics": m,
                "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                "threshold": thr}, CKPT / "best_model.pt")
    json.dump(m, open(CKPT / "test_metrics.json", "w"), indent=2)
    print(f"\nsaved -> {CKPT}/best_model.pt + test_metrics.json")


if __name__ == "__main__":
    main()
