"""
SentryMesh — Kaggle GPU Trainer (self-contained)

Trains the landslide AND flood susceptibility models on Kaggle's free GPU.
No local imports — just upload the parquet datasets and run.

SETUP ON KAGGLE
  1. Build the datasets locally first:
        python build_susceptibility.py          -> data/susceptibility.parquet
        python build_flood_susceptibility.py     -> data/flood_susceptibility.parquet
  2. Kaggle → Datasets → New Dataset → upload BOTH parquet files.
  3. New Notebook → Add Data (your dataset) → Settings → Accelerator → GPU.
  4. Paste this file into a cell and Run All.

It auto-finds the parquets under /kaggle/input and writes results to
/kaggle/working. Runs on CPU locally too (paths fall back to ./data).
"""

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, classification_report,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

FEAT = ["rain_1d", "rain_3d", "rain_7d", "rain_30d", "rain_max3", "rain_api",
        "elev", "slope_deg", "relief", "lat", "lon", "month_sin", "month_cos",
        "rp10_risk", "rp100_risk"]   # rp* present only in flood parquet; landslide -> 0

CFG = {
    "epochs": 400, "batch_size": 256, "lr": 1e-3, "weight_decay": 1e-4,
    "patience": 40, "device": "cuda" if torch.cuda.is_available() else "cpu",
}
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("checkpoints")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_parquet(name):
    """Locate a parquet by filename under /kaggle/input, else ./data."""
    for base in ["/kaggle/input", "data", "."]:
        hits = glob.glob(f"{base}/**/{name}", recursive=True)
        if hits:
            return hits[0]
    return None


class TabularEventNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def best_threshold(y, p):
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def make_loader(x, y, bs, shuffle):
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def predict_proba(model, x, device, bs=4096):
    model.eval()
    out = []
    for xb, _ in make_loader(x, np.zeros(len(x), np.float32), bs, False):
        out.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(out)


def train_model(model, train_loader, x_val, y_val, pos_weight, device):
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=10, factor=0.5)
    best_state, best_f1, stale = None, -1.0, 0
    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        p_val = predict_proba(model, x_val, device)
        thr, val_f1 = best_threshold(y_val, p_val)
        sched.step(val_f1)
        if val_f1 > best_f1:
            best_f1, stale = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 20 == 0:
            print(f"  epoch {epoch:03d} | loss {np.mean(losses):.4f} | val_f1 {val_f1:.4f} | thr {thr:.2f}")
        if stale >= CFG["patience"]:
            print(f"  early stop @ epoch {epoch}")
            break
    model.load_state_dict(best_state)
    return model


def run(name, parquet, pos_label, neg_label, tag):
    path = find_parquet(parquet)
    if not path:
        print(f"\n[{name}] {parquet} not found — skipping. Upload it to your Kaggle dataset.")
        return
    print(f"\n{'='*60}\n{name}  ({path})\n{'='*60}")
    df = pd.read_parquet(path)
    print(f"rows: {len(df)}  pos: {int(df['label'].sum())}  neg: {int((df['label']==0).sum())}  | device: {CFG['device']}")

    x = df.reindex(columns=FEAT).fillna(0.0).to_numpy(np.float32)   # missing cols (e.g. rp* in landslide) -> 0
    y = df["label"].to_numpy(np.float32)
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(x, y, test_size=0.30, stratify=y, random_state=42)
    x_val, x_te, y_val, y_te = train_test_split(x_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

    sc = StandardScaler()
    x_tr = sc.fit_transform(x_tr).astype(np.float32)
    x_val = sc.transform(x_val).astype(np.float32)
    x_te = sc.transform(x_te).astype(np.float32)

    device = torch.device(CFG["device"])
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    model = TabularEventNet(len(FEAT)).to(device)
    model = train_model(model, make_loader(x_tr, y_tr, CFG["batch_size"], True),
                        x_val, y_val, pos_weight, device)

    thr, _ = best_threshold(y_val, predict_proba(model, x_val, device))
    p_te = predict_proba(model, x_te, device)
    pred = (p_te >= thr).astype(int)
    m = {"threshold": thr,
         "f1": f1_score(y_te, pred, zero_division=0),
         "precision": precision_score(y_te, pred, zero_division=0),
         "recall": recall_score(y_te, pred, zero_division=0),
         "roc_auc": roc_auc_score(y_te, p_te),
         "avg_prec": average_precision_score(y_te, p_te)}
    print(f"\n{name} — Test Results")
    for k, v in m.items():
        print(f"  {k:10s}: {v:.4f}")
    print("\n", classification_report(y_te, pred, target_names=[neg_label, pos_label], zero_division=0))

    torch.save({"model_state": model.state_dict(), "feature_cols": FEAT,
                "scaler_mean": sc.mean_, "scaler_scale": sc.scale_,
                "threshold": thr, "metrics": m}, OUT_DIR / f"{tag}_model.pt")
    with open(OUT_DIR / f"{tag}_metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    print(f"saved → {OUT_DIR}/{tag}_model.pt")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    run("LANDSLIDE SUSCEPTIBILITY", "susceptibility.parquet", "Landslide", "No event", "landslide")
    run("FLOOD SUSCEPTIBILITY", "flood_susceptibility.parquet", "Flood", "No flood", "flood")
    print("\n✓ done")
