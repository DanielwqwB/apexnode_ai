"""
SentryMesh - Flood Susceptibility Model (GPU PyTorch version)

Trains on data/flood_susceptibility.parquet from build_flood_susceptibility.py.
Uses CUDA automatically when available.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Import torch BEFORE numpy/sklearn — on this Windows env, loading numpy's OpenMP
# first breaks torch's c10.dll init (WinError 1114).
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEAT = [
    "rain_1d", "rain_3d", "rain_7d", "rain_30d", "rain_max3", "rain_api",
    "elev", "slope_deg", "relief", "lat", "lon", "month_sin", "month_cos",
    # Static, leakage-free flood-hazard prior (Aqueduct return-period risk)
    "rp10_risk", "rp100_risk",
]

CFG = {
    "epochs": 400,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 40,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "checkpoints/flood_susceptibility_model.pt",
    "metrics_path": "checkpoints/flood_susceptibility_metrics.json",
}


class TabularEventNet(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
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


def make_loader(x, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def predict_proba(model, x, device, batch_size=4096):
    model.eval()
    probs = []
    dummy_y = np.zeros(len(x), dtype=np.float32)
    loader = make_loader(x, dummy_y, batch_size=batch_size, shuffle=False)
    for xb, _ in loader:
        logits = model(xb.to(device, non_blocking=True))
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def train_model(model, train_loader, x_val, y_val, pos_weight, cfg):
    device = torch.device(cfg["device"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=10, factor=0.5)

    best_state, best_f1, stale = None, -1.0, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        p_val = predict_proba(model, x_val, device)
        thr, val_f1 = best_threshold(y_val, p_val)
        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % 20 == 0:
            print(f"epoch {epoch:03d} | loss {np.mean(losses):.4f} | val_f1 {val_f1:.4f} | thr {thr:.2f}")
        if stale >= cfg["patience"]:
            print(f"early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.read_parquet("data/flood_susceptibility.parquet")
    print(f"rows: {len(df)}  pos: {int(df['label'].sum())}  neg: {int((df['label'] == 0).sum())}")
    print(f"device: {CFG['device']}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    x = df.reindex(columns=FEAT).fillna(0.0).to_numpy(np.float32)
    y = df["label"].to_numpy(np.float32)
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(x, y, test_size=0.30, stratify=y, random_state=42)
    x_val, x_te, y_val, y_te = train_test_split(x_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_tr).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_te = scaler.transform(x_te).astype(np.float32)

    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    train_loader = make_loader(x_tr, y_tr, CFG["batch_size"], shuffle=True)

    device = torch.device(CFG["device"])
    model = TabularEventNet(len(FEAT)).to(device)
    model = train_model(model, train_loader, x_val, y_val, pos_weight, CFG)

    p_val = predict_proba(model, x_val, device)
    thr, _ = best_threshold(y_val, p_val)
    p_te = predict_proba(model, x_te, device)
    pred = (p_te >= thr).astype(int)

    m = {
        "threshold": thr,
        "f1": f1_score(y_te, pred, zero_division=0),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_te, p_te),
        "avg_prec": average_precision_score(y_te, p_te),
        "device": CFG["device"],
    }
    print("\nFlood Susceptibility - Test Results")
    for k, v in m.items():
        print(f"  {k:10s}: {v:.4f}" if isinstance(v, float) else f"  {k:10s}: {v}")
    print("\n", classification_report(y_te, pred, target_names=["No flood", "Flood"], zero_division=0))

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "feature_cols": FEAT,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "threshold": thr,
        "metrics": m,
    }, CFG["model_path"])
    with open(CFG["metrics_path"], "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    print(f"\nsaved {CFG['model_path']}")
    print(f"saved {CFG['metrics_path']}")


if __name__ == "__main__":
    main()
