"""
SentryMesh — Results Figures for the pitch deck.

Generates (into figures/):
  1. model_comparison.png   — F1 / AUC bars across all 3 models      (always)
  2. metric_breakdown.png   — precision/recall/F1 per model          (always)
  3. stgnn_training.png     — ST-GNN loss + val F1/AUC curves         (needs history.json)
  4. roc_pr_<model>.png     — ROC + PR curves                        (needs data parquet)
  5. confusion_<model>.png  — confusion matrix                       (needs data parquet)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CK = Path("checkpoints")
FIG = Path("figures")
FIG.mkdir(exist_ok=True)


def load(p):
    return json.load(open(p)) if Path(p).exists() else None


# ── 1 & 2. Metric comparison (always available) ──────────────────────────────
def metric_figs():
    models = {
        "Landslide\nSusceptibility": load(CK / "susceptibility_metrics.json"),
        "Flood\nSusceptibility":     load(CK / "flood_susceptibility_metrics.json"),
        "ST-GNN\n(forecasting)":     load(CK / "test_metrics.json"),
    }
    models = {k: v for k, v in models.items() if v}
    names = list(models)

    # comparison: F1 + AUC
    f1s  = [models[n]["f1"] for n in names]
    aucs = [models[n]["roc_auc"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, f1s, 0.4, label="F1", color="#2a9d8f")
    ax.bar(x + 0.2, aucs, 0.4, label="ROC-AUC", color="#e76f51")
    for i, (f, a) in enumerate(zip(f1s, aucs)):
        ax.text(i - 0.2, f + 0.02, f"{f:.2f}", ha="center", fontweight="bold")
        ax.text(i + 0.2, a + 0.02, f"{a:.2f}", ha="center")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("SentryMesh — Model Performance", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "model_comparison.png", dpi=150)
    plt.close(fig)

    # per-model breakdown
    metrics = ["precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.25
    colors = ["#264653", "#2a9d8f", "#e9c46a"]
    for j, m in enumerate(metrics):
        vals = [models[n][m] for n in names]
        ax.bar(x + (j - 1) * w, vals, w, label=m.capitalize(), color=colors[j])
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 by Model", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "metric_breakdown.png", dpi=150)
    plt.close(fig)
    print("✓ model_comparison.png, metric_breakdown.png")


# ── 3. ST-GNN training curves ────────────────────────────────────────────────
def stgnn_curves():
    h = load(CK / "history.json")
    if not h:
        print("· no history.json — skipping ST-GNN curves")
        return
    ep = [r["epoch"] for r in h]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(ep, [r["loss"] for r in h], color="#e76f51", label="train loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss", color="#e76f51")
    ax2 = ax1.twinx()
    ax2.plot(ep, [r["f1"] for r in h], color="#2a9d8f", label="val F1")
    ax2.plot(ep, [r["roc_auc"] for r in h], color="#264653", ls="--", label="val AUC")
    ax2.set_ylabel("Val F1 / AUC"); ax2.set_ylim(0, 1)
    ax1.set_title("ST-GNN Training Curves", fontweight="bold")
    l1, la = ax1.get_legend_handles_labels()
    l2, lb = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la + lb, loc="center right")
    fig.tight_layout(); fig.savefig(FIG / "stgnn_training.png", dpi=150)
    plt.close(fig)
    print("✓ stgnn_training.png")


# ── 4 & 5. ROC / PR / confusion (need data parquet + model) ──────────────────
def susceptibility_curves(tag, parquet, model_pt, title):
    import pandas as pd
    if not (Path(parquet).exists() and Path(model_pt).exists()):
        print(f"· {parquet} not local — skipping {tag} ROC/PR/confusion "
              f"(download it from Kaggle to enable)")
        return
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

    FEAT = ["rain_1d", "rain_3d", "rain_7d", "rain_30d", "rain_max3", "rain_api",
            "elev", "slope_deg", "relief", "lat", "lon", "month_sin", "month_cos"]

    class Net(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.15),
                nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        def forward(self, x): return self.net(x).squeeze(-1)

    df = pd.read_parquet(parquet)
    X = df[FEAT].fillna(0.0).to_numpy(np.float32)
    y = df["label"].to_numpy(int)
    _, X_tmp, _, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    _, X_te, _, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

    ck = torch.load(model_pt, map_location="cpu", weights_only=False)
    X_te = (X_te - ck["scaler_mean"]) / ck["scaler_scale"]
    net = Net(len(FEAT)); net.load_state_dict(ck["model_state"]); net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(torch.tensor(X_te, dtype=torch.float32))).numpy()
    thr = ck["threshold"]

    # ROC + PR
    fpr, tpr, _ = roc_curve(y_te, p)
    prec, rec, _ = precision_recall_curve(y_te, p)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5))
    a.plot(fpr, tpr, color="#e76f51"); a.plot([0, 1], [0, 1], "k--", alpha=0.4)
    a.set_xlabel("False Positive Rate"); a.set_ylabel("True Positive Rate")
    a.set_title(f"ROC (AUC={ck['metrics']['roc_auc']:.3f})")
    b.plot(rec, prec, color="#2a9d8f")
    b.set_xlabel("Recall"); b.set_ylabel("Precision")
    b.set_title(f"Precision-Recall (AP={ck['metrics']['avg_prec']:.3f})")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / f"roc_pr_{tag}.png", dpi=150)
    plt.close(fig)

    # confusion matrix
    cm = confusion_matrix(y_te, (p >= thr).astype(int))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Greens")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"]); ax.set_yticklabels(["No", "Yes"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{title}\nConfusion Matrix")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(FIG / f"confusion_{tag}.png", dpi=150)
    plt.close(fig)
    print(f"✓ roc_pr_{tag}.png, confusion_{tag}.png")


if __name__ == "__main__":
    metric_figs()
    stgnn_curves()
    susceptibility_curves("landslide", "data/susceptibility.parquet",
                          "checkpoints/landslide_susceptibility_model.pt",
                          "Landslide Susceptibility")
    susceptibility_curves("flood", "data/flood_susceptibility.parquet",
                          "checkpoints/flood_susceptibility_model.pt",
                          "Flood Susceptibility")
    print(f"\n✓ figures saved to {FIG}/")
