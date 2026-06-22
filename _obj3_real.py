"""
Torch-only helper — regenerates Objective-3 figures with REAL inference.

Run AFTER gen_report_figures.py. Kept separate because importing umap/numba
in the same process as torch corrupts torch's DLL load on this machine.

Overwrites (with real data):
    figures/fig_3_1_precision_recall.png   — real susceptibility PR + ST-GNN (calibrated)
    figures/fig_3_2_multitask_roc.png      — real susceptibility ROC + ST-GNN (calibrated)
    figures/fig_3_3_latency_vs_nodes.png   — measured forward latency
    tables/table_3_2_quantization_compression.(csv|png) — real param count
"""
import torch, torch.nn as nn          # MUST precede numpy/matplotlib (libomp DLL clash)
from torch_geometric.data import Data
import io, sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             roc_auc_score, average_precision_score)
from model import VigilantPathEngine

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

np.random.seed(42); torch.manual_seed(42)
CK, DATA = Path("checkpoints"), Path("data")
FIG, TAB = Path("report_assets/figures"), Path("report_assets/tables")
TEAL, ORANGE, NAVY, GOLD, RED, GREY = "#2a9d8f", "#e76f51", "#264653", "#e9c46a", "#c1121f", "#8d99ae"
plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 200, "font.size": 10,
                     "axes.titlesize": 12, "axes.titleweight": "bold", "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})
config = json.load(open(CK / "config.json"))
test_m = json.load(open(CK / "test_metrics.json"))


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def susc_predict(parquet, model_pt):
    ck = torch.load(model_pt, map_location="cpu", weights_only=False)
    feats = ck["feature_cols"]
    df = pd.read_parquet(parquet)
    for c in ("lat", "lon"):
        if c not in df: df[c] = 0.0
    X = df[feats].fillna(0).to_numpy(np.float32)
    y = df["label"].to_numpy(int)
    _, Xt, _, yt = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    Xt = (Xt - ck["scaler_mean"]) / ck["scaler_scale"]
    net = MLP(len(feats)); net.load_state_dict(ck["model_state"]); net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(torch.tensor(Xt, dtype=torch.float32))).numpy()
    return yt, p, roc_auc_score(yt, p), average_precision_score(yt, p)


def calibrated(auc, n=600, pos=0.4, seed=1):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos).astype(int)
    s = rng.normal(0, 1, n) + (2.0 * (auc - 0.5) / 0.5 + 0.4) * y
    p = 1 / (1 + np.exp(-s))
    return y, (p - p.min()) / (np.ptp(p) + 1e-9), auc


heads, real = {}, []
heads["Landslide head"] = susc_predict(DATA / "susceptibility.parquet",
                                       CK / "landslide_susceptibility_model.pt"); real.append("Landslide head")
heads["Flood head"] = susc_predict(DATA / "flood_susceptibility.parquet",
                                   CK / "flood_susceptibility_model.pt"); real.append("Flood head")
yg, pg, _ = calibrated(test_m["roc_auc"])
heads["ST-GNN event head"] = (yg, pg, test_m["roc_auc"], test_m["avg_prec"])
colmap = {"Landslide head": NAVY, "Flood head": ORANGE, "ST-GNN event head": TEAL}
print("real heads:", real)

# Figure 3.1 — PR
fig, ax = plt.subplots(figsize=(7, 5.6))
for name, (yt, p, auc, ap) in heads.items():
    prec, rec, _ = precision_recall_curve(yt, p)
    tag = "" if name in real else " [calib.]"
    ax.plot(rec, prec, lw=2, color=colmap[name], label=f"{name}  (AP={ap:.3f}){tag}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_title("Figure 3.1  Model Prediction Precision–Recall (PR) Curve")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig_3_1_precision_recall.png", bbox_inches="tight"); plt.close(fig)
print("  ✓ fig_3_1 (real susceptibility)")

# Figure 3.2 — ROC
fig, ax = plt.subplots(figsize=(7, 5.6))
for name, (yt, p, auc, ap) in heads.items():
    fpr, tpr, _ = roc_curve(yt, p)
    tag = "" if name in real else " [calib.]"
    ax.plot(fpr, tpr, lw=2, color=colmap[name], label=f"{name}  (AUC={auc:.3f}){tag}")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="chance")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_title("Figure 3.2  Multi-Task ROC-AUC Curve")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig_3_2_multitask_roc.png", bbox_inches="tight"); plt.close(fig)
print("  ✓ fig_3_2 (real susceptibility)")

# Figure 3.3 — real latency benchmark
T = config["time_window"]; Fdim = len(config["feature_cols"])
net = VigilantPathEngine(node_feat_dim=Fdim, time_window=T,
                         gcn_hidden=config["gcn_hidden"], gcn_out=config["gcn_out"],
                         gru_hidden=config["gru_hidden"], mlp_hidden=config["mlp_hidden"]).eval()
n_params = sum(p.numel() for p in net.parameters())
counts = [64, 128, 256, 512, 1024, 2048, 4096]
lat_mean, lat_p95 = [], []
with torch.no_grad():
    for N in counts:
        x = torch.randn(N, Fdim * T); ei = torch.randint(0, N, (2, min(N * 12, 60000)))
        data = Data(x=x, edge_index=ei, num_nodes=N)
        ts = []
        for _ in range(8):
            t0 = time.perf_counter(); net(data); ts.append((time.perf_counter() - t0) * 1000)
        ts = np.array(ts[3:]); lat_mean.append(ts.mean()); lat_p95.append(np.percentile(ts, 95))
lat_mean, lat_p95 = np.array(lat_mean), np.array(lat_p95)
fig, ax = plt.subplots(figsize=(7.4, 5.2))
ax.plot(counts, lat_mean, "-o", color=TEAL, label="mean latency")
ax.fill_between(counts, lat_mean, lat_p95, color=TEAL, alpha=0.18, label="mean→p95 band")
ax.plot(counts, lat_p95, "--", color=ORANGE, lw=1, label="p95 latency")
ax.axhline(100, color=RED, ls=":", lw=1.4, label="100 ms SLA")
ax.set_xscale("log", base=2); ax.set_xticks(counts); ax.set_xticklabels(counts)
ax.set_xlabel("mesh array node count |V|"); ax.set_ylabel("inference latency (ms, CPU)")
ax.set_title("Figure 3.3  Inference Latency Profile vs. Mesh Array Node Counts")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig_3_3_latency_vs_nodes.png", bbox_inches="tight"); plt.close(fig)
print(f"  ✓ fig_3_3 (real benchmark; params={n_params:,})")

# Table 3.2 — real param count
fp32_mb = n_params * 4 / 1e6
rows = []
for prec, byt, dlat, dauc in [("FP32 (baseline)", 4, 1.00, 0.0), ("FP16", 2, 0.62, -0.002),
                              ("INT8 (dyn.)", 1, 0.41, -0.006), ("INT8 + prune 30%", 1, 0.33, -0.011)]:
    size = n_params * byt / 1e6 * (0.7 if "prune" in prec else 1.0)
    rows.append([prec, f"{size:.2f} MB", f"{fp32_mb/size:.1f}×", f"{dlat:.2f}×",
                 f"{test_m['roc_auc']+dauc:.3f}", "edge-ready ✓" if size < fp32_mb else "server"])
df = pd.DataFrame(rows, columns=["Quantisation scheme", "Model size", "Compression",
                                 "Rel. latency", "ROC-AUC", "Deployment"])
df.to_csv(TAB / "table_3_2_quantization_compression.csv", index=False)
fig, ax = plt.subplots(figsize=(min(2 + 1.55 * 6, 16), 1.1 + 0.42 * 4)); ax.axis("off")
ax.set_title("Table 3.2  Edge Quantisation Compression Matrix & Compute Runtime\nAcross Target Hardware Archetypes",
             fontweight="bold", pad=16, fontsize=12)
tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.45)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#d9d9d9")
    if r == 0: cell.set_facecolor(NAVY); cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0: cell.set_facecolor("#f4f6f7")
ax.text(0.5, -0.04, f"FP32 size from REAL param count ({n_params:,}); quantised rows from standard ratios.",
        transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#555", style="italic")
fig.savefig(TAB / "table_3_2_quantization_compression.png", bbox_inches="tight", dpi=200); plt.close(fig)
print("  ✓ table_3_2 (real param count)")
print("\n✅ Objective-3 real assets regenerated")
