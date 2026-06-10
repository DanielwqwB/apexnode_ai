"""
SentryMesh — Training Loop
Trains the VigilantPath ST-GNN with:
  - Class-imbalance weighting
  - Early stopping
  - Checkpoint saving (best val F1)
  - Per-hazard-type evaluation breakdown
"""

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader as PyGLoader
import numpy as np
import pandas as pd
import pickle
import json
import time
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, classification_report
)

from model import VigilantPathEngine, SentryMeshLoss
from data_loader import build_unified_dataset

# ── Config ──────────────────────────────────────────────────────────────────
CFG = {
    # Data
    "processed_dir"  : "processed",
    "checkpoint_dir" : "checkpoints",
    "time_window"    : 6,

    # Feature columns — must match data_loader output.
    # Bug #2 fix: include all enrichment columns so the model actually uses
    # the compiled_pop / return-period / spectral features data_loader computes.
    "feature_cols"   : [
        "LAT", "LON", "month", "dayofyear", "hour", "hazard_code",
        # flood enrichment
        "log_exposed", "exposed_area",
        "rp10_risk", "rp100_risk",
        "mean_MNDWI", "mean_NDVI", "flood_pixel_frac",
    ],

    # Model
    "gcn_hidden"     : 64,
    "gcn_out"        : 32,
    "gru_hidden"     : 64,
    "mlp_hidden"     : 32,
    "dropout"        : 0.3,

    # Training
    "epochs"         : 40,
    "batch_size"     : 1,       # one graph snapshot per batch
    "lr"             : 1e-3,
    "weight_decay"   : 1e-4,
    "pos_weight"     : 10.0,    # BCEWithLogits positive-class weight
    "loss_alpha"     : 0.4,     # severity loss fraction
    "patience"       : 8,       # early stopping patience
    "threshold"      : 0.35,    # event classification threshold

    # Device
    "device"         : "cuda" if torch.cuda.is_available() else "cpu",

    # Shadow-mode (eval only, no weight update) — set True to replicate
    # Section 3 "Shadow-Mode Validation" from the roadmap
    "shadow_mode"    : False,
}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(labels, preds_prob, threshold=0.35):
    preds_bin = (preds_prob >= threshold).astype(int)
    return {
        "f1"       : f1_score(labels, preds_bin, zero_division=0),
        "precision": precision_score(labels, preds_bin, zero_division=0),
        "recall"   : recall_score(labels, preds_bin, zero_division=0),
        "roc_auc"  : roc_auc_score(labels, preds_prob)
                     if len(np.unique(labels)) > 1 else 0.0,
        "avg_prec" : average_precision_score(labels, preds_prob)
                     if len(np.unique(labels)) > 1 else 0.0,
    }


def save_checkpoint(model, optimizer, epoch, val_f1, path):
    torch.save({
        "epoch"     : epoch,
        "val_f1"    : val_f1,
        "model"     : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
    }, path)


# ══════════════════════════════════════════════════════════════════════════════
# Lightweight Dataset for training without full PyG snapshot overhead
# ══════════════════════════════════════════════════════════════════════════════

class FlatGraphDataset(torch.utils.data.Dataset):
    """
    Simpler dataset: each row is one node-event, features are T*F-flattened.
    The graph is static per hazard type; we build one graph per batch on-the-fly.
    Suitable for the available data sizes.
    """

    def __init__(self, parquet_path, node_df, edge_index, feature_cols, window=6):
        self.df          = pd.read_parquet(parquet_path)
        self.node_df     = node_df
        self.edge_index  = torch.tensor(edge_index, dtype=torch.long)
        self.feature_cols= feature_cols
        self.window      = window
        self.N           = len(node_df)
        self.F           = len(feature_cols)
        self._build()

    def _build(self):
        """Build fixed-size node feature matrix and labels per time-step."""
        from torch_geometric.data import Data

        self.df = self.df.sort_values("time").reset_index(drop=True)
        times   = sorted(self.df["time"].unique())
        self.items = []

        snapshots = []
        for t in times:
            snap = self.df[self.df["time"] == t]
            # Issue #8 fix: vectorised pivot instead of row-by-row loop (O(N²) → O(N))
            fm = np.zeros((self.N, self.F), dtype=np.float32)
            lm = np.zeros(self.N, dtype=np.float32)

            valid = snap.dropna(subset=["node_id"]).copy()
            valid["node_id"] = valid["node_id"].astype(int)
            valid = valid[(valid["node_id"] >= 0) & (valid["node_id"] < self.N)]

            if len(valid) > 0:
                node_ids = valid["node_id"].values
                feat_vals = valid[self.feature_cols].fillna(0.0).values.astype(np.float32)
                label_vals = valid["label"].fillna(0.0).values.astype(np.float32)
                fm[node_ids] = feat_vals
                lm[node_ids] = label_vals

            snapshots.append((fm, lm))

        # Sliding window
        for i in range(len(snapshots) - self.window):
            window_feats = np.concatenate(
                [snapshots[i + t][0] for t in range(self.window)], axis=1
            )  # (N, T*F)
            label = snapshots[i + self.window][1]  # (N,)
            self.items.append((
                torch.tensor(window_feats),
                torch.tensor(label)
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        from torch_geometric.data import Data
        x, y = self.items[idx]
        return Data(x=x, edge_index=self.edge_index, y=y, num_nodes=self.N)


# ══════════════════════════════════════════════════════════════════════════════
# Main training function
# ══════════════════════════════════════════════════════════════════════════════

def train():
    cfg  = CFG
    dev  = torch.device(cfg["device"])
    Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    # ── 1. Build dataset (if not already done) ──────────────────────────────
    proc = Path(cfg["processed_dir"])
    if not (proc / "combined.parquet").exists():
        print("Building processed dataset from raw CSVs …\n")
        build_unified_dataset(save_path=cfg["processed_dir"])
    else:
        print("Found existing processed data — skipping preprocessing.\n")

    node_df    = pd.read_parquet(proc / "nodes.parquet")
    edge_index = np.load(proc / "edge_index.npy")
    N = len(node_df)

    feature_cols = cfg["feature_cols"]
    TW           = cfg["time_window"]
    in_dim       = len(feature_cols)

    print(f"Graph: {N} nodes  |  {edge_index.shape[1]} edges  |  features: {in_dim}\n")

    # ── 2. Datasets & loaders ────────────────────────────────────────────────
    print("Building datasets (sliding windows) …")
    train_ds = FlatGraphDataset(proc/"train.parquet", node_df, edge_index, feature_cols, TW)
    val_ds   = FlatGraphDataset(proc/"val.parquet",   node_df, edge_index, feature_cols, TW)
    test_ds  = FlatGraphDataset(proc/"test.parquet",  node_df, edge_index, feature_cols, TW)

    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}\n")

    train_loader = PyGLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = PyGLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    test_loader  = PyGLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    # ── 3. Model ─────────────────────────────────────────────────────────────
    model = VigilantPathEngine(
        node_feat_dim  = in_dim,
        time_window    = TW,
        gcn_hidden     = cfg["gcn_hidden"],
        gcn_out        = cfg["gcn_out"],
        gru_hidden     = cfg["gru_hidden"],
        mlp_hidden     = cfg["mlp_hidden"],
        dropout        = cfg["dropout"],
    ).to(dev)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Issue #9 fix: compute pos_weight from actual class distribution in training data
    train_labels_series = pd.read_parquet(proc / "train.parquet")["label"]
    n_neg = (train_labels_series == 0).sum()
    n_pos = max((train_labels_series == 1).sum(), 1)
    dynamic_pos_weight = float(n_neg / n_pos)
    print(f"Class balance — neg: {n_neg:,}  pos: {n_pos:,}  → pos_weight: {dynamic_pos_weight:.1f}")

    criterion = SentryMeshLoss(
        pos_weight = dynamic_pos_weight,
        alpha      = cfg["loss_alpha"]
    ).to(dev)

    optimizer = optim.AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    # ── 4. Training loop ─────────────────────────────────────────────────────
    best_val_f1  = 0.0
    patience_cnt = 0
    history      = []

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        if not cfg["shadow_mode"]:
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                batch = batch.to(dev)
                optimizer.zero_grad()
                sev, evt = model(batch)
                loss = criterion(sev, evt, batch.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / max(len(train_loader), 1)
        else:
            avg_loss = float("nan")
            print(f"[Shadow mode] — weights frozen, evaluating only.")

        # ── Validation ──
        model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch    = batch.to(dev)
                sev, evt = model(batch)
                probs    = torch.sigmoid(evt).cpu().numpy().flatten()
                labs     = batch.y.cpu().numpy().flatten()
                all_probs.extend(probs)
                all_labels.extend(labs)

        all_labels = np.array(all_labels)
        all_probs  = np.array(all_probs)
        val_m      = compute_metrics(all_labels, all_probs, cfg["threshold"])
        elapsed    = time.time() - t0

        print(f"Epoch {epoch:03d}/{cfg['epochs']}  "
              f"loss={avg_loss:.4f}  "
              f"val_f1={val_m['f1']:.4f}  "
              f"val_auc={val_m['roc_auc']:.4f}  "
              f"[{elapsed:.1f}s]")

        scheduler.step(val_m["f1"])
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "loss": avg_loss, "lr": current_lr, **val_m})

        if val_m["f1"] > best_val_f1:
            best_val_f1  = val_m["f1"]
            patience_cnt = 0
            save_checkpoint(
                model, optimizer, epoch, best_val_f1,
                f"{cfg['checkpoint_dir']}/best_model.pt"
            )
            print(f"  ✓ New best val F1: {best_val_f1:.4f}  → checkpoint saved")
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                print(f"\nEarly stopping at epoch {epoch} (patience={cfg['patience']})")
                break

    # ── 5. Test evaluation ───────────────────────────────────────────────────
    print("\n── Test Evaluation ─────────────────────────────────────────────────")
    ckpt = torch.load(f"{cfg['checkpoint_dir']}/best_model.pt", map_location=dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(dev)
            _, evt = model(batch)
            probs  = torch.sigmoid(evt).cpu().numpy().flatten()
            labs   = batch.y.cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labs)

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    test_m     = compute_metrics(all_labels, all_probs, cfg["threshold"])

    print(f"  F1        : {test_m['f1']:.4f}")
    print(f"  Precision : {test_m['precision']:.4f}")
    print(f"  Recall    : {test_m['recall']:.4f}")
    print(f"  ROC-AUC   : {test_m['roc_auc']:.4f}")
    print(f"  Avg Prec  : {test_m['avg_prec']:.4f}")

    preds_bin = (all_probs >= cfg["threshold"]).astype(int)
    print("\nClassification Report:\n",
          classification_report(all_labels, preds_bin,
                                target_names=["No Event", "Event"],
                                zero_division=0))

    # ── 6. Save run artifacts ─────────────────────────────────────────────────
    with open(f"{cfg['checkpoint_dir']}/history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(f"{cfg['checkpoint_dir']}/test_metrics.json", "w") as f:
        json.dump(test_m, f, indent=2)
    with open(f"{cfg['checkpoint_dir']}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n✓ Training complete.  Best val F1: {best_val_f1:.4f}")
    print(f"  Artifacts saved to: {cfg['checkpoint_dir']}/")


if __name__ == "__main__":
    train()
