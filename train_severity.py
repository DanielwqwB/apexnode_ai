"""
SentryMesh — VigilantPath Severity Engine (event-level classification)

Well-posed reframing of the task: instead of forecasting which grid cell
activates next week (near-random, AUC capped ~0.75), classify the SEVERITY of
each hazard EVENT from its physical conditions — antecedent rainfall, terrain,
season, location, return-period risk, satellite spectral signal, trigger type.
This is the "Severity Engine" described in the roadmap and is what lets the
strong rainfall/terrain features actually be used.

Hazards: FLOOD + LANDSLIDE only (typhoon handled separately via OpenWeather).

LEAKAGE GUARD: features that define the label are excluded —
  landslide label = fatality>0   → drop log_fatality, log_injury
  flood label     = Severity>=2 or exposure → drop Severity, log_exposed, exposed_area
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score, classification_report)

from data_loader import load_flood, load_landslide

# Columns that leak the label — never use as features
LEAK_COLS = {"Severity", "log_exposed", "exposed_area", "log_fatality", "log_injury"}
ID_COLS   = {"event_id", "time", "hazard_type", "label", "LAT", "LON"}


def build_event_table():
    flood = load_flood()
    land  = load_landslide()
    df = pd.concat([flood, land], ignore_index=True, sort=False)

    # cyclical season + keep raw lat/lon as features
    t = pd.to_datetime(df["time"], errors="coerce")
    df["month_sin"] = np.sin(2 * np.pi * t.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * t.dt.month / 12)
    df["day_sin"]   = np.sin(2 * np.pi * t.dt.dayofyear / 365)
    df["day_cos"]   = np.cos(2 * np.pi * t.dt.dayofyear / 365)
    df["lat_feat"]  = df["LAT"]
    df["lon_feat"]  = df["LON"]
    df["hazard_code"] = (df["hazard_type"] == "flood").astype(int)

    df = df.fillna(0.0)
    return df


def select_features(df):
    feat_cols = [c for c in df.select_dtypes(include=np.number).columns
                 if c not in ID_COLS and c not in LEAK_COLS]
    return feat_cols


def find_best_threshold(y, p):
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def main():
    print("Building event table …")
    df = build_event_table()
    feat_cols = select_features(df)
    print(f"  events: {len(df):,}  |  features: {len(feat_cols)}")
    print(f"  positives: {int(df['label'].sum())} ({df['label'].mean()*100:.1f}%)")
    print(f"  features used: {feat_cols}\n")

    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=int)

    # stratified split: 70 train / 15 val (threshold) / 15 test
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30,
                                                stratify=y, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50,
                                                stratify=y_tmp, random_state=42)

    # class-balanced sample weights
    pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    sw = np.where(y_tr == 1, pos_w, 1.0)

    print("Training HistGradientBoosting …")
    clf = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_depth=None,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=42,
    )
    clf.fit(X_tr, y_tr, sample_weight=sw)

    p_val = clf.predict_proba(X_val)[:, 1]
    thr, val_f1 = find_best_threshold(y_val, p_val)

    p_te = clf.predict_proba(X_te)[:, 1]
    pred = (p_te >= thr).astype(int)
    metrics = {
        "threshold": thr,
        "f1":        f1_score(y_te, pred, zero_division=0),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall":    recall_score(y_te, pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_te, p_te),
        "avg_prec":  average_precision_score(y_te, p_te),
    }

    print("\n── Test Results (event-level severity) ─────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    print("\n", classification_report(y_te, pred,
          target_names=["Low/None", "Severe"], zero_division=0))

    # feature importance via permutation (quick) on test
    from sklearn.inspection import permutation_importance
    imp = permutation_importance(clf, X_te, y_te, n_repeats=5,
                                 random_state=42, scoring="f1")
    order = np.argsort(imp.importances_mean)[::-1][:12]
    print("Top features:")
    for i in order:
        print(f"  {feat_cols[i]:18s} {imp.importances_mean[i]:.4f}")

    Path("checkpoints").mkdir(exist_ok=True)
    with open("checkpoints/severity_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n✓ saved checkpoints/severity_metrics.json")


if __name__ == "__main__":
    main()
