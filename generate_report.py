"""
Generate evaluation reports for the trained SentryMesh model.

Outputs are written to checkpoints/reports by default:
  - predictions_test.csv
  - metrics_summary.json
  - classification_report.csv / .json / .txt / .png
  - confusion_matrix.png / .csv
  - confusion_matrix_normalized.png / .csv
  - roc_curve.png
  - precision_recall_curve.png
  - prediction_beeswarm.png
  - prediction_histogram.png
  - training_history.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch_geometric.loader import DataLoader as PyGLoader

from model import VigilantPathEngine
from train import CFG, LazyGraphDataset


CLASS_NAMES = ["No Event", "Event"]


def load_config(checkpoint_dir: Path) -> dict:
    cfg = CFG.copy()
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))

    if cfg.get("device") == "cuda" and not torch.cuda.is_available():
        cfg["device"] = "cpu"
    return cfg


def load_model(cfg: dict, checkpoint_path: Path, device: torch.device):
    model = VigilantPathEngine(
        node_feat_dim=len(cfg["feature_cols"]),
        time_window=cfg["time_window"],
        gcn_hidden=cfg["gcn_hidden"],
        gcn_out=cfg["gcn_out"],
        gru_hidden=cfg["gru_hidden"],
        mlp_hidden=cfg["mlp_hidden"],
        dropout=cfg["dropout"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def collect_predictions(cfg: dict, model, split: str, device: torch.device) -> pd.DataFrame:
    proc = Path(cfg["processed_dir"])
    node_df = pd.read_parquet(proc / "nodes.parquet")
    edge_index = np.load(proc / "edge_index.npy")

    dataset = LazyGraphDataset(
        proc / f"{split}.parquet",
        node_df,
        edge_index,
        cfg["feature_cols"],
        cfg["time_window"],
        cfg.get("snapshot_freq", "M"),
    )
    loader = PyGLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    rows = []
    threshold = cfg["threshold"]
    with torch.no_grad():
        for fallback_idx, batch in enumerate(loader):
            batch = batch.to(device)
            severity, logits = model(batch)
            prob = torch.sigmoid(logits).cpu().numpy()
            severity = severity.cpu().numpy()
            label = batch.y.cpu().numpy().astype(int)
            pred = (prob >= threshold).astype(int)
            node_id = batch.global_node_id.cpu().numpy()
            window_idx = int(batch.window_idx.cpu().numpy()[0]) if hasattr(batch, "window_idx") else fallback_idx

            rows.append(pd.DataFrame({
                "split": split,
                "window_idx": window_idx,
                "global_node_id": node_id,
                "label": label,
                "pred": pred,
                "prob_event": prob,
                "severity_score": severity,
                "correct": label == pred,
            }))

    if not rows:
        return pd.DataFrame(columns=[
            "split", "window_idx", "global_node_id", "label", "pred",
            "prob_event", "severity_score", "correct",
        ])
    return pd.concat(rows, ignore_index=True)


def save_confusion_matrix(cm: np.ndarray, out_path: Path, title: str, normalized: bool = False):
    plt.figure(figsize=(6.5, 5.5))
    fmt = ".2%" if normalized else "d"
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_classification_table(report_df: pd.DataFrame, out_path: Path):
    display_df = report_df.copy()
    for col in display_df.columns:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce")
    display_df = display_df.round(4).fillna("")

    fig_height = max(3.0, 0.48 * len(display_df) + 1.0)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        rowLabels=display_df.index,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)
    ax.set_title("Classification Report", pad=16, weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_curves(df: pd.DataFrame, out_dir: Path):
    y_true = df["label"].to_numpy()
    y_prob = df["prob_event"].to_numpy()
    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6.5, 5))
    plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="#7a869a", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=180)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_prec = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6.5, 5))
    plt.plot(recall, precision, color="#059669", lw=2, label=f"AP = {avg_prec:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "precision_recall_curve.png", dpi=180)
    plt.close()


def save_prediction_plots(df: pd.DataFrame, out_dir: Path, threshold: float):
    samples = []
    for _, group in df.groupby("label", sort=True):
        samples.append(group.sample(min(len(group), 5000), random_state=7))
    plot_df = pd.concat(samples, ignore_index=True)

    rng = np.random.default_rng(7)
    x = plot_df["label"].to_numpy(dtype=float) + rng.normal(0, 0.075, size=len(plot_df))
    colors = np.where(plot_df["correct"].to_numpy(), "#2563eb", "#dc2626")

    plt.figure(figsize=(7, 5.2))
    plt.scatter(x, plot_df["prob_event"], s=9, c=colors, alpha=0.35, linewidths=0)
    plt.axhline(threshold, color="#111827", linestyle="--", linewidth=1.2, label=f"threshold = {threshold:.2f}")
    plt.xticks([0, 1], CLASS_NAMES)
    plt.ylabel("Predicted Event Probability")
    plt.title("Prediction Beeswarm by True Class")
    plt.legend(loc="upper left")
    plt.grid(axis="y", alpha=0.22)
    plt.tight_layout()
    plt.savefig(out_dir / "prediction_beeswarm.png", dpi=180)
    plt.close()

    hist_df = df.copy()
    hist_df["actual"] = hist_df["label"].map({0: "No Event", 1: "Event"})

    plt.figure(figsize=(7, 5))
    sns.histplot(
        data=hist_df,
        x="prob_event",
        hue="actual",
        bins=50,
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
    )
    plt.axvline(threshold, color="#111827", linestyle="--", linewidth=1.2)
    plt.xlabel("Predicted Event Probability")
    plt.title("Prediction Probability Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "prediction_histogram.png", dpi=180)
    plt.close()


def save_training_history(checkpoint_dir: Path, out_dir: Path):
    history_path = checkpoint_dir / "history.json"
    if not history_path.exists():
        return

    with open(history_path, "r", encoding="utf-8") as f:
        history = pd.DataFrame(json.load(f))
    if history.empty or "epoch" not in history:
        return

    plt.figure(figsize=(8, 5))
    if "loss" in history:
        plt.plot(history["epoch"], history["loss"], marker="o", label="loss")
    if "f1" in history:
        plt.plot(history["epoch"], history["f1"], marker="o", label="val_f1")
    if "roc_auc" in history:
        plt.plot(history["epoch"], history["roc_auc"], marker="o", label="val_auc")
    plt.xlabel("Epoch")
    plt.title("Training History")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_history.png", dpi=180)
    plt.close()


def write_markdown_summary(out_dir: Path, metrics: dict):
    lines = [
        "# SentryMesh Evaluation Report",
        "",
        f"- Split: `{metrics['split']}`",
        f"- Threshold: `{metrics['threshold']:.2f}`",
        f"- Accuracy: `{metrics['accuracy']:.4f}`",
        f"- F1: `{metrics['f1']:.4f}`",
        f"- Precision: `{metrics['precision']:.4f}`",
        f"- Recall: `{metrics['recall']:.4f}`",
        f"- ROC-AUC: `{metrics['roc_auc']:.4f}`",
        f"- Average precision: `{metrics['avg_precision']:.4f}`",
        "",
        "## Files",
        "",
        "- `confusion_matrix.png`",
        "- `confusion_matrix_normalized.png`",
        "- `classification_report.png`",
        "- `roc_curve.png`",
        "- `precision_recall_curve.png`",
        "- `prediction_beeswarm.png`",
        "- `prediction_histogram.png`",
        "- `training_history.png`",
        "- `predictions_test.csv`",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate SentryMesh model report artifacts.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir = checkpoint_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(checkpoint_dir)
    device = torch.device(cfg["device"])
    model = load_model(cfg, checkpoint_dir / "best_model.pt", device)
    df = collect_predictions(cfg, model, args.split, device)

    prediction_path = out_dir / f"predictions_{args.split}.csv"
    df.to_csv(prediction_path, index=False)

    y_true = df["label"].to_numpy()
    y_pred = df["pred"].to_numpy()
    y_prob = df["prob_event"].to_numpy()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(out_dir / "confusion_matrix.csv")
    save_confusion_matrix(cm, out_dir / "confusion_matrix.png", "Confusion Matrix")

    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm_df = pd.DataFrame(cm_norm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_norm_df.to_csv(out_dir / "confusion_matrix_normalized.csv")
    save_confusion_matrix(
        cm_norm,
        out_dir / "confusion_matrix_normalized.png",
        "Confusion Matrix Normalized by Actual Class",
        normalized=True,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(out_dir / "classification_report.csv")
    with open(out_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            zero_division=0,
        ))
    save_classification_table(report_df, out_dir / "classification_report.png")

    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "split": args.split,
        "threshold": float(cfg["threshold"]),
        "total": int(len(df)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "avg_precision": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
    }
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_curves(df, out_dir)
    save_prediction_plots(df, out_dir, cfg["threshold"])
    save_training_history(checkpoint_dir, out_dir)
    write_markdown_summary(out_dir, metrics)

    print(f"Report generated at: {out_dir}")
    print(f"Confusion matrix: TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")
    print(f"F1={metrics['f1']:.4f}  Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}")


if __name__ == "__main__":
    main()
