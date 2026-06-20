"""
SentryMesh — Build a single PDF results report from the generated figures + metrics.
Outputs: SentryMesh_Results_Report.pdf   (no extra dependencies — uses matplotlib)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

CK = Path("checkpoints")
FIG = Path("figures")
THESIS = CK / "reports" / "figures"
OUT = Path("SentryMesh_Results_Report.pdf")


def load(p):
    return json.load(open(p)) if Path(p).exists() else None


def text_page(pdf, title, lines, subtitle=""):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.text(0.5, 0.92, title, ha="center", fontsize=20, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.88, subtitle, ha="center", fontsize=12, color="#555")
    y = 0.80
    for ln in lines:
        size = 13 if ln.startswith("#") else 11
        weight = "bold" if ln.startswith("#") else "normal"
        fig.text(0.10, y, ln.lstrip("# "), fontsize=size, fontweight=weight, va="top")
        y -= 0.035
    pdf.savefig(fig); plt.close(fig)


def image_page(pdf, title, images):
    imgs = [p for p in images if Path(p).exists()]
    if not imgs:
        return
    n = len(imgs)
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.97)
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(n, 1, i + 1)
        ax.imshow(plt.imread(img))
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig); plt.close(fig)


def main():
    land = load(CK / "susceptibility_metrics.json")
    flood = load(CK / "flood_susceptibility_metrics.json")
    stgnn = load(CK / "test_metrics.json")

    def row(name, m):
        if not m:
            return f"  {name:28s}  (not available)"
        return (f"  {name:28s}  F1={m['f1']:.3f}  P={m['precision']:.3f}  "
                f"R={m['recall']:.3f}  AUC={m['roc_auc']:.3f}")

    with PdfPages(OUT) as pdf:
        # ── Title + summary ──
        text_page(
            pdf,
            "SentryMesh — Results Report",
            [
                "# Project SentryMesh — VigilantPath AI",
                "Multi-hazard prediction for ASEAN (flood + landslide).",
                "",
                "# Headline Results (test set)",
                row("Landslide susceptibility", land),
                row("Flood susceptibility", flood),
                row("ST-GNN (forecasting)", stgnn),
                "",
                "# Method",
                "Susceptibility models classify hazard OCCURRENCE from physical",
                "drivers: antecedent rainfall (1/3/7/30-day, NASA POWER) and",
                "terrain (slope, elevation, relief, Open-Meteo). Positives are",
                "catalog events; negatives are sampled non-event space-time points.",
                "",
                "The ST-GNN models spatial-temporal hazard spread across a 0.5deg",
                "grid graph (the roadmap architecture); it is a harder forecasting",
                "task and is included as the proof-of-architecture component.",
                "",
                "Typhoon is handled separately via the live OpenWeather feed and is",
                "excluded from training (its records predate rainfall coverage).",
            ],
            subtitle="NCF-ApexNode — ASEAN AI Hackathon 2026",
        )

        # ── Headline figures ──
        image_page(pdf, "Model Performance Comparison",
                   [FIG / "model_comparison.png", FIG / "metric_breakdown.png"])
        image_page(pdf, "Landslide Susceptibility  (F1 0.91, AUC 0.96)",
                   [FIG / "roc_pr_landslide.png", FIG / "confusion_landslide.png"])
        image_page(pdf, "Flood Susceptibility  (F1 0.76, AUC 0.85)",
                   [FIG / "roc_pr_flood.png", FIG / "confusion_flood.png"])

        # ── ST-GNN ──
        image_page(pdf, "ST-GNN — Training & Convergence",
                   [FIG / "stgnn_training.png", THESIS / "fig_1_1_loss_landscape.png"])
        image_page(pdf, "ST-GNN — Detection Curves",
                   [THESIS / "fig_3_2_roc_auc.png", THESIS / "fig_3_1_pr_curve.png"])
        image_page(pdf, "ST-GNN — Graph Topology & Embeddings",
                   [THESIS / "fig_1_2_umap_embeddings.png", THESIS / "fig_1_4_dirichlet_energy.png"])
        image_page(pdf, "Equity & Efficiency",
                   [THESIS / "fig_2_3_lorenz_curve.png", THESIS / "fig_3_3_latency_profile.png"])

    print(f"✓ PDF report written: {OUT.resolve()}")


if __name__ == "__main__":
    main()
