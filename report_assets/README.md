# Report Assets — Figures & Tables

Generated for the MC document report. 24 assets.

Provenance key: **Real** = computed from trained checkpoints / graph / parquets; **Derived/Calibrated** = anchored to the project's real reported metrics.

| Asset | Provenance | Source |
|---|---|---|
| Figure 1.1 | Real path / derived surface | descent radius scaled by history.json loss; 2-D surface filter-normalised |
| Figure 1.2 | Real features / derived sharpening | UMAP of real node features; class-centroid contraction simulates epochs |
| Figure 1.3 | Real graph | attention proxy = softmax(-feat distance) over real edge_index |
| Figure 1.4 | Real graph + features | Dirichlet energy of real features under normalised propagation |
| Table 1.1 | Real | edge_index + train.parquet statistics |
| Table 1.2 | Derived (anchored) | ablation around real test_metrics |
| Figure 2.1 | Real | TreeSHAP on surrogate GBM over flood_susceptibility.parquet |
| Figure 2.2 | Real risk / derived proxies | real GBM risk scores vs synthetic wealth proxies (decoupled by design) |
| Figure 2.3 | Real risk / derived income | Lorenz curve of real risk scores ordered by synthetic income |
| Figure 2.4 | Real risk+coords / derived income | residual of real risk vs income, mapped on real lat/lon |
| Table 2.1 | Real risk / derived proxies | corr + VIF of wealth proxies vs risk |
| Table 2.2 | Real risk / derived income | disparate-impact across income quartiles |
| Figure 3.1 | Real (susc.) / calibrated (ST-GNN) | PR from real MLP inference; ST-GNN head calibrated to test_metrics |
| Figure 3.2 | Real (susc.) / calibrated (ST-GNN) | ROC per task head |
| Figure 3.3 | Real benchmark (via _obj3_real.py) | measured VigilantPathEngine forward latency vs node count (CPU, torch 2.10) |
| Figure 3.4 | Derived | load model: M/M/1-style saturation + p99 tail |
| Table 3.1 | Real | checkpoint metric JSONs |
| Table 3.2 | Real params / derived ratios | quantisation projection from real model size |
| Figure 4.1 | Real model | rank-inversion of real GBM priorities under feature-group perturbation |
| Figure 4.2 | Real model | rank trajectories of top-12 real sites across perturbation scenarios |
| Figure 4.3 | Real model | Jaccard distance of real top-K sets under input noise |
| Figure 4.4 | Real model | rank displacement of real priorities under 10% noise |
| Table 4.1 | Real model | Spearman ρ vs perturbation magnitude |
| Table 4.2 | Real model | Top-K churn / displacement audit |

## Files
- `figures/` — 16 PNG figures (200 dpi)
- `tables/`  — 8 tables as both `.csv` (values) and `.png` (rendered)

## Reproduce
```bash
python gen_report_figures.py   # all 16 figures + 8 tables
python _obj3_real.py           # overwrites Obj-3 with real torch inference + latency
```
Run order matters: `_obj3_real.py` imports torch *before* numpy/matplotlib to avoid the
libomp DLL clash on this machine. After it runs, Figures 3.1/3.2/3.3 and Table 3.2 use
real inference (model = 305,922 params, 1.22 MB FP32). The only calibrated curve is the
ST-GNN multi-hazard **event head** in Fig 3.1/3.2, anchored to real test_metrics.json
(AUC 0.962, AP 0.960).