# SentryMesh — VigilantPath AI Severity Engine

> **ASEAN AI Hackathon 2026 · Climate Change Track**
> NCF-ApexNode · Naga College Foundation Inc. · Philippines

An internet-independent disaster-response system powered by a **Spatial-Temporal Graph Neural Network (ST-GNN)** and an **Equity-First rescue prioritization algorithm**. SentryMesh predicts multi-hazard severity across ASEAN sensor meshes and ranks rescue needs purely on objective threat — never on socioeconomic status.

---

## Table of Contents

- [Problem](#problem)
- [AI Architecture](#ai-architecture)
  - [VigilantPath ST-GNN](#1-vigilantpath-st-gnn)
  - [Equity-First Prioritization](#2-equity-first-prioritization)
  - [Why This Architecture](#3-why-this-architecture)
- [Model Performance](#model-performance)
- [Known Limitations](#known-limitations)
- [Ethics & Human Oversight](#ethics--human-oversight)
- [Data Sources](#data-sources)
- [API Reference](#api-reference)
- [Quick Start](#quick-start)
- [Training](#training)
- [Deploy to Render](#deploy-to-render)
- [Flutter Integration](#flutter-integration)

---

## Problem

During Category 4–5 typhoons, **80% of local networks fail within the first 72 hours** — turning manageable disasters into humanitarian crises. Rural, coastal, and mountainous ASEAN communities become communication black holes: rescuers cannot locate victims, and traditional triage systems deprioritize poor communities because they show lower economic exposure value.

SentryMesh solves both problems: LoRaWAN mesh nodes keep data flowing when the internet is gone, and the VigilantPath AI engine ranks rescue priorities by **how dangerous the situation is**, not how valuable the property is.

---

## AI Architecture

### 1. VigilantPath ST-GNN

The core model is a **Spatial-Temporal Graph Neural Network** implemented in PyTorch Geometric. It models the sensor mesh as a live graph where each node is a LoRaWAN sensor and each edge connects sensors within ~0.5° (~55 km).

```
Raw node features (19 dims)
        │
        ▼
┌─────────────────────────┐
│   Spatial Encoder       │  2-layer GCN (GCNConv)
│   GCN hidden: 64        │  learns which neighboring nodes elevate risk
│   GCN out:   32         │
└────────────┬────────────┘
             │  spatial embeddings (N × 32)
             │
┌────────────▼────────────┐
│   Temporal Encoder      │  2-layer GRU, per-node
│   GRU hidden: 64        │  learns how hazard evolves over time window T
│   Time window: T steps  │
└────────────┬────────────┘
             │  temporal embeddings (N × 64)
             │
┌────────────▼────────────┐
│   Fusion + LayerNorm    │  concat → (N × 96), then LayerNorm
└────────────┬────────────┘
             │
     ┌───────┴────────┐
     ▼                ▼
Severity Head     Event Head
MLP → Sigmoid     MLP → Logit
[0, 1] score      binary flood/landslide
```

**Loss function:** Combined weighted MSE (severity regression) + BCE with logits (event classification), with positive-class weight = 10.0 to handle the severe class imbalance in disaster datasets.

**Input features per node (19 total):**

| Feature | Description | Used in Equity-First |
|---|---|---|
| `LAT`, `LON` | Node location | No (spatial only) |
| `month`, `dayofyear`, `hour` | Temporal context | No |
| `hazard_code` | Flood / landslide / typhoon | No |
| `log_exposed` | Log of exposed population value | **Excluded** (socioeconomic proxy) |
| `exposed_area` | Area of exposure (km²) | No |
| `rp10_risk` | 10-year return period risk | **Yes** |
| `rp100_risk` | 100-year return period risk | **Yes** |
| `mean_MNDWI` | Modified Normalized Difference Water Index | No |
| `mean_NDVI` | Vegetation index | No |
| `flood_pixel_frac` | Fraction of flooded satellite pixels | **Yes** |
| `WMO_WIND` | Wind speed (knots) | **Yes** |
| `WMO_PRES` | Atmospheric pressure (hPa) | No |
| `STORM_SPEED` | Storm translation speed | No |
| `DIST2LAND` | Distance to coast (km) | No |
| `elevation` | Node elevation (m) | **Yes** |
| `slope` | Terrain slope (°) | No |

---

### 2. Equity-First Prioritization

After the ST-GNN produces `severity_score` and `event_prob` for each node, a second layer — the **Equity-First algorithm** — computes a composite rescue priority score.

#### Why a separate layer?

The ST-GNN optimizes for prediction accuracy (F1, ROC-AUC). Sorting purely by `event_prob` would still implicitly favor nodes with higher `log_exposed` values — which correlate with wealthier, denser, better-instrumented areas. The Equity-First layer strips that out.

#### The scoring formula

```
equity_score =
    0.35 × event_prob          (model flood/hazard probability)
  + 0.20 × severity_score      (model continuous severity [0,1])
  + 0.15 × flood_pixel_frac    (actual observed water coverage)
  + 0.10 × rp10_risk           (10-year return period risk)
  + 0.10 × rp100_risk          (100-year return period risk)
  + 0.05 × (1 − elevation/2500) (lower elevation = higher flood risk)
  + 0.05 × (WMO_WIND / 120)    (storm intensity, normalized)
```

**Intentionally excluded:** `log_exposed` — this is a log of population economic exposure value and acts as a socioeconomic proxy. Including it would rank wealthier communities higher regardless of actual danger. Equity-First ensures a poor coastal barangay with rising water is ranked above a wealthy low-risk suburb.

#### What the output looks like

Every node in every API response now carries:

```json
{
  "node_id": 702,
  "lat": 14.5995,
  "lon": 120.9842,
  "severity": 0.817,
  "event_prob": 0.912,
  "alert": true,
  "alert_level": "CRITICAL",
  "equity_score": 0.7341,
  "rescue_rank": 1
}
```

`rescue_rank: 1` means this node has the highest Equity-First priority in the current prediction window. All nodes are sorted by `equity_score` descending before being returned.

---

### 3. Why This Architecture

| Design choice | Reason |
|---|---|
| **GCN for spatial** | Flood and landslide risk propagates between neighboring sensor nodes — a flat MLP would miss this. GCN explicitly models "node A is dangerous because its neighbors are flooded." |
| **GRU for temporal** | Hazard severity is not a snapshot — it evolves. A GRU over the time window captures whether water is rising or falling, which a single-timestep model cannot. |
| **Separate equity layer** | Keeping prioritization logic outside the neural network makes it auditable, adjustable, and explainable to disaster-response agencies without retraining the model. |
| **Multi-task head** | Predicting both severity (regression) and event occurrence (classification) jointly improves both tasks through shared representations. |
| **Positive-class weight = 10** | Flood events are rare in the dataset. Without upweighting, the model would learn to always predict "no flood" and still achieve high accuracy. The weight forces it to treat missed detections as costly. |
| **`log_exposed` excluded from equity** | Population exposure value correlates with urban wealth. Using it for rescue ranking would systematically deprioritize rural and indigenous communities. |

---

## Model Performance

These are the results from the held-out test set after training on 165,888 records (1,561 positive disaster events — an 83:1 class imbalance).

| Metric | Score | Notes |
|---|---|---|
| **ROC-AUC** | **0.9443** | Primary indicator for this use case |
| **Recall** | **98.45%** | Nearly all real disasters are detected |
| Accuracy | 88.58% | Overall across all nodes |
| Precision | 15.16% | ~6 false alerts per genuine event |
| F1 Score | 0.263 | Reflects intentional recall-over-precision trade-off |
| Training stopped | Epoch 32 / 40 | Early stopping confirmed efficient convergence |

**Why precision is low by design:** With `pos_weight=10.0` applied to counter the 83:1 class imbalance, the model is deliberately tuned to miss as few real disasters as possible. In a life-safety system, a false alarm that dispatches an unnecessary team is recoverable — a missed disaster is not. ROC-AUC of 0.9443 is the meaningful performance number here, not F1.

---

## Known Limitations

These are honest limitations identified during post-training analysis, documented in our AI-Use & Ethics Report.

**1. Precision trade-off (15.16%)**
The model flags approximately six safe nodes for every genuine event. Every AI-generated alert must be confirmed by a human responder before resources are dispatched. This is a feature, not a bug — it is why human oversight is mandatory at every alert decision point.

**2. Standalone flood/landslide confidence**
Flood and landslide events make up only 1,561 of 165,888 training records. The model is most confident on typhoon-driven compound events. Confidence on standalone flood or landslide scenarios without typhoon context is lower. Collecting more standalone event records is a priority for the next training cycle.

**3. Next steps identified**
- Retrain with `pos_weight=15` and early-stopping patience of 20 epochs
- Validate against additional standalone flood records from NASA LANCE near-real-time data
- Expand ASEAN node coverage beyond Philippines, Indonesia, Vietnam, Thailand

---

## Ethics & Human Oversight

SentryMesh is built on the principle that AI amplifies human capacity but never replaces human judgment. The following intervention points are mandatory, not optional:

| Decision point | Human role |
|---|---|
| Alert dispatch | Responders must confirm before sending rescue teams |
| Community Pulse reports | Flagged-uncertain reports require local responder review before influencing the model |
| Rescue route override | Ground teams can override AI-recommended routes using local knowledge |
| Alert threshold | Emergency coordinators can adjust the threshold in real time |
| Federated model updates | National disaster agencies retain full authority over regional model updates |
| New regional deployment | A human evaluation team must validate Shadow-Mode accuracy against historical records before go-live |

**Equity by design:** The Equity-First algorithm explicitly excludes `log_exposed` (a log of economic exposure value) from rescue ranking. Without this exclusion, the model would implicitly prioritize wealthier, better-instrumented areas. Rescue priority is determined solely by objective threat severity.

**Open-source commitment:** All training data uses Creative Commons or public-domain sources. No personally identifiable information is stored. Community Pulse SMS/USSD reports are opt-in with local-language privacy notices.

---

## Data Sources

| Dataset | Source | Used for |
|---|---|---|
| Global Flood Database (GFD) | Dartmouth Flood Observatory | Flood events, severity labels, satellite indices |
| IBTrACS-style typhoon tracks | `data/typhoon/` | Wind, pressure, storm speed features |
| NASA Global Landslide Catalog | `data/landslide/` | Landslide events and locations |
| GHSL compiled population | `compiled_pop_ghsl_ts_2019_08_04.csv` | `log_exposed` feature (excluded from equity scoring) |
| GFD return period summaries | `gfd_popsummary.csv` | `rp10_risk`, `rp100_risk` features |
| GFD validation points | `gfd_validation_points_2018_12_17.csv` | `mean_MNDWI`, `mean_NDVI`, `flood_pixel_frac` |
| Aqueduct country risk | `aqueductcountrydata.csv` | Baseline country-level water risk |

All datasets are open-access (Creative Commons / public domain). No personally identifiable information is used.

---

## API Reference

Base URL (local): `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/model/info` | Feature columns, thresholds, trained metrics |
| `POST` | `/predict` | Hazard severity prediction for supplied nodes |
| `POST` | `/predict/flood` | Alias for `/predict` |
| `POST` | `/predict/landslide` | Alias for `/predict` |
| `GET/POST` | `/predict/demo` | Demo prediction using 20 real node locations |
| `GET/POST` | `/rescue/priority` | **Equity-First ranked rescue queue (demo data)** |
| `POST` | `/safe-route` | Safe evacuation route between two coordinates |
| `POST` | `/analyze` | AI triage summary for a rescue request |

### POST `/predict` — request body

```json
{
  "nodes": [
    {
      "node_id": 42,
      "features": [14.5, 121.0, 6, 162, 14, 1, 8.2, 7.3, 0.3, 0.6,
                   0.4, 0.2, 0.05, 65, 985, 18, 120, 20, 4],
      "feature_history": null
    }
  ],
  "edge_index": null
}
```

Features must be in the exact order returned by `GET /model/info → feature_cols`. Pass `feature_history` as a `T × 19` array to supply a real time window; omit it to repeat the current reading across all time steps.

### Response (all predict endpoints)

```json
{
  "threshold": 0.35,
  "total_nodes": 20,
  "alert_count": 3,
  "ranking_method": "equity_first",
  "nodes": [
    {
      "node_id": 702,
      "lat": 14.5995,
      "lon": 120.9842,
      "severity": 0.817,
      "event_prob": 0.912,
      "alert": true,
      "alert_level": "CRITICAL",
      "equity_score": 0.7341,
      "rescue_rank": 1
    }
  ]
}
```

**Alert levels:**

| Level | `event_prob` threshold |
|---|---|
| `CRITICAL` | ≥ 0.85 |
| `HIGH` | ≥ 0.65 |
| `MODERATE` | ≥ 0.45 |
| `LOW` | ≥ model threshold |
| `SAFE` | below threshold |

---

## Quick Start

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd final-ai-flood

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the API server
uvicorn serve:app --host 0.0.0.0 --port 8000 --reload
```

Then open:

- `http://localhost:8000/` — health check
- `http://localhost:8000/model/info` — model config and feature order
- `http://localhost:8000/predict/demo` — live demo prediction
- `http://localhost:8000/rescue/priority` — Equity-First ranked rescue queue

---

## Training

To retrain the model from scratch:

```bash
python run_training.py
```

This will:
1. Check and install dependencies (including PyTorch Geometric)
2. Preprocess all data sources → `processed/combined.parquet`
3. Train the ST-GNN → `checkpoints/best_model.pt`
4. Output metrics → `checkpoints/test_metrics.json`

Requires the `data/` folder with all source CSVs. Place `data.zip` in the project root if the folder is missing — the runner will extract it automatically.

---

## Deploy to Render

This repo includes `render.yaml`, `requirements.txt`, and `.python-version`.

1. Push this repository to GitHub.
2. In Render, create a new **Blueprint** from this repo, or create a **Python Web Service** manually.
3. Manual Web Service settings:
   - **Build command:** `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start command:** `uvicorn serve:app --host 0.0.0.0 --port $PORT`
   - **Health check path:** `/`
4. After deploy, use the Render URL as your Flutter API base URL:

```
https://sentrymesh-vigilantpath-api.onrender.com
```

---

## Flutter Integration

Add `http` to `pubspec.yaml`, then:

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

const apiBaseUrl = 'https://sentrymesh-vigilantpath-api.onrender.com';

// Equity-First rescue priority list
Future<Map<String, dynamic>> fetchRescuePriority() async {
  final uri = Uri.parse('$apiBaseUrl/rescue/priority');
  final response = await http.get(uri);
  if (response.statusCode != 200) {
    throw Exception('API error ${response.statusCode}: ${response.body}');
  }
  return jsonDecode(response.body) as Map<String, dynamic>;
}

// Prediction for live sensor nodes
Future<Map<String, dynamic>> predictNodes(List<Map<String, dynamic>> nodes) async {
  final uri = Uri.parse('$apiBaseUrl/predict');
  final response = await http.post(
    uri,
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({'nodes': nodes}),
  );
  if (response.statusCode != 200) {
    throw Exception('API error ${response.statusCode}: ${response.body}');
  }
  return jsonDecode(response.body) as Map<String, dynamic>;
}
```

Call `GET /model/info` first to get the correct `feature_cols` order before building your feature vectors.

---

*Built for the ASEAN AI Hackathon 2026 — Climate Change Track. Targeting SDG 9, SDG 11, and SDG 13.*
