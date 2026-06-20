"""
SentryMesh — Prediction API (v2, susceptibility models)

Serves the leakage-free flood + landslide susceptibility models for the Flutter
app. Each model is a tabular net: features -> hazard probability (the severity /
alert signal). No graph, no socioeconomic inputs.

Run locally:  uvicorn serve:app --host 0.0.0.0 --port 8000 --reload
Render:       start command  ->  uvicorn serve:app --host 0.0.0.0 --port $PORT
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# torch first (Windows numpy-OpenMP / c10.dll guard)
import torch
import torch.nn as nn

import math
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

CKPT = Path("checkpoints")


class TabularEventNet(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_model(path: Path):
    c = torch.load(path, map_location="cpu", weights_only=False)
    m = TabularEventNet(len(c["feature_cols"]))
    m.load_state_dict(c["model_state"])
    m.eval()
    return {"model": m, "cols": c["feature_cols"],
            "mean": np.asarray(c["scaler_mean"], np.float32),
            "scale": np.asarray(c["scaler_scale"], np.float32),
            "thr": float(c["threshold"]), "metrics": c.get("metrics", {})}


print("Loading SentryMesh models …")
FLOOD = load_model(CKPT / "flood_susceptibility_model.pt")
LAND = (load_model(CKPT / "landslide_susceptibility_model.pt")
        if (CKPT / "landslide_susceptibility_model.pt").exists() else None)
print(f"  flood model loaded (thr={FLOOD['thr']:.2f})"
      + (f"; landslide loaded (thr={LAND['thr']:.2f})" if LAND else "; landslide not found"))

app = FastAPI(title="SentryMesh Prediction API",
              description="Leakage-free flood & landslide susceptibility for ASEAN disaster response",
              version="2.0")


class Node(BaseModel):
    lat: float
    lon: float
    month: int = 1
    rain_1d: float = 0.0
    rain_3d: float = 0.0
    rain_7d: float = 0.0
    rain_30d: float = 0.0
    rain_max3: float = 0.0
    rain_api: float = 0.0
    elev: float = 0.0
    slope_deg: float = 0.0
    relief: float = 0.0
    rp10_risk: float = 0.0
    rp100_risk: float = 0.0
    node_id: Optional[int] = None


class PredictRequest(BaseModel):
    nodes: List[Node]


def _alert_level(p: float, thr: float) -> str:
    return ("CRITICAL" if p >= 0.85 else "HIGH" if p >= 0.65 else
            "MODERATE" if p >= 0.45 else "LOW" if p >= thr else "SAFE")


def _features(n: Node) -> dict:
    return {"rain_1d": n.rain_1d, "rain_3d": n.rain_3d, "rain_7d": n.rain_7d,
            "rain_30d": n.rain_30d, "rain_max3": n.rain_max3, "rain_api": n.rain_api,
            "elev": n.elev, "slope_deg": n.slope_deg, "relief": n.relief,
            "lat": n.lat, "lon": n.lon,
            "month_sin": math.sin(2 * math.pi * n.month / 12),
            "month_cos": math.cos(2 * math.pi * n.month / 12),
            "rp10_risk": n.rp10_risk, "rp100_risk": n.rp100_risk}


def _infer(M: dict, nodes: List[Node]) -> List[dict]:
    if not nodes:
        return []
    X = np.array([[_features(n).get(c, 0.0) for c in M["cols"]] for n in nodes], np.float32)
    X = (X - M["mean"]) / M["scale"]
    with torch.no_grad():
        probs = torch.sigmoid(M["model"](torch.from_numpy(X))).numpy()
    out = []
    for n, p in zip(nodes, probs):
        p = float(p)
        out.append({"node_id": n.node_id, "lat": n.lat, "lon": n.lon,
                    "probability": round(p, 4), "alert": p >= M["thr"],
                    "alert_level": _alert_level(p, M["thr"])})
    # Equity-first: rank purely by threat probability (no socioeconomic input)
    out.sort(key=lambda r: r["probability"], reverse=True)
    for i, r in enumerate(out, 1):
        r["rescue_rank"] = i
    return out


def _respond(M: dict, nodes: List[Node]) -> dict:
    res = _infer(M, nodes)
    return {"threshold": round(M["thr"], 3), "total_nodes": len(res),
            "alert_count": sum(r["alert"] for r in res),
            "ranking_method": "equity_first_threat", "nodes": res}


@app.get("/")
def health():
    return {"status": "ok", "service": "SentryMesh Prediction API v2",
            "models": ["flood"] + (["landslide"] if LAND else [])}


@app.get("/model/info")
def model_info():
    info = {"flood": {"feature_cols": FLOOD["cols"], "threshold": FLOOD["thr"],
                      "metrics": FLOOD["metrics"]}}
    if LAND:
        info["landslide"] = {"feature_cols": LAND["cols"], "threshold": LAND["thr"],
                             "metrics": LAND["metrics"]}
    info["note"] = ("Send antecedent rainfall + terrain per node. Probability is the "
                    "severity/alert signal. rp10_risk/rp100_risk apply to flood only.")
    return info


@app.post("/predict")
@app.post("/predict/flood")
def predict_flood(req: PredictRequest):
    return _respond(FLOOD, req.nodes)


@app.post("/predict/landslide")
def predict_landslide(req: PredictRequest):
    if not LAND:
        return {"error": "landslide model not available"}
    return _respond(LAND, req.nodes)


@app.get("/predict/demo")
def demo():
    """Naga City, Bicol — typhoon vs calm scenarios (no external calls)."""
    nodes = [
        Node(node_id=1, lat=13.62, lon=123.18, month=11, rain_1d=120, rain_3d=260,
             rain_7d=460, rain_30d=700, rain_max3=160, rain_api=180,
             elev=5, slope_deg=0.4, relief=15, rp10_risk=0.07, rp100_risk=0.10),
        Node(node_id=2, lat=13.62, lon=123.18, month=4, rain_1d=3, rain_3d=8,
             rain_7d=20, rain_30d=60, rain_max3=6, rain_api=10,
             elev=5, slope_deg=0.4, relief=15, rp10_risk=0.07, rp100_risk=0.10),
    ]
    return _respond(FLOOD, nodes)
