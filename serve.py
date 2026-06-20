"""
SentryMesh — Prediction API (v2)

Serves the leakage-free flood + landslide susceptibility models to the NestJS
backend (apexnode-ai.onrender.com contract). The backend sends a location; this
service fetches LIVE antecedent rainfall + terrain itself (Open-Meteo, no key)
and returns hazard probability — the severity / alert signal.

Endpoints (match the backend's AI_* paths):
    GET  /                 health
    GET  /model/info       feature cols + metrics
    POST /predict          flood prediction (alias /predict/flood)
    POST /predict/flood    "
    POST /predict/landslide landslide prediction
    GET  /predict/demo     Naga City live demo
    POST /safe-route       great-circle evac route with hazard risk
    POST /analyze          rescue-request triage summary

Run:    uvicorn serve:app --host 0.0.0.0 --port 8000 --reload
Render: uvicorn serve:app --host 0.0.0.0 --port $PORT
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import torch.nn as nn

import json
import math
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Any, Dict

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

CKPT = Path("checkpoints")
# Philippines Aqueduct return-period priors (deployment target = Bicol/PH)
PH_RP10, PH_RP100 = 0.07, 0.10
OM_TIMEOUT = 12


# ─────────────────────────────── model ──────────────────────────────────────
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
print(f"  flood (thr={FLOOD['thr']:.2f})" + (f"; landslide (thr={LAND['thr']:.2f})" if LAND else ""))


# ───────────────────────── live feature fetch (Open-Meteo) ──────────────────
def _get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "SentryMesh/2.0"})
    with urllib.request.urlopen(req, timeout=OM_TIMEOUT) as r:
        return json.loads(r.read())


def fetch_rainfall(lat: float, lon: float) -> Dict[str, float]:
    """Last ~35 days of daily precip ending today (real-time), via Open-Meteo."""
    keys = ["rain_1d", "rain_3d", "rain_7d", "rain_30d", "rain_max3", "rain_api"]
    try:
        url = ("https://api.open-meteo.com/v1/forecast?"
               + urllib.parse.urlencode({"latitude": lat, "longitude": lon,
                                         "daily": "precipitation_sum",
                                         "past_days": 35, "forecast_days": 1,
                                         "timezone": "auto"}))
        arr = [v or 0.0 for v in _get_json(url)["daily"]["precipitation_sum"]]
        if not arr:
            return {k: 0.0 for k in keys}
        api = 0.0
        for v in arr[-15:-1]:            # 14 antecedent days
            api = api * 0.9 + float(v)
        return {"rain_1d": float(arr[-1]), "rain_3d": float(sum(arr[-3:])),
                "rain_7d": float(sum(arr[-7:])), "rain_30d": float(sum(arr[-30:])),
                "rain_max3": float(max(arr[-3:])), "rain_api": api}
    except Exception:
        return {k: 0.0 for k in keys}


def fetch_terrain(lat: float, lon: float) -> Dict[str, float]:
    """Elevation + crude slope/relief from a 5-point stencil, via Open-Meteo."""
    try:
        d = 0.045
        la = [lat, lat + d, lat - d, lat, lat]
        lo = [lon, lon, lon, lon + d, lon - d]
        url = ("https://api.open-meteo.com/v1/elevation?"
               + urllib.parse.urlencode({"latitude": ",".join(f"{x:.4f}" for x in la),
                                         "longitude": ",".join(f"{x:.4f}" for x in lo)}))
        e = _get_json(url)["elevation"]
        c = e[0]
        grad = max(abs(c - e[1]), abs(c - e[2]), abs(c - e[3]), abs(c - e[4]))
        return {"elev": float(c), "slope_deg": math.degrees(math.atan(grad / 5000.0)),
                "relief": float(max(e) - min(e))}
    except Exception:
        return {"elev": 10.0, "slope_deg": 0.5, "relief": 10.0}


# ───────────────────────────── request parsing ──────────────────────────────
def _num(v, default=None):
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def extract_latlon(node: Dict[str, Any]):
    """Backend sends {features:[LAT,LON,...]} OR {lat/lon}. Handle both."""
    feats = node.get("features")
    if isinstance(feats, list) and len(feats) >= 2:
        return _num(feats[0]), _num(feats[1])
    lat = _num(node.get("lat") or node.get("latitude"))
    lon = _num(node.get("lon") or node.get("lng") or node.get("longitude"))
    return lat, lon


def _alert_level(p: float, thr: float) -> str:
    return ("CRITICAL" if p >= 0.85 else "HIGH" if p >= 0.65 else
            "MODERATE" if p >= 0.45 else "LOW" if p >= thr else "SAFE")


def equity_score(p: float, rain: Dict[str, float], terr: Dict[str, float],
                 rp100: float) -> float:
    """Equity-First rescue priority — THREAT factors only, NO socioeconomic input.

    Population / wealth / exposure are deliberately excluded so a poor low-lying
    barangay with rising water outranks a wealthy low-risk area. All terms are
    objective hazard signals, normalised to [0, 1].
    """
    rainfall_intensity = min(rain.get("rain_7d", 0.0) / 200.0, 1.0)
    low_elevation = 1.0 - min(max(terr.get("elev", 0.0), 0.0), 500.0) / 500.0
    rp = min(max(rp100, 0.0), 1.0)
    return (0.55 * p + 0.20 * rainfall_intensity
            + 0.15 * low_elevation + 0.10 * rp)


def _build_feature_vector(M, lat, lon, month, rain, terr, rp10, rp100):
    vals = {**rain, **terr, "lat": lat, "lon": lon,
            "month_sin": math.sin(2 * math.pi * month / 12),
            "month_cos": math.cos(2 * math.pi * month / 12),
            "rp10_risk": rp10, "rp100_risk": rp100}
    return np.array([vals.get(c, 0.0) for c in M["cols"]], np.float32)


def run_prediction(M, body: Dict[str, Any]) -> Dict[str, Any]:
    import datetime as dt
    month = dt.datetime.now().month
    raw_nodes = body.get("nodes") if isinstance(body, dict) else None
    if not raw_nodes:
        raw_nodes = [body if isinstance(body, dict) else {}]

    rows = []
    for i, node in enumerate(raw_nodes):
        lat, lon = extract_latlon(node)
        if lat is None or lon is None:
            continue
        rain = fetch_rainfall(lat, lon)
        terr = fetch_terrain(lat, lon)
        x = _build_feature_vector(M, lat, lon, month, rain, terr, PH_RP10, PH_RP100)
        xs = (x.reshape(1, -1) - M["mean"]) / M["scale"]
        with torch.no_grad():
            p = float(torch.sigmoid(M["model"](torch.from_numpy(xs.astype(np.float32)))).item())
        equity = equity_score(p, rain, terr, PH_RP100)
        rows.append({"node_id": node.get("node_id", i), "lat": lat, "lon": lon,
                     "event_prob": round(p, 4), "probability": round(p, 4),
                     "severity": round(p, 4), "alert": p >= M["thr"],
                     "alert_level": _alert_level(p, M["thr"]),
                     "equity_score": round(equity, 4),
                     "rainfall": {k: round(v, 1) for k, v in rain.items()},
                     "elevation": round(terr["elev"], 1)})

    # Equity-first: rank by threat-only equity score (NO socioeconomic input)
    rows.sort(key=lambda r: r["equity_score"], reverse=True)
    for rank, r in enumerate(rows, 1):
        r["rescue_rank"] = rank
    return {"threshold": round(M["thr"], 3), "total_nodes": len(rows),
            "alert_count": sum(r["alert"] for r in rows),
            "ranking_method": "equity_first_threat", "nodes": rows}


# ───────────────────────────────── app ──────────────────────────────────────
app = FastAPI(title="SentryMesh Prediction API", version="2.0",
              description="Leakage-free flood & landslide susceptibility (live rainfall).")


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
    info["note"] = "Send {nodes:[{lat,lon}]} or {latitude,longitude}; rainfall/terrain fetched live."
    return info


@app.post("/predict")
@app.post("/predict/flood")
def predict_flood(body: Dict[str, Any]):
    return run_prediction(FLOOD, body)


@app.post("/predict/landslide")
def predict_landslide(body: Dict[str, Any]):
    if not LAND:
        return {"prediction_available": False, "message": "landslide model not loaded"}
    return run_prediction(LAND, body)


@app.get("/predict/demo")
def demo():
    return run_prediction(FLOOD, {"nodes": [{"node_id": 1, "lat": 13.62, "lon": 123.18}]})


# ───────────────────────── safe-route + analyze (backend contract) ───────────
class Point(BaseModel):
    latitude: float
    longitude: float


class SafeRouteRequest(BaseModel):
    origin: Point
    destination: Point


def _haversine_km(a: Point, b: Point) -> float:
    R = 6371.0
    p1, p2 = math.radians(a.latitude), math.radians(b.latitude)
    dp = math.radians(b.latitude - a.latitude)
    dl = math.radians(b.longitude - a.longitude)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


@app.post("/safe-route")
def safe_route(req: SafeRouteRequest):
    dist = _haversine_km(req.origin, req.destination)
    n = 6
    waypoints = [{"latitude": req.origin.latitude + (req.destination.latitude - req.origin.latitude) * t / n,
                  "longitude": req.origin.longitude + (req.destination.longitude - req.origin.longitude) * t / n}
                 for t in range(n + 1)]
    # hazard risk at the midpoint
    mid = waypoints[n // 2]
    pred = run_prediction(FLOOD, {"nodes": [{"lat": mid["latitude"], "lon": mid["longitude"]}]})
    risk = pred["nodes"][0]["alert_level"] if pred["nodes"] else "UNKNOWN"
    return {"label": "AI safe route", "distance_km": round(dist, 2),
            "estimated_minutes": round(dist / 30 * 60, 1), "risk_level": risk,
            "waypoints": waypoints}


@app.post("/analyze")
def analyze(body: Dict[str, Any]):
    lat = _num(body.get("latitude") or body.get("lat"))
    lon = _num(body.get("longitude") or body.get("lon") or body.get("lng"))
    people = body.get("people_count") or body.get("people") or "unknown number of"
    hazard = body.get("hazard_type") or body.get("hazard") or "hazard"
    risk = "unknown"
    if lat is not None and lon is not None:
        pred = run_prediction(FLOOD, {"nodes": [{"lat": lat, "lon": lon}]})
        if pred["nodes"]:
            risk = pred["nodes"][0]["alert_level"]
    summary = (f"Rescue request involving {people} people under {hazard} conditions. "
               f"Live flood risk at the location is {risk}. "
               + ("Prioritize immediate evacuation and confirm with a responder."
                  if risk in ("CRITICAL", "HIGH") else
                  "Monitor conditions; dispatch on responder confirmation."))
    return {"summary": summary, "risk_level": risk}
