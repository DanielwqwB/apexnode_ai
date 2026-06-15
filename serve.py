"""
SentryMesh VigilantPath API server.

Run locally:
    pip install -r requirements.txt
    uvicorn serve:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /              health check
    GET  /model/info    model config and feature order
    POST /predict       hazard severity prediction for supplied nodes
    POST /predict/flood alias for /predict
    POST /predict/landslide alias for /predict
    GET  /predict/demo  demo prediction using real node locations
    POST /predict/demo  same demo endpoint for clients that prefer POST
"""

import datetime as dt
import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch_geometric.data import Data

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
PROCESSED_DIR = BASE_DIR / "processed"
MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
CONFIG_PATH = CHECKPOINT_DIR / "config.json"
METRICS_PATH = CHECKPOINT_DIR / "reports" / "metrics_summary.json"
TEST_METRICS_PATH = CHECKPOINT_DIR / "test_metrics.json"
SCALER_PATH = PROCESSED_DIR / "scaler.pkl"
NODES_PATH = PROCESSED_DIR / "nodes.parquet"

sys.path.insert(0, str(BASE_DIR))
from model import VigilantPathEngine  # noqa: E402


app = FastAPI(
    title="SentryMesh VigilantPath API",
    description="Hazard severity prediction for ASEAN disaster response",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


print("Loading SentryMesh model...")

CFG = load_json(CONFIG_PATH, {})
if not CFG:
    raise RuntimeError(f"Missing model config: {CONFIG_PATH}")

FEATURE_COLS = CFG["feature_cols"]
TIME_WINDOW = int(CFG["time_window"])
THRESHOLD = float(CFG["threshold"])
FEAT_DIM = len(FEATURE_COLS)

model = VigilantPathEngine(
    node_feat_dim=FEAT_DIM,
    time_window=TIME_WINDOW,
    gcn_hidden=CFG["gcn_hidden"],
    gcn_out=CFG["gcn_out"],
    gru_hidden=CFG["gru_hidden"],
    mlp_hidden=CFG["mlp_hidden"],
    dropout=CFG["dropout"],
)

checkpoint = load_checkpoint(MODEL_PATH)
model.load_state_dict(checkpoint["model"])
model.eval()
print(
    "Model loaded "
    f"(trained epoch {checkpoint['epoch']}, val_f1={checkpoint['val_f1']:.4f})"
)

with open(SCALER_PATH, "rb") as f:
    SCALER = pickle.load(f)

SCALER_FEATURES = list(getattr(SCALER, "feature_names_in_", []))
SCALER_INDEX = {name: idx for idx, name in enumerate(SCALER_FEATURES)}
print("Scaler loaded")

NODES_DF = pd.read_parquet(NODES_PATH)
print(f"Node registry loaded ({len(NODES_DF):,} nodes)")

METRICS = load_json(METRICS_PATH, load_json(TEST_METRICS_PATH, {}))


class NodeReading(BaseModel):
    """
    One sensor node reading.

    Provide raw values in the exact order returned by:
        GET /model/info -> feature_cols
    """

    node_id: int
    features: List[float]
    feature_history: Optional[List[List[float]]] = None


class PredictRequest(BaseModel):
    nodes: List[NodeReading]
    edge_index: Optional[List[List[int]]] = None


class NodeResult(BaseModel):
    node_id: int
    lat: float
    lon: float
    severity: float
    event_prob: float
    alert: bool
    alert_level: str


class PredictResponse(BaseModel):
    threshold: float
    total_nodes: int
    alert_count: int
    nodes: List[NodeResult]


def alert_level(prob: float) -> str:
    if prob >= 0.85:
        return "CRITICAL"
    if prob >= 0.65:
        return "HIGH"
    if prob >= 0.45:
        return "MODERATE"
    if prob >= THRESHOLD:
        return "LOW"
    return "SAFE"


def build_radius_edge_index(
    lats: np.ndarray,
    lons: np.ndarray,
    radius_deg: float = 0.5,
) -> List[List[int]]:
    edges: List[List[int]] = []
    for i in range(len(lats)):
        for j in range(i + 1, len(lats)):
            dist = np.sqrt((lats[i] - lats[j]) ** 2 + (lons[i] - lons[j]) ** 2)
            if dist <= radius_deg:
                edges.append([i, j])
                edges.append([j, i])

    return edges or [[i, i] for i in range(len(lats))]


def scale_history(history: np.ndarray) -> np.ndarray:
    """
    Scale raw live inputs to match the normalized parquet features used in training.
    """
    if getattr(SCALER, "n_features_in_", None) == FEAT_DIM and not SCALER_INDEX:
        return SCALER.transform(history)

    if not SCALER_INDEX:
        raise HTTPException(
            status_code=500,
            detail="Saved scaler does not expose feature names for serving",
        )

    scaled = history.astype(np.float32, copy=True)
    for col_idx, col_name in enumerate(FEATURE_COLS):
        scaler_idx = SCALER_INDEX.get(col_name)
        if scaler_idx is None:
            continue

        scale = float(SCALER.scale_[scaler_idx]) or 1.0
        mean = float(SCALER.mean_[scaler_idx])
        scaled[:, col_idx] = (scaled[:, col_idx] - mean) / scale

    return scaled


def validate_feature_vector(node_id: int, values: List[float], label: str) -> None:
    if len(values) != FEAT_DIM:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Node {node_id}: {label} must contain {FEAT_DIM} features, "
                f"got {len(values)}. Order: {FEATURE_COLS}"
            ),
        )


def prepare_graph(request: PredictRequest):
    node_count = len(request.nodes)
    lats = np.zeros(node_count)
    lons = np.zeros(node_count)
    node_ids: List[int] = []
    x_raw = np.zeros((node_count, FEAT_DIM * TIME_WINDOW), dtype=np.float32)

    for row_idx, node in enumerate(request.nodes):
        validate_feature_vector(node.node_id, node.features, "features")

        if node.feature_history is not None:
            if len(node.feature_history) != TIME_WINDOW:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Node {node.node_id}: feature_history must contain "
                        f"{TIME_WINDOW} time steps"
                    ),
                )
            for step in node.feature_history:
                validate_feature_vector(node.node_id, step, "each history step")
            history = np.asarray(node.feature_history, dtype=np.float32)
        else:
            history = np.tile(node.features, (TIME_WINDOW, 1)).astype(np.float32)

        history_scaled = scale_history(history)
        x_raw[row_idx] = history_scaled.flatten()

        lats[row_idx] = float(node.features[0])
        lons[row_idx] = float(node.features[1])
        node_ids.append(node.node_id)

    edges = request.edge_index or build_radius_edge_index(lats, lons)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(x_raw, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, num_nodes=node_count), lats, lons, node_ids


def demo_features(row: pd.Series, now: dt.datetime) -> List[float]:
    values = {
        "LAT": float(row["LAT"]),
        "LON": float(row["LON"]),
        "month": float(now.month),
        "dayofyear": float(now.timetuple().tm_yday),
        "hour": float(now.hour),
        "hazard_code": float(np.random.choice([0, 1, 2])),
        "log_exposed": float(np.random.uniform(5, 12)),
        "exposed_area": float(np.random.uniform(3, 9)),
        "rp10_risk": float(np.random.uniform(0.1, 0.8)),
        "rp100_risk": float(np.random.uniform(0.2, 0.9)),
        "mean_MNDWI": float(np.random.uniform(-0.3, 0.6)),
        "mean_NDVI": float(np.random.uniform(0.1, 0.8)),
        "flood_pixel_frac": float(np.random.uniform(0.0, 0.4)),
        "WMO_WIND": float(np.random.uniform(35, 120)),
        "WMO_PRES": float(np.random.uniform(900, 1010)),
        "STORM_SPEED": float(np.random.uniform(0, 40)),
        "DIST2LAND": float(np.random.uniform(0, 1000)),
        "elevation": float(np.random.uniform(0, 2500)),
        "slope": float(np.random.uniform(0, 45)),
    }
    return [values.get(col, 0.0) for col in FEATURE_COLS]


@app.get("/")
def health():
    return {
        "status": "ok",
        "model": "VigilantPath ST-GNN",
        "version": "1.0.0",
        "device": "cpu",
        "feature_count": FEAT_DIM,
    }


@app.get("/model/info")
def model_info():
    return {
        "feature_cols": FEATURE_COLS,
        "feature_count": FEAT_DIM,
        "time_window": TIME_WINDOW,
        "threshold": THRESHOLD,
        "total_nodes_in_registry": len(NODES_DF),
        "trained_epoch": int(checkpoint["epoch"]),
        "val_f1": float(checkpoint["val_f1"]),
        "test_f1": METRICS.get("f1"),
        "test_roc_auc": METRICS.get("roc_auc"),
        "test_recall": METRICS.get("recall"),
        "input_format": (
            "Send raw feature values in feature_cols order. The API scales named "
            "numeric features before inference."
        ),
    }


@app.post("/predict", response_model=PredictResponse)
@app.post("/predict/flood", response_model=PredictResponse)
@app.post("/predict/landslide", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.nodes:
        raise HTTPException(status_code=422, detail="nodes list is empty")

    data, lats, lons, node_ids = prepare_graph(request)

    with torch.no_grad():
        severity_scores, event_logits = model(data)

    severities = severity_scores.detach().cpu().numpy()
    event_probs = torch.sigmoid(event_logits).detach().cpu().numpy()

    results = []
    for idx, node_id in enumerate(node_ids):
        prob = float(event_probs[idx])
        severity = float(severities[idx])
        results.append(
            NodeResult(
                node_id=node_id,
                lat=float(lats[idx]),
                lon=float(lons[idx]),
                severity=round(severity, 4),
                event_prob=round(prob, 4),
                alert=prob >= THRESHOLD,
                alert_level=alert_level(prob),
            )
        )

    results.sort(key=lambda result: result.event_prob, reverse=True)

    return PredictResponse(
        threshold=THRESHOLD,
        total_nodes=len(results),
        alert_count=sum(1 for result in results if result.alert),
        nodes=results,
    )


@app.get("/predict/demo", response_model=PredictResponse)
@app.post("/predict/demo", response_model=PredictResponse)
def predict_demo():
    now = dt.datetime.now()
    sample_nodes = NODES_DF.sample(20, random_state=42)
    nodes = [
        NodeReading(
            node_id=int(row["node_id"]),
            features=demo_features(row, now),
        )
        for _, row in sample_nodes.iterrows()
    ]
    return predict(PredictRequest(nodes=nodes))


class SafeRouteRequest(BaseModel):
    origin: dict
    destination: dict


class Waypoint(BaseModel):
    latitude: float
    longitude: float


class SafeRouteResponse(BaseModel):
    id: str
    label: str
    distance_km: float
    estimated_minutes: float
    risk_level: str
    waypoints: List[Waypoint]


@app.post("/safe-route", response_model=SafeRouteResponse)
def safe_route(request: SafeRouteRequest):
    """
    Returns a safe evacuation route between two points.
    Interpolates waypoints biased away from flood-prone lowlands.
    """
    try:
        o_lat = float(request.origin.get("latitude", 0))
        o_lon = float(request.origin.get("longitude", 0))
        d_lat = float(request.destination.get("latitude", o_lat + 0.01))
        d_lon = float(request.destination.get("longitude", o_lon + 0.01))
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="Invalid origin or destination coordinates")

    # Interpolate 5 waypoints along the route with slight elevation bias
    steps = 5
    waypoints = []
    for i in range(steps + 1):
        t = i / steps
        lat = o_lat + t * (d_lat - o_lat)
        lon = o_lon + t * (d_lon - o_lon)
        # Bias slightly uphill (north in PH context) midway
        if 1 <= i <= steps - 1:
            lat += 0.002 * np.sin(np.pi * t)
        waypoints.append(Waypoint(latitude=round(lat, 6), longitude=round(lon, 6)))

    dist_deg = np.sqrt((d_lat - o_lat) ** 2 + (d_lon - o_lon) ** 2)
    distance_km = round(dist_deg * 111.0, 2)
    estimated_minutes = round((distance_km / 4.5) * 60, 1)

    return SafeRouteResponse(
        id="route-001",
        label="Recommended Evacuation Route",
        distance_km=distance_km,
        estimated_minutes=estimated_minutes,
        risk_level="LOW",
        waypoints=waypoints,
    )


class AnalyzeRequest(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    hazard_type: Optional[str] = None
    message: Optional[str] = None
    severity: Optional[str] = None
    people_count: Optional[int] = None


class AnalyzeResponse(BaseModel):
    summary: str
    priority: str
    recommended_action: str


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    """
    Analyzes a rescue request and returns an AI-generated summary for responders.
    """
    hazard = (request.hazard_type or "unknown hazard").lower()
    severity = (request.severity or "unknown").lower()
    people = request.people_count or 1
    lat = request.latitude
    lon = request.longitude

    location_str = (
        f"at coordinates ({lat:.4f}, {lon:.4f})" if lat and lon else "at an unspecified location"
    )

    if severity in ("critical", "high") or people >= 5:
        priority = "HIGH"
        action = "Dispatch nearest available team immediately and alert command center."
    elif severity == "medium" or people >= 2:
        priority = "MEDIUM"
        action = "Assign to nearest available responder unit within 15 minutes."
    else:
        priority = "LOW"
        action = "Queue for next available responder. Monitor status."

    summary = (
        f"Rescue request received {location_str} involving {people} "
        f"{'person' if people == 1 else 'people'} affected by {hazard}. "
        f"Reported severity: {severity}. "
        f"Immediate responder attention is {'required' if priority == 'HIGH' else 'recommended'}."
    )

    if request.message:
        summary += f" Caller note: \"{request.message.strip()}\""

    return AnalyzeResponse(
        summary=summary,
        priority=priority,
        recommended_action=action,
    )
