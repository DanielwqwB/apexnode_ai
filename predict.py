"""
SentryMesh — Flood Susceptibility Predictor (real-time / typhoon test)

Loads the trained hard-negative flood model and returns flood PROBABILITY (which
is the severity/alert signal) for a location given antecedent rainfall + terrain.

Usage (programmatic):
    from predict import predict_flood
    out = predict_flood(lat=14.6, lon=121.0, month=8,
                        rain_1d=80, rain_3d=180, rain_7d=260, rain_30d=400,
                        rain_max3=120, rain_api=150,
                        elev=15, slope_deg=1.2, relief=40,
                        rp10_risk=0.07, rp100_risk=0.10)
    print(out)   # {'flood_probability':0.93,'alert':True,'alert_level':'CRITICAL',...}

Quick demo:
    python predict.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import torch.nn as nn

import numpy as np

CKPT = "checkpoints/flood_susceptibility_model.pt"
FEAT = ["rain_1d", "rain_3d", "rain_7d", "rain_30d", "rain_max3", "rain_api",
        "elev", "slope_deg", "relief", "lat", "lon", "month_sin", "month_cos",
        "rp10_risk", "rp100_risk"]


class TabularEventNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


_ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
_model = TabularEventNet(len(_ckpt["feature_cols"]))
_model.load_state_dict(_ckpt["model_state"])
_model.eval()
_mean = np.asarray(_ckpt["scaler_mean"], np.float32)
_scale = np.asarray(_ckpt["scaler_scale"], np.float32)
_thr = float(_ckpt["threshold"])
_cols = _ckpt["feature_cols"]


def _alert_level(p):
    return ("CRITICAL" if p >= 0.85 else "HIGH" if p >= 0.65 else
            "MODERATE" if p >= 0.45 else "LOW" if p >= _thr else "SAFE")


def predict_flood(lat, lon, month, rain_1d=0.0, rain_3d=0.0, rain_7d=0.0,
                  rain_30d=0.0, rain_max3=0.0, rain_api=0.0, elev=0.0,
                  slope_deg=0.0, relief=0.0, rp10_risk=0.0, rp100_risk=0.0):
    vals = {"rain_1d": rain_1d, "rain_3d": rain_3d, "rain_7d": rain_7d,
            "rain_30d": rain_30d, "rain_max3": rain_max3, "rain_api": rain_api,
            "elev": elev, "slope_deg": slope_deg, "relief": relief,
            "lat": lat, "lon": lon,
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "rp10_risk": rp10_risk, "rp100_risk": rp100_risk}
    x = np.array([[vals[c] for c in _cols]], np.float32)
    x = (x - _mean) / _scale
    with torch.no_grad():
        p = torch.sigmoid(_model(torch.from_numpy(x))).item()
    return {"flood_probability": round(p, 4), "threshold": round(_thr, 3),
            "alert": p >= _thr, "alert_level": _alert_level(p)}


if __name__ == "__main__":
    print("Manila, typhoon-level antecedent rain (Aug):")
    print(" ", predict_flood(lat=14.6, lon=121.0, month=8, rain_1d=85, rain_3d=190,
                             rain_7d=270, rain_30d=420, rain_max3=120, rain_api=160,
                             elev=12, slope_deg=1.1, relief=35,
                             rp10_risk=0.07, rp100_risk=0.10))
    print("Same place, dry-season normal rain (Mar):")
    print(" ", predict_flood(lat=14.6, lon=121.0, month=3, rain_1d=2, rain_3d=5,
                             rain_7d=12, rain_30d=40, rain_max3=4, rain_api=8,
                             elev=12, slope_deg=1.1, relief=35,
                             rp10_risk=0.07, rp100_risk=0.10))
