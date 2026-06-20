"""
SentryMesh — Flood Susceptibility Dataset Builder v2  (Kaggle-ready, self-contained)

Reads data/flood_positives_expanded.csv (318 ASEAN flood positives merged from
DFO masterlist + EM-DAT + GFD, built by prepare_flood_positives.py) and:

  1. Fetches antecedent rainfall (NASA POWER) + terrain (Open-Meteo) per positive
  2. Generates NEG_RATIO× random (location, date) negatives and fetches the same
  3. Writes data/flood_susceptibility.parquet for train_flood_susceptibility.py

Rainfall fetches run in a thread pool (NASA POWER is per-point and latency-bound),
so the whole build takes a few minutes instead of ~40. Fully resumable: positive
features cache to flood_pos_features.parquet and negatives to
flood_susceptibility_neg.parquet — re-run to continue after any interruption.
"""

import random
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from fetch_antecedent_rainfall import _fetch_point, _features
from fetch_terrain import _fetch_elev, DELTA, DIST_M

random.seed(7)
np.random.seed(7)

DATA_ROOT = Path("data")
POS_CSV   = DATA_ROOT / "flood_positives_expanded.csv"
POS_CACHE = DATA_ROOT / "flood_pos_features.parquet"
NEG_CACHE = DATA_ROOT / "flood_susceptibility_neg.parquet"
OUT       = DATA_ROOT / "flood_susceptibility.parquet"

FEAT = ["rain_1d", "rain_3d", "rain_7d", "rain_30d", "rain_max3", "rain_api",
        "elev", "slope_deg", "relief"]
RAIN_KEYS = ["rain_1d", "rain_3d", "rain_7d", "rain_30d", "rain_max3", "rain_api"]
NEG_RATIO = 2
WORKERS = 12          # parallel NASA POWER requests


def fetch_rain_one(lat, lon, date):
    end = pd.Timestamp(date).normalize()
    start = (end - pd.Timedelta(days=35)).strftime("%Y%m%d")
    series = _fetch_point(lat, lon, start, end.strftime("%Y%m%d"))
    return _features(series, end) if series else {k: 0.0 for k in RAIN_KEYS}


def fetch_terrain_batch(chunk):
    """One Open-Meteo request for up to 20 points (5 samples each)."""
    lats, lons = [], []
    for _, r in chunk.iterrows():
        lats += [r.lat, r.lat + DELTA, r.lat - DELTA, r.lat, r.lat]
        lons += [r.lon, r.lon, r.lon, r.lon + DELTA, r.lon - DELTA]
    elev = _fetch_elev(lats, lons) or [0.0] * len(lats)
    out = []
    for i in range(len(chunk)):
        e = elev[i * 5:(i + 1) * 5]
        c = e[0]
        grad = max(abs(c - e[1]), abs(c - e[2]), abs(c - e[3]), abs(c - e[4]))
        out.append({
            "elev": float(c),
            "slope_deg": math.degrees(math.atan(grad / DIST_M)),
            "relief": float(max(e) - min(e)),
        })
    return out


def fetch_points(points: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    """points: [event_id, lat, lon, date] → features (FEAT). Threaded + resumable."""
    results, done = [], set()
    if cache_path.exists():
        prev = pd.read_parquet(cache_path)
        results = prev.to_dict("records")
        done = set(prev["event_id"])
        print(f"  resuming {cache_path.name}: {len(done)} cached")
    todo = points[~points["event_id"].isin(done)].reset_index(drop=True)
    print(f"  fetching {len(todo)} points (workers={WORKERS}) …", flush=True)
    if len(todo) == 0:
        return pd.DataFrame(results)

    # 1) terrain in batches of 20 (cheap, sequential)
    terr_map = {}
    for b in range(0, len(todo), 20):
        chunk = todo.iloc[b:b + 20]
        for (_, r), t in zip(chunk.iterrows(), fetch_terrain_batch(chunk)):
            terr_map[r["event_id"]] = t

    # 2) rainfall in parallel
    def work(row):
        rain = fetch_rain_one(row.lat, row.lon, row.date)
        return {"event_id": row["event_id"], "lat": float(row.lat),
                "lon": float(row.lon), "date": pd.Timestamp(row.date),
                **rain, **terr_map[row["event_id"]]}

    n_done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(work, row) for _, row in todo.iterrows()]
        for fut in as_completed(futures):
            results.append(fut.result())
            n_done += 1
            if n_done % 50 == 0:
                pd.DataFrame(results).to_parquet(cache_path, index=False)
                print(f"    {n_done}/{len(todo)} done (checkpoint)", flush=True)

    pd.DataFrame(results).to_parquet(cache_path, index=False)
    return pd.DataFrame(results)


def gen_negatives(n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "event_id": f"FNEG_{i:05d}",
            "lat": random.uniform(2.0, 26.0),
            "lon": random.uniform(95.0, 140.0),
            "date": pd.Timestamp(random.randint(2000, 2017),
                                 random.randint(1, 12), random.randint(1, 28)),
        })
    return pd.DataFrame(rows)


def main():
    print("Loading expanded positives …", flush=True)
    pos_pts = pd.read_csv(POS_CSV, parse_dates=["date"])
    print(f"  positives: {len(pos_pts)}")

    print("Fetching positive features …", flush=True)
    pos = fetch_points(pos_pts[["event_id", "lat", "lon", "date"]], POS_CACHE)
    pos["label"] = 1

    print("Generating + fetching negatives …", flush=True)
    negs = gen_negatives(len(pos_pts) * NEG_RATIO)
    neg = fetch_points(negs, NEG_CACHE)
    neg["label"] = 0

    cols = ["lat", "lon", "date"] + FEAT + ["label"]
    full = pd.concat([pos[cols], neg[cols]], ignore_index=True)
    t = pd.to_datetime(full["date"])
    full["month_sin"] = np.sin(2 * np.pi * t.dt.month / 12)
    full["month_cos"] = np.cos(2 * np.pi * t.dt.month / 12)
    full.to_parquet(OUT, index=False)

    print(f"\n[OK] flood susceptibility dataset: {len(full)} rows "
          f"({int(full['label'].sum())} pos / {int((full['label'] == 0).sum())} neg) -> {OUT}",
          flush=True)


if __name__ == "__main__":
    main()
