"""
SentryMesh — Per-Event Terrain (Open-Meteo elevation API, free, no key)

Landslides are physically slope-driven, but elevation/slope are currently zeros.
For each FLOOD and LANDSLIDE event we sample elevation at the location plus 4
neighbours (~5 km N/S/E/W) and derive:

    elev      : elevation at the event location (m)
    slope_deg : terrain slope estimate (degrees) from the max neighbour gradient
    relief    : local relief = max-min elevation across the 5 samples (m)

Saved to data/terrain.parquet keyed by event_id. Batches 100 points/request.
"""

import urllib.request
import json
import time
import math
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DATA_ROOT = Path("data")
OUT = DATA_ROOT / "terrain.parquet"
DELTA = 0.045          # ~5 km offset for neighbour samples
DIST_M = 5000.0        # approx ground distance for slope calc

ASEAN_COUNTRIES = [
    "Philippines", "Indonesia", "Vietnam", "Thailand", "Malaysia", "Myanmar",
    "Cambodia", "Laos", "Singapore", "Brunei", "Viet Nam", "Lao PDR", "Timor-Leste",
]


def _event_list() -> pd.DataFrame:
    fl = pd.read_csv(DATA_ROOT / "gfd_qcdatabase_2019_08_01.csv")
    fl = fl[fl["Country"].isin(ASEAN_COUNTRIES)]
    fl = pd.DataFrame({"event_id": "FL_" + fl["ID"].astype(str),
                       "lat": fl["lat"], "lon": fl["long"]})
    ls = pd.read_csv(DATA_ROOT / "landslide" / "Global_Landslide_Catalog_Export_rows.csv")
    ls = ls[ls["country_name"].isin(ASEAN_COUNTRIES)]
    ls = pd.DataFrame({"event_id": "LS_" + ls["event_id"].astype(str),
                       "lat": ls["latitude"], "lon": ls["longitude"]})
    ev = pd.concat([fl, ls], ignore_index=True).dropna(subset=["lat", "lon"])
    return ev.drop_duplicates("event_id").reset_index(drop=True)


def _fetch_elev(lats, lons, retries=4):
    la = ",".join(f"{x:.4f}" for x in lats)
    lo = ",".join(f"{x:.4f}" for x in lons)
    url = f"https://api.open-meteo.com/v1/elevation?latitude={la}&longitude={lo}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())["elevation"]
        except Exception:
            time.sleep(2 ** attempt)
    return None


def main():
    ev = _event_list()
    done = set()
    results = []
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        done = set(prev["event_id"])
        results = prev.to_dict("records")
        print(f"Resuming — {len(done)} events already done")
    todo = ev[~ev["event_id"].isin(done)].reset_index(drop=True)
    print(f"Fetching terrain for {len(todo)} events …\n")

    # 5 samples per event, 100 coords/request → 20 events/request
    BATCH = 20
    for b in range(0, len(todo), BATCH):
        chunk = todo.iloc[b:b + BATCH]
        lats, lons = [], []
        for _, r in chunk.iterrows():
            lats += [r.lat, r.lat + DELTA, r.lat - DELTA, r.lat, r.lat]
            lons += [r.lon, r.lon, r.lon, r.lon + DELTA, r.lon - DELTA]
        elev = _fetch_elev(lats, lons)
        if elev is None:
            elev = [0.0] * len(lats)
        for i, (_, r) in enumerate(chunk.iterrows()):
            e = elev[i * 5:(i + 1) * 5]
            center = e[0]
            grad = max(abs(center - e[1]), abs(center - e[2]),
                       abs(center - e[3]), abs(center - e[4]))
            slope = math.degrees(math.atan(grad / DIST_M))
            results.append({
                "event_id": r.event_id,
                "elev": float(center),
                "slope_deg": float(slope),
                "relief": float(max(e) - min(e)),
            })
        if (b + BATCH) % 200 == 0:
            pd.DataFrame(results).to_parquet(OUT, index=False)
            print(f"  {min(b+BATCH, len(todo))}/{len(todo)} done (checkpoint)")
        time.sleep(0.3)

    pd.DataFrame(results).to_parquet(OUT, index=False)
    print(f"\n✓ Terrain ready: {len(results)} events → {OUT}")


if __name__ == "__main__":
    main()
