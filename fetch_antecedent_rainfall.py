"""
SentryMesh — Per-Event Antecedent Rainfall (NASA POWER point API)

For every FLOOD and LANDSLIDE event, fetch daily precipitation for the 35 days
BEFORE the event at the event location, then compute antecedent-rainfall
features — the textbook physical trigger for both hazards:

    rain_1d   : rain on the event day
    rain_3d   : total rain in the 3 days before+incl event
    rain_7d   : total rain in the 7 days before
    rain_30d  : total rain in the 30 days before
    rain_max3 : wettest single day in the 3-day window
    rain_api  : antecedent precipitation index (exponentially weighted 14-day)

Saved to data/rainfall/antecedent.parquet keyed by event_id. Resumable: existing
event_ids are skipped, so you can stop/restart freely.

Typhoon is intentionally excluded (pre-1981, handled via OpenWeather).
"""

import urllib.request
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DATA_ROOT = Path("data")
OUT = DATA_ROOT / "rainfall" / "antecedent.parquet"
MISSING = -999.0
WINDOW_DAYS = 35

ASEAN_COUNTRIES = [
    "Philippines", "Indonesia", "Vietnam", "Thailand", "Malaysia", "Myanmar",
    "Cambodia", "Laos", "Singapore", "Brunei", "Viet Nam", "Lao PDR", "Timor-Leste",
]


def _event_list() -> pd.DataFrame:
    """Minimal (event_id, lat, lon, date) for flood + landslide ASEAN events."""
    # Flood
    fl = pd.read_csv(DATA_ROOT / "gfd_qcdatabase_2019_08_01.csv", parse_dates=["Began"])
    fl = fl[fl["Country"].isin(ASEAN_COUNTRIES)]
    fl = pd.DataFrame({
        "event_id": "FL_" + fl["ID"].astype(str),
        "lat": fl["lat"], "lon": fl["long"], "date": pd.to_datetime(fl["Began"], errors="coerce"),
    })
    # Landslide
    ls = pd.read_csv(DATA_ROOT / "landslide" / "Global_Landslide_Catalog_Export_rows.csv",
                     parse_dates=["event_date"])
    ls = ls[ls["country_name"].isin(ASEAN_COUNTRIES)]
    ls = pd.DataFrame({
        "event_id": "LS_" + ls["event_id"].astype(str),
        "lat": ls["latitude"], "lon": ls["longitude"],
        "date": pd.to_datetime(ls["event_date"], errors="coerce"),
    })
    ev = pd.concat([fl, ls], ignore_index=True)
    ev = ev.dropna(subset=["lat", "lon", "date"])
    # NASA POWER starts 1981
    ev = ev[ev["date"] >= "1981-01-02"]
    return ev.drop_duplicates("event_id").reset_index(drop=True)


def _fetch_point(lat, lon, start, end, retries=4):
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point?"
        "parameters=PRECTOTCORR&community=AG"
        f"&longitude={lon:.4f}&latitude={lat:.4f}"
        f"&start={start}&end={end}&format=JSON"
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())["properties"]["parameter"]["PRECTOTCORR"]
        except Exception as e:
            time.sleep(2 ** attempt)
    return None


def _features(series: dict, event_date: pd.Timestamp) -> dict:
    s = pd.Series(series, dtype="float64").replace(MISSING, np.nan)
    s.index = pd.to_datetime(s.index, format="%Y%m%d")
    s = s.sort_index().fillna(0.0)
    end = event_date.normalize()
    def total(days):
        win = s[(s.index > end - pd.Timedelta(days=days)) & (s.index <= end)]
        return float(win.sum())
    # exponentially weighted antecedent precip index (k=0.9, 14 days)
    api = 0.0
    for d in range(14, 0, -1):
        day = end - pd.Timedelta(days=d)
        if day in s.index:
            api = api * 0.9 + float(s.loc[day])
    win3 = s[(s.index > end - pd.Timedelta(days=3)) & (s.index <= end)]
    return {
        "rain_1d":   float(s.get(end, 0.0)),
        "rain_3d":   total(3),
        "rain_7d":   total(7),
        "rain_30d":  total(30),
        "rain_max3": float(win3.max()) if len(win3) else 0.0,
        "rain_api":  api,
    }


def main():
    ev = _event_list()
    done = set()
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        done = set(prev["event_id"])
        results = prev.to_dict("records")
        print(f"Resuming — {len(done)} events already fetched")
    else:
        results = []

    todo = ev[~ev["event_id"].isin(done)].reset_index(drop=True)
    print(f"Fetching antecedent rainfall for {len(todo)} events …\n")

    for i, row in todo.iterrows():
        end = row["date"].normalize()
        start = (end - pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y%m%d")
        series = _fetch_point(row["lat"], row["lon"], start, end.strftime("%Y%m%d"))
        feats = _features(series, row["date"]) if series else {
            k: 0.0 for k in ["rain_1d", "rain_3d", "rain_7d", "rain_30d", "rain_max3", "rain_api"]
        }
        results.append({"event_id": row["event_id"], **feats})

        if (i + 1) % 50 == 0:
            pd.DataFrame(results).to_parquet(OUT, index=False)
            print(f"  {i+1}/{len(todo)} done  (checkpoint saved)")
        time.sleep(0.25)

    pd.DataFrame(results).to_parquet(OUT, index=False)
    print(f"\n✓ Antecedent rainfall ready: {len(results)} events → {OUT}")


if __name__ == "__main__":
    main()
