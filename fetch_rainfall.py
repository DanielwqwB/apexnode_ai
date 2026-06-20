"""
SentryMesh — NASA POWER Rainfall Fetcher
Downloads daily precipitation (PRECTOTCORR, mm/day) for the ASEAN region from
the NASA POWER regional API, aggregates to MONTHLY per 0.5° grid cell, and
saves data/rainfall/rainfall_monthly.parquet for use by data_loader.

NOTE: Typhoon events (pre-1987) are intentionally NOT covered here — NASA POWER
starts in 1981 and typhoon rainfall is handled separately via OpenWeather.
This fetch targets the flood (2000–2018) and landslide (1988–2017) windows.

Constraints discovered empirically:
  - regional endpoint allows up to ~10°×10° per request
  - regional endpoint caps the date range at 1 year per request
So we tile space into 10° boxes and loop one year at a time. Resumable: each
spatial tile is cached to data/rainfall/tiles/ and skipped if already present.
"""

import urllib.request
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

# ── Region (matches ASEAN_LAT/ASEAN_LON in data_loader.py) ──
LAT_MIN, LAT_MAX = 0.0, 28.0
LON_MIN, LON_MAX = 90.0, 145.0
YEAR_START, YEAR_END = 2000, 2018      # flood + landslide coverage window
TILE = 10                              # degrees per spatial tile (API limit ~10)
CELL = 0.5                             # must match CELL in data_loader.py
MISSING = -999.0

OUT_DIR  = Path("data/rainfall")
TILE_DIR = OUT_DIR / "tiles"
OUT_FILE = OUT_DIR / "rainfall_monthly.parquet"


def _fetch(la0, la1, lo0, lo1, year, retries=4):
    """Fetch one spatial tile for one year; return list of features or None."""
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/regional?"
        "parameters=PRECTOTCORR&community=AG"
        f"&latitude-min={la0}&latitude-max={la1}"
        f"&longitude-min={lo0}&longitude-max={lo1}"
        f"&start={year}0101&end={year}1231&format=JSON"
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return json.loads(r.read())["features"]
        except Exception as e:
            wait = 2 ** attempt
            print(f"    retry {attempt+1}/{retries} ({e}) — waiting {wait}s")
            time.sleep(wait)
    print(f"    GAVE UP on tile ({la0},{lo0}) {year}")
    return None


def _features_to_monthly(features):
    """Aggregate a year's daily features → monthly rows snapped to 0.5° cells."""
    rows = []
    for feat in features:
        lon, lat = feat["geometry"]["coordinates"][:2]
        cell_lat = round(lat / CELL) * CELL
        cell_lon = round(lon / CELL) * CELL
        series = feat["properties"]["parameter"]["PRECTOTCORR"]
        s = pd.Series(series, dtype="float64")
        s = s.replace(MISSING, np.nan)
        idx = pd.to_datetime(s.index, format="%Y%m%d")
        df = pd.DataFrame({"precip": s.values, "year": idx.year, "month": idx.month})
        g = df.groupby(["year", "month"])["precip"].agg(["mean", "max", "sum"])
        g = g.reset_index()
        g["cell_lat"] = cell_lat
        g["cell_lon"] = cell_lon
        rows.append(g)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def fetch_all():
    TILE_DIR.mkdir(parents=True, exist_ok=True)

    lat_edges = list(range(int(LAT_MIN), int(LAT_MAX), TILE))
    lon_edges = list(range(int(LON_MIN), int(LON_MAX), TILE))
    tiles = [(la, min(la + TILE, LAT_MAX), lo, min(lo + TILE, LON_MAX))
             for la in lat_edges for lo in lon_edges]
    print(f"Fetching {len(tiles)} spatial tiles × {YEAR_END-YEAR_START+1} years …\n")

    for ti, (la0, la1, lo0, lo1) in enumerate(tiles, 1):
        tile_path = TILE_DIR / f"tile_{int(la0)}_{int(lo0)}.parquet"
        if tile_path.exists():
            print(f"[{ti}/{len(tiles)}] tile ({la0},{lo0}) — cached, skip")
            continue

        print(f"[{ti}/{len(tiles)}] tile lat {la0}-{la1}, lon {lo0}-{lo1}")
        year_frames = []
        for year in range(YEAR_START, YEAR_END + 1):
            feats = _fetch(la0, la1, lo0, lo1, year)
            if feats:
                year_frames.append(_features_to_monthly(feats))
            time.sleep(0.3)   # be gentle on the API

        if year_frames:
            tile_df = pd.concat(year_frames, ignore_index=True)
            # average across NASA points that snapped to the same 0.5° cell
            tile_df = (tile_df
                       .groupby(["cell_lat", "cell_lon", "year", "month"])
                       .agg(rain_mean=("mean", "mean"),
                            rain_max=("max", "max"),
                            rain_sum=("sum", "mean"))
                       .reset_index())
            tile_df.to_parquet(tile_path, index=False)
            print(f"    saved {len(tile_df):,} cell-months → {tile_path.name}")

    # ── Combine all tiles ──
    parts = [pd.read_parquet(p) for p in sorted(TILE_DIR.glob("tile_*.parquet"))]
    if not parts:
        print("No tiles fetched — aborting.")
        return
    full = pd.concat(parts, ignore_index=True)
    full = (full.groupby(["cell_lat", "cell_lon", "year", "month"])
                .agg(rain_mean=("rain_mean", "mean"),
                     rain_max=("rain_max", "max"),
                     rain_sum=("rain_sum", "mean"))
                .reset_index())
    full.to_parquet(OUT_FILE, index=False)
    print(f"\n✓ Rainfall ready: {len(full):,} cell-months → {OUT_FILE}")
    print(f"  cells covered: {full[['cell_lat','cell_lon']].drop_duplicates().shape[0]:,}")


if __name__ == "__main__":
    fetch_all()
