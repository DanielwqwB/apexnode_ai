"""
SentryMesh — Expanded Flood Positives Builder

Combines ASEAN flood events (label=1) from three sources into one deduplicated
list of (lat, lon, date) points for the flood-susceptibility model:

  1. DFO Global Active Archive masterlist  (1985-2010, 536 ASEAN, all w/ centroid)
  2. EM-DAT public export                  (2000-2026, coordinate-tagged rows)
  3. gfd_qcdatabase_2019_08_01.csv         (existing, →2018)

Output: data/flood_positives_expanded.csv  (event_id, lat, lon, date, source)

Dedup key: (lat rounded 0.25°, lon rounded 0.25°, year-month) — the same event
reported in two archives collapses to one positive.
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA_ROOT = Path("data")
DL = Path.home() / "Downloads"

DFO_DBF  = DL / "_dfo" / "wlf_nhr_fl_dfomasterlist_20190418.dbf"
EMDAT_XLSX = DL / "public_emdat_custom_request_2026-06-20_e65e2fa8-6cb6-40bb-837f-e4dcb7b50a3f.xlsx"
GFD_CSV  = DATA_ROOT / "gfd_qcdatabase_2019_08_01.csv"
OUT      = DATA_ROOT / "flood_positives_expanded.csv"

ASEAN_KW = ["Philippines", "Indonesia", "Vietnam", "Viet Nam", "Thailand",
            "Malaysia", "Myanmar", "Burma", "Cambodia", "Laos", "Lao",
            "Singapore", "Brunei", "Timor"]
ASEAN_EXACT = ["Philippines", "Indonesia", "Viet Nam", "Vietnam", "Thailand",
               "Malaysia", "Myanmar", "Cambodia", "Laos",
               "Lao People's Democratic Republic", "Singapore",
               "Brunei Darussalam", "Timor-Leste"]
# ASEAN bounding box sanity filter
LAT_MIN, LAT_MAX = -12.0, 29.0
LON_MIN, LON_MAX = 90.0, 142.0


def _is_asean_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.contains("|".join(ASEAN_KW), case=False, na=False)


def load_dfo() -> pd.DataFrame:
    from dbfread import DBF
    rows = list(DBF(str(DFO_DBF), encoding="latin-1"))
    df = pd.DataFrame(rows)
    df = df[_is_asean_text(df["Country__c"])]
    out = pd.DataFrame({
        "lat": pd.to_numeric(df["Centroid_Y"], errors="coerce"),
        "lon": pd.to_numeric(df["Centroid_X"], errors="coerce"),
        "date": pd.to_datetime(df["Began"], errors="coerce"),
        "source": "DFO",
    }).dropna(subset=["lat", "lon", "date"])
    print(f"  DFO:    {len(out)} ASEAN events")
    return out


def load_emdat() -> pd.DataFrame:
    x = pd.read_excel(EMDAT_XLSX)
    fl = x[x["Disaster Type"].str.contains("Flood", case=False, na=False)]
    fl = fl[fl["Country"].isin(ASEAN_EXACT)]
    fl = fl.dropna(subset=["Latitude", "Longitude"])
    # Build a date from Start Year/Month/Day (default to mid-month/day if missing)
    y = pd.to_numeric(fl["Start Year"], errors="coerce")
    m = pd.to_numeric(fl["Start Month"], errors="coerce").fillna(6).clip(1, 12)
    d = pd.to_numeric(fl["Start Day"], errors="coerce").fillna(15).clip(1, 28)
    date = pd.to_datetime(dict(year=y, month=m, day=d), errors="coerce")
    out = pd.DataFrame({
        "lat": pd.to_numeric(fl["Latitude"], errors="coerce"),
        "lon": pd.to_numeric(fl["Longitude"], errors="coerce"),
        "date": date,
        "source": "EMDAT",
    }).dropna(subset=["lat", "lon", "date"])
    print(f"  EM-DAT: {len(out)} ASEAN events (with coordinates)")
    return out


def load_gfd() -> pd.DataFrame:
    df = pd.read_csv(GFD_CSV, parse_dates=["Began"])
    df = df[df["Country"].isin(ASEAN_EXACT)]
    out = pd.DataFrame({
        "lat": pd.to_numeric(df["lat"], errors="coerce"),
        "lon": pd.to_numeric(df["long"], errors="coerce"),
        "date": pd.to_datetime(df["Began"], errors="coerce"),
        "source": "GFD",
    }).dropna(subset=["lat", "lon", "date"])
    print(f"  GFD:    {len(out)} ASEAN events")
    return out


def main():
    print("Loading flood positives from 3 sources …")
    frames = []
    for loader in (load_dfo, load_emdat, load_gfd):
        try:
            frames.append(loader())
        except Exception as e:
            print(f"  ! {loader.__name__} failed: {e}")
    allp = pd.concat(frames, ignore_index=True)

    # Bounding-box sanity filter
    allp = allp[
        allp["lat"].between(LAT_MIN, LAT_MAX) & allp["lon"].between(LON_MIN, LON_MAX)
    ]
    print(f"\n  combined (pre-dedup): {len(allp)}")

    # Dedup: same ~0.25° cell + same month = same event
    allp["_klat"] = (allp["lat"] / 0.25).round() * 0.25
    allp["_klon"] = (allp["lon"] / 0.25).round() * 0.25
    allp["_kym"]  = allp["date"].dt.to_period("M").astype(str)
    # Prefer DFO > GFD > EMDAT when collapsing duplicates (DFO has best centroids)
    pri = {"DFO": 0, "GFD": 1, "EMDAT": 2}
    allp["_pri"] = allp["source"].map(pri)
    allp = allp.sort_values("_pri").drop_duplicates(subset=["_klat", "_klon", "_kym"])

    allp = allp.sort_values("date").reset_index(drop=True)
    allp["event_id"] = "FLX_" + allp.index.astype(str)
    out = allp[["event_id", "lat", "lon", "date", "source"]]
    out.to_csv(OUT, index=False)

    print(f"\n[OK] deduplicated positives: {len(out)}")
    print(out["source"].value_counts().to_string())
    print(f"  date range: {out['date'].min().date()} -> {out['date'].max().date()}")
    print(f"  post-2010 (new vs DFO): {(out['date'] >= '2011-01-01').sum()}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
