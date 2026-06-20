"""
SentryMesh — Geocode coordinate-less EM-DAT flood events to province centroids.

Many recent (2021-2026) EM-DAT ASEAN floods have no lat/lon but DO carry GADM
admin-1 (province) info. We geocode each event's PRIMARY province to a centroid
via OpenStreetMap/Nominatim (cached per unique province, 1 req/sec per policy),
yielding extra flood positives the original coordinate filter dropped.

Output: data/flood_positives_emdat_geocoded.csv  (event_id, lat, lon, date, source)
"""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
import pandas as pd

DATA_ROOT = Path("data")
DL = Path.home() / "Downloads"
EMDAT_XLSX = DL / "public_emdat_custom_request_2026-06-20_e65e2fa8-6cb6-40bb-837f-e4dcb7b50a3f.xlsx"
CACHE = DATA_ROOT / "geocode_cache.json"
OUT = DATA_ROOT / "flood_positives_emdat_geocoded.csv"

ASEAN = ["Philippines", "Indonesia", "Viet Nam", "Vietnam", "Thailand", "Malaysia",
         "Myanmar", "Cambodia", "Laos", "Singapore", "Brunei Darussalam", "Timor-Leste"]
# EM-DAT country name -> geocoding-friendly country
CTRY = {"Viet Nam": "Vietnam", "Brunei Darussalam": "Brunei",
        "Lao People's Democratic Republic": "Laos"}
# ASEAN bbox sanity
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = -12.0, 29.0, 90.0, 142.0


def primary_province(row):
    """Best-effort primary admin-1 name from GADM Admin Units / Admin Units."""
    for col in ("GADM Admin Units", "Admin Units"):
        val = row.get(col)
        if isinstance(val, str) and val.strip().startswith("["):
            try:
                arr = json.loads(val)
                if arr:
                    d = arr[0]
                    return d.get("name_1") or d.get("adm1_name")
            except Exception:
                pass
    return None


def load_cache():
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {}


def geocode(query, cache):
    if query in cache:
        return cache[query]
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "limit": 1})
    req = urllib.request.Request(url, headers={"User-Agent": "SentryMesh-research/1.0 (academic flood study)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            res = json.loads(r.read())
        if res:
            cache[query] = [float(res[0]["lat"]), float(res[0]["lon"])]
        else:
            cache[query] = None
    except Exception as e:
        print(f"   ! geocode failed for {query!r}: {e}")
        cache[query] = None
    time.sleep(1.1)   # Nominatim policy: max 1 req/sec
    return cache[query]


def main():
    x = pd.read_excel(EMDAT_XLSX)
    fl = x[(x["Disaster Type"].str.contains("Flood", case=False, na=False)) &
           (x["Country"].isin(ASEAN)) & (x["Latitude"].isna())].copy()
    fl["prov"] = fl.apply(primary_province, axis=1)
    fl = fl.dropna(subset=["prov"])
    print(f"coordinate-less floods with a province: {len(fl)}")

    cache = load_cache()
    fl["query"] = fl["prov"].astype(str) + ", " + fl["Country"].map(lambda c: CTRY.get(c, c))
    uniq = sorted(fl["query"].unique())
    print(f"unique provinces to geocode: {len(uniq)}  (~{len(uniq)*1.1/60:.1f} min)")

    for i, q in enumerate(uniq):
        geocode(q, cache)
        if (i + 1) % 25 == 0:
            CACHE.write_text(json.dumps(cache), encoding="utf-8")
            print(f"  {i+1}/{len(uniq)} geocoded")
    CACHE.write_text(json.dumps(cache), encoding="utf-8")

    y = pd.to_numeric(fl["Start Year"], errors="coerce")
    m = pd.to_numeric(fl["Start Month"], errors="coerce").fillna(6).clip(1, 12)
    d = pd.to_numeric(fl["Start Day"], errors="coerce").fillna(15).clip(1, 28)
    fl["date"] = pd.to_datetime(dict(year=y, month=m, day=d), errors="coerce")
    fl["coord"] = fl["query"].map(lambda q: cache.get(q))
    fl = fl.dropna(subset=["coord", "date"])
    fl["lat"] = fl["coord"].map(lambda c: c[0])
    fl["lon"] = fl["coord"].map(lambda c: c[1])
    fl = fl[fl["lat"].between(LAT_MIN, LAT_MAX) & fl["lon"].between(LON_MIN, LON_MAX)]

    out = pd.DataFrame({
        "event_id": "EMG_" + fl["DisNo."].astype(str),
        "lat": fl["lat"], "lon": fl["lon"], "date": fl["date"], "source": "EMDAT_GEO",
    }).dropna()
    out.to_csv(OUT, index=False)
    print(f"\n[OK] geocoded EM-DAT positives: {len(out)}  "
          f"(year>=2021: {(out['date'] >= '2021-01-01').sum()}) -> {OUT}")


if __name__ == "__main__":
    main()
