"""
SentryMesh — Flood Susceptibility Dataset Builder

Same occurrence framing as landslides, for floods:
    Positives = GFD flood events (antecedent rainfall + terrain at event place/time)
    Negatives = random (location, date) points with NO flood
    Target    = will a flood occur, given antecedent rain + (low) elevation + season?

Floods are driven by heavy antecedent rainfall in low-lying terrain, so the
physical features carry real signal. Resumable: negatives cache to
data/flood_susceptibility_neg.parquet.
"""

import random
import math
import time
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
NEG_CACHE = DATA_ROOT / "flood_susceptibility_neg.parquet"
OUT       = DATA_ROOT / "flood_susceptibility.parquet"
ASEAN = ["Philippines", "Indonesia", "Vietnam", "Thailand", "Malaysia", "Myanmar",
         "Cambodia", "Laos", "Singapore", "Brunei", "Viet Nam", "Lao PDR", "Timor-Leste"]
FEAT = ["rain_1d", "rain_3d", "rain_7d", "rain_30d", "rain_max3", "rain_api",
        "elev", "slope_deg", "relief"]
NEG_RATIO = 2   # 2 negatives per flood event (only 147 positives, so widen the negative space)


def load_positives() -> pd.DataFrame:
    fl = pd.read_csv(DATA_ROOT / "gfd_qcdatabase_2019_08_01.csv", parse_dates=["Began"])
    fl = fl[fl["Country"].isin(ASEAN)]
    ev = pd.DataFrame({
        "event_id": "FL_" + fl["ID"].astype(str),
        "lat": fl["lat"], "lon": fl["long"],
        "date": pd.to_datetime(fl["Began"], errors="coerce"),
    }).dropna(subset=["lat", "lon", "date"])
    ev = ev[ev["date"] >= "1981-01-02"].drop_duplicates("event_id")

    ante = pd.read_parquet(DATA_ROOT / "rainfall" / "antecedent.parquet")
    terr = pd.read_parquet(DATA_ROOT / "terrain.parquet")
    ev = ev.merge(ante, on="event_id", how="left").merge(terr, on="event_id", how="left")
    ev = ev.dropna(subset=["rain_7d", "slope_deg"])
    ev["label"] = 1
    print(f"  flood positives with full features: {len(ev)}")
    return ev[["lat", "lon", "date"] + FEAT + ["label"]]


def gen_negatives(n):
    rows = []
    for i in range(n):
        lat = random.uniform(2.0, 26.0)
        lon = random.uniform(95.0, 140.0)
        y = random.randint(2000, 2017)
        m = random.randint(1, 12)
        d = random.randint(1, 28)
        rows.append({"neg_id": f"FNEG_{i:05d}", "lat": lat, "lon": lon,
                     "date": pd.Timestamp(y, m, d)})
    return pd.DataFrame(rows)


def fetch_neg_features(negs):
    results, done = [], set()
    if NEG_CACHE.exists():
        prev = pd.read_parquet(NEG_CACHE)
        results = prev.to_dict("records")
        done = set(prev["neg_id"])
        print(f"  resuming — {len(done)} negatives cached")
    todo = negs[~negs["neg_id"].isin(done)].reset_index(drop=True)
    print(f"  fetching features for {len(todo)} negatives …")

    for i, r in todo.iterrows():
        end = r["date"].normalize()
        start = (end - pd.Timedelta(days=35)).strftime("%Y%m%d")
        series = _fetch_point(r["lat"], r["lon"], start, end.strftime("%Y%m%d"))
        feats = _features(series, r["date"]) if series else {k: 0.0 for k in FEAT[:6]}
        e = _fetch_elev([r.lat, r.lat + DELTA, r.lat - DELTA, r.lat, r.lat],
                        [r.lon, r.lon, r.lon, r.lon + DELTA, r.lon - DELTA])
        if e:
            grad = max(abs(e[0] - e[1]), abs(e[0] - e[2]), abs(e[0] - e[3]), abs(e[0] - e[4]))
            terr = {"elev": float(e[0]),
                    "slope_deg": math.degrees(math.atan(grad / DIST_M)),
                    "relief": float(max(e) - min(e))}
        else:
            terr = {"elev": 0.0, "slope_deg": 0.0, "relief": 0.0}
        results.append({"neg_id": r.neg_id, "lat": r.lat, "lon": r.lon,
                        "date": r.date, **feats, **terr})
        if (i + 1) % 50 == 0:
            pd.DataFrame(results).to_parquet(NEG_CACHE, index=False)
            print(f"    {i+1}/{len(todo)} done")
        time.sleep(0.25)

    df = pd.DataFrame(results)
    df.to_parquet(NEG_CACHE, index=False)
    df["label"] = 0
    return df[["lat", "lon", "date"] + FEAT + ["label"]]


def main():
    print("Loading flood positives …")
    pos = load_positives()
    print("Generating + fetching negatives …")
    negs = gen_negatives(len(pos) * NEG_RATIO)
    neg = fetch_neg_features(negs)

    full = pd.concat([pos, neg], ignore_index=True)
    t = pd.to_datetime(full["date"])
    full["month_sin"] = np.sin(2 * np.pi * t.dt.month / 12)
    full["month_cos"] = np.cos(2 * np.pi * t.dt.month / 12)
    full.to_parquet(OUT, index=False)
    print(f"\n✓ Flood susceptibility dataset: {len(full)} rows "
          f"({int(full['label'].sum())} pos / {int((full['label']==0).sum())} neg) → {OUT}")


if __name__ == "__main__":
    main()
