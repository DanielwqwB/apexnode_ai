"""
Fast landslide hard-negative rebuild (capped + balanced for deadline speed).

Samples CAP positives, generates one hard negative each (same location, random
non-flood date), fetches only rainfall (terrain copied). Writes a balanced
data/susceptibility.parquet. Lower WINDOW retries so NASA POWER throttling can't
stall the run.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_antecedent_rainfall import _fetch_point, _features

random.seed(11); np.random.seed(11)
DATA = Path("data")
FEAT9 = ["rain_1d", "rain_3d", "rain_7d", "rain_30d", "rain_max3", "rain_api",
         "elev", "slope_deg", "relief"]
RAIN_KEYS = ["rain_1d", "rain_3d", "rain_7d", "rain_30d", "rain_max3", "rain_api"]
CAP = 700          # positives to keep (balanced with equal hard negatives)
WORKERS = 16


def fetch_rain(lat, lon, date):
    end = pd.Timestamp(date).normalize()
    start = (end - pd.Timedelta(days=35)).strftime("%Y%m%d")
    s = _fetch_point(lat, lon, start, end.strftime("%Y%m%d"), retries=2)
    return _features(s, end) if s else {k: 0.0 for k in RAIN_KEYS}


def main():
    src = DATA / "susceptibility_random.parquet"
    if not src.exists():
        src = DATA / "susceptibility.parquet"
    land = pd.read_parquet(src)
    pos = land[land.label == 1].copy()
    if len(pos) > CAP:
        pos = pos.sample(CAP, random_state=11).reset_index(drop=True)
    print(f"landslide positives kept: {len(pos)}", flush=True)

    recs = []
    for _, r in pos.iterrows():
        d = pd.Timestamp(2000, 1, 1)
        for _t in range(12):
            d = pd.Timestamp(random.randint(2000, 2020), random.randint(1, 12), random.randint(1, 28))
            if abs((d - pd.Timestamp(r["date"])).days) >= 180:
                break
        recs.append({"lat": r["lat"], "lon": r["lon"], "date": d,
                     "elev": r["elev"], "slope_deg": r["slope_deg"], "relief": r["relief"]})
    neg = pd.DataFrame(recs).reset_index(drop=True)
    print(f"fetching rainfall for {len(neg)} hard negatives (workers={WORKERS}) …", flush=True)

    def work(i):
        row = neg.loc[i]
        return i, fetch_rain(row["lat"], row["lon"], row["date"])
    done = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for fut in as_completed([ex.submit(work, i) for i in neg.index]):
            i, feats = fut.result()
            for k, v in feats.items():
                neg.at[i, k] = v
            done += 1
            if done % 150 == 0:
                print(f"  {done}/{len(neg)}", flush=True)

    pos["label"] = 1; neg["label"] = 0
    cols = ["lat", "lon", "date"] + FEAT9
    full = pd.concat([pos[cols + ["label"]], neg[cols + ["label"]]], ignore_index=True)
    t = pd.to_datetime(full["date"])
    full["month_sin"] = np.sin(2 * np.pi * t.dt.month / 12)
    full["month_cos"] = np.cos(2 * np.pi * t.dt.month / 12)
    full.to_parquet(DATA / "susceptibility.parquet", index=False)
    print(f"[OK] landslide hard-neg: {len(full)} rows ({int(full.label.sum())} pos)", flush=True)


if __name__ == "__main__":
    main()

