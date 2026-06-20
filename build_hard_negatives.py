"""
SentryMesh — Hard-Negative Rebuild (removes negative-sampling optimism)

Random negatives let the model separate classes by geography/elevation (ocean vs
flood-prone land). Hard negatives fix this: each negative is placed at a POSITIVE
location but on a random non-flood date (>=180 days from the event). Terrain, rp,
and lat/lon are therefore IDENTICAL between a positive and its negatives — the only
thing that can distinguish them is antecedent rainfall (and season). That is the
honest operational question: "will THIS place flood on THIS day?"

Rebuilds (overwrites, after backing up *_random.parquet):
  data/flood_susceptibility.parquet
  data/susceptibility.parquet
Only rainfall is fetched (terrain is copied from the matched positive).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import reverse_geocoder as rg

from fetch_antecedent_rainfall import _fetch_point, _features

random.seed(11)
np.random.seed(11)

DATA = Path("data")
FEAT9 = ["rain_1d", "rain_3d", "rain_7d", "rain_30d", "rain_max3", "rain_api",
         "elev", "slope_deg", "relief"]
RAIN_KEYS = ["rain_1d", "rain_3d", "rain_7d", "rain_30d", "rain_max3", "rain_api"]
WORKERS = 12
ASEAN_UPPER = ["PHILIPPINES", "INDONESIA", "THAILAND", "CAMBODIA", "MYANMAR",
               "VIET NAM", "MALAYSIA", "LAO PDR", "SINGAPORE", "BRUNEI"]
CC2NAME = {"PH": "Philippines", "ID": "Indonesia", "TH": "Thailand", "VN": "Vietnam",
           "MM": "Myanmar", "KH": "Cambodia", "MY": "Malaysia", "LA": "Laos",
           "SG": "Singapore", "BN": "Brunei", "TL": "Timor-Leste"}


def return_period_table():
    ps = pd.read_csv(DATA / "gfd_popsummary.csv")
    ps = ps[ps["unit_name"].isin(ASEAN_UPPER)].copy()
    ps["pop2030"] = ps["pop2030"].replace(0, np.nan).fillna(ps["pop2030"].median())
    ps["rp10_risk"] = (ps["P10_bh_10"] / ps["pop2030"]).clip(0, 1)
    ps["rp100_risk"] = (ps["P10_bh_100"] / ps["pop2030"]).clip(0, 1)
    nm = {"PHILIPPINES": "Philippines", "INDONESIA": "Indonesia", "THAILAND": "Thailand",
          "CAMBODIA": "Cambodia", "MYANMAR": "Myanmar", "VIET NAM": "Vietnam"}
    ps["Country"] = ps["unit_name"].map(nm).fillna(ps["unit_name"].str.title())
    return ps.set_index("Country")[["rp10_risk", "rp100_risk"]]


def fetch_rain(lat, lon, date):
    end = pd.Timestamp(date).normalize()
    start = (end - pd.Timedelta(days=35)).strftime("%Y%m%d")
    s = _fetch_point(lat, lon, start, end.strftime("%Y%m%d"))
    return _features(s, end) if s else {k: 0.0 for k in RAIN_KEYS}


def make_hard_negs(pos_df, ratio, has_rp):
    recs = []
    for _, r in pos_df.iterrows():
        for _ in range(ratio):
            d = pd.Timestamp(2000, 1, 1)
            for _t in range(12):
                d = pd.Timestamp(random.randint(2000, 2020), random.randint(1, 12),
                                 random.randint(1, 28))
                if abs((d - pd.Timestamp(r["date"])).days) >= 180:
                    break
            rec = {"lat": r["lat"], "lon": r["lon"], "date": d, "elev": r["elev"],
                   "slope_deg": r["slope_deg"], "relief": r["relief"]}
            if has_rp:
                rec["rp10_risk"] = r["rp10_risk"]; rec["rp100_risk"] = r["rp100_risk"]
            recs.append(rec)
    neg = pd.DataFrame(recs).reset_index(drop=True)
    print(f"    fetching rainfall for {len(neg)} hard negatives …", flush=True)

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
            if done % 200 == 0:
                print(f"      {done}/{len(neg)}", flush=True)
    return neg


def finalize(pos, neg, cols, out):
    pos = pos.copy(); pos["label"] = 1
    neg = neg.copy(); neg["label"] = 0
    full = pd.concat([pos[cols + ["label"]], neg[cols + ["label"]]], ignore_index=True)
    t = pd.to_datetime(full["date"])
    full["month_sin"] = np.sin(2 * np.pi * t.dt.month / 12)
    full["month_cos"] = np.cos(2 * np.pi * t.dt.month / 12)
    full.to_parquet(out, index=False)
    return full


def main():
    for p in ["flood_susceptibility.parquet", "susceptibility.parquet"]:
        src = DATA / p
        if src.exists():
            shutil.copy(src, DATA / p.replace(".parquet", "_random.parquet"))
    print("backed up random-negative versions -> *_random.parquet")

    # ---- FLOOD ----
    print("FLOOD: building positives + hard negatives …", flush=True)
    rp = return_period_table()
    meta = pd.read_csv(DATA / "flood_positives_expanded.csv")
    feats = pd.read_parquet(DATA / "flood_pos_features.parquet")
    pos = feats[feats.event_id.isin(set(meta[meta.source != "EMDAT_GEO"]["event_id"]))].copy()
    cc = rg.search([(float(a), float(b)) for a, b in zip(pos.lat, pos.lon)])
    pos["country"] = [CC2NAME.get(r["cc"]) for r in cc]
    pos["rp10_risk"] = pos.country.map(rp["rp10_risk"]).fillna(0.0)
    pos["rp100_risk"] = pos.country.map(rp["rp100_risk"]).fillna(0.0)
    fneg = make_hard_negs(pos, ratio=2, has_rp=True)
    fcols = ["lat", "lon", "date"] + FEAT9 + ["rp10_risk", "rp100_risk"]
    ff = finalize(pos, fneg, fcols, DATA / "flood_susceptibility.parquet")
    print(f"  flood: {len(ff)} rows ({int(ff.label.sum())} pos)")

    # ---- LANDSLIDE ----
    print("LANDSLIDE: building positives + hard negatives …", flush=True)
    land = pd.read_parquet(DATA / "susceptibility.parquet")
    lpos = land[land.label == 1].copy()
    lneg = make_hard_negs(lpos, ratio=1, has_rp=False)
    lcols = ["lat", "lon", "date"] + FEAT9
    lf = finalize(lpos, lneg, lcols, DATA / "susceptibility.parquet")
    print(f"  landslide: {len(lf)} rows ({int(lf.label.sum())} pos)")
    print("[OK] hard-negative rebuild complete.")


if __name__ == "__main__":
    main()
