"""
SentryMesh — Rebuild flood_susceptibility.parquet with Aqueduct return-period risk.

Adds two LEAKAGE-FREE static predictors (rp10_risk, rp100_risk) — country-level
flood-hazard priors from gfd_popsummary.csv — to BOTH positives and negatives.
Reuses cached rainfall/terrain features (no refetch); maps each point's (lat,lon)
to its country offline via reverse_geocoder.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import reverse_geocoder as rg

DATA = Path("data")
FEAT = ["rain_1d", "rain_3d", "rain_7d", "rain_30d", "rain_max3", "rain_api",
        "elev", "slope_deg", "relief"]
ASEAN_UPPER = ["PHILIPPINES", "INDONESIA", "THAILAND", "CAMBODIA", "MYANMAR",
               "VIET NAM", "MALAYSIA", "LAO PDR", "SINGAPORE", "BRUNEI"]
CC2NAME = {"PH": "Philippines", "ID": "Indonesia", "TH": "Thailand",
           "VN": "Vietnam", "MM": "Myanmar", "KH": "Cambodia", "MY": "Malaysia",
           "LA": "Laos", "SG": "Singapore", "BN": "Brunei", "TL": "Timor-Leste"}


def return_period_table():
    ps = pd.read_csv(DATA / "gfd_popsummary.csv")
    ps = ps[ps["unit_name"].isin(ASEAN_UPPER)].copy()
    ps["pop2030"] = ps["pop2030"].replace(0, np.nan).fillna(ps["pop2030"].median())
    ps["rp10_risk"] = (ps["P10_bh_10"] / ps["pop2030"]).clip(0, 1)
    ps["rp100_risk"] = (ps["P10_bh_100"] / ps["pop2030"]).clip(0, 1)
    name_map = {"PHILIPPINES": "Philippines", "INDONESIA": "Indonesia",
                "THAILAND": "Thailand", "CAMBODIA": "Cambodia", "MYANMAR": "Myanmar",
                "VIET NAM": "Vietnam"}
    ps["Country"] = ps["unit_name"].map(name_map).fillna(ps["unit_name"].str.title())
    return ps.set_index("Country")[["rp10_risk", "rp100_risk"]]


def country_of(df):
    res = rg.search([(float(a), float(b)) for a, b in zip(df["lat"], df["lon"])])
    return [CC2NAME.get(r["cc"], None) for r in res]


def main():
    rp = return_period_table()

    # Reconstruct the 318-positive set + 636 negatives from caches (no refetch)
    meta = pd.read_csv(DATA / "flood_positives_expanded.csv")
    feats = pd.read_parquet(DATA / "flood_pos_features.parquet")
    keep = set(meta[meta.source != "EMDAT_GEO"]["event_id"])
    pos = feats[feats.event_id.isin(keep)].copy(); pos["label"] = 1
    neg = pd.read_parquet(DATA / "flood_susceptibility_neg.parquet").head(2 * len(pos)).copy()
    neg["label"] = 0

    full = pd.concat([pos, neg], ignore_index=True)
    full["country"] = country_of(full)
    full["rp10_risk"] = full["country"].map(rp["rp10_risk"]).fillna(0.0)
    full["rp100_risk"] = full["country"].map(rp["rp100_risk"]).fillna(0.0)

    t = pd.to_datetime(full["date"])
    full["month_sin"] = np.sin(2 * np.pi * t.dt.month / 12)
    full["month_cos"] = np.cos(2 * np.pi * t.dt.month / 12)

    cols = ["lat", "lon", "date"] + FEAT + ["rp10_risk", "rp100_risk",
                                            "month_sin", "month_cos", "label"]
    out = full[cols]
    out.to_parquet(DATA / "flood_susceptibility.parquet", index=False)
    print(f"[OK] {len(out)} rows ({int(out.label.sum())} pos)  "
          f"rp10>0: {(out.rp10_risk > 0).sum()}  rp100>0: {(out.rp100_risk > 0).sum()}")
    print(out.groupby("label")[["rp10_risk", "rp100_risk"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
