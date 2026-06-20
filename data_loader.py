"""
SentryMesh — Data Loader & Preprocessor  (enriched)
Handles: Typhoon (IBTrACS-style), GFD Flood, NASA Global Landslide Catalog

Enrichments over v1:
  flood → compiled_pop join       : adds log_exposed + pop_severity_label
  flood → popsummary return period: adds country-level P10_bh_10 / P10_bh_100
  flood → validation_points       : adds mean MNDWI, NDVI, flood_pixel_frac
                                    (pixel-level satellite features, aggregated per event)
  label → composite severity      : flood label now uses exposed population +
                                    original Severity score instead of Severity alone

Produces: Unified ASEAN spatio-temporal graph dataset for ST-GNN training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── ASEAN constants ──────────────────────────────────────────────────────────
ASEAN_LAT = (0.0, 28.0)
ASEAN_LON = (90.0, 145.0)

ASEAN_COUNTRIES = [
    "Philippines", "Indonesia", "Vietnam", "Thailand",
    "Malaysia", "Myanmar", "Cambodia", "Laos", "Singapore", "Brunei",
    "Viet Nam", "Lao PDR", "Timor-Leste"
]

# Upper-case versions for popsummary join
ASEAN_UPPER = {c.upper() for c in ASEAN_COUNTRIES} | {"VIET NAM", "LAO PDR"}

DATA_ROOT = Path("data")


# ══════════════════════════════════════════════════════════════════════════════
# ENRICHMENT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_compiled_pop() -> pd.DataFrame:
    """
    compiled_pop_ghsl_ts_2019_08_04.csv
    Key: index (= GFD flood ID), country, area, exposed (population count)
    Returns per-ID aggregated exposed population (sum across multi-country events).
    """
    path = DATA_ROOT / "compiled_pop_ghsl_ts_2019_08_04.csv"
    if not path.exists():
        return pd.DataFrame(columns=["flood_id", "log_exposed", "exposed_area"])

    cp = pd.read_csv(path)

    # Bug #5 fix: normalise country name variants before filtering
    # GFD may use "Burma" for older records; UN name changed to Myanmar
    COUNTRY_ALIASES = {"Burma": "Myanmar", "Lao People's Democratic Republic": "Lao PDR"}
    cp["country"] = cp["country"].replace(COUNTRY_ALIASES)

    cp = cp[cp["country"].isin(ASEAN_COUNTRIES)].copy()
    cp = cp.rename(columns={"index": "flood_id"})

    # Sum exposed population across all country rows for the same event
    agg = cp.groupby("flood_id").agg(
        total_exposed=("exposed", "sum"),
        total_area=("area", "sum"),
    ).reset_index()

    agg["log_exposed"]   = np.log1p(agg["total_exposed"])
    agg["exposed_area"]  = np.log1p(agg["total_area"])
    return agg[["flood_id", "log_exposed", "exposed_area"]]


def _load_return_period() -> pd.DataFrame:
    """
    gfd_popsummary.csv
    Key: unit_name (UPPER-CASE country)
    Returns country → P10_bh_10, P10_bh_100 (pop exposed at 10yr / 100yr return)
    Normalised to [0, 1] across ASEAN countries.
    """
    path = DATA_ROOT / "gfd_popsummary.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Country", "rp10_risk", "rp100_risk"])

    ps = pd.read_csv(path)
    ps = ps[ps["unit_name"].isin(ASEAN_UPPER)].copy()

    # Normalise by dividing by 2030 population
    ps["pop2030"] = ps["pop2030"].replace(0, np.nan).fillna(ps["pop2030"].median())
    ps["rp10_risk"]  = ps["P10_bh_10"]  / ps["pop2030"]
    ps["rp100_risk"] = ps["P10_bh_100"] / ps["pop2030"]

    # Clip to [0,1]
    ps["rp10_risk"]  = ps["rp10_risk"].clip(0, 1)
    ps["rp100_risk"] = ps["rp100_risk"].clip(0, 1)

    # Map back to mixed-case country names for join
    name_map = {
        "PHILIPPINES": "Philippines", "INDONESIA": "Indonesia",
        "THAILAND": "Thailand", "CAMBODIA": "Cambodia",
        "MYANMAR": "Myanmar", "VIET NAM": "Vietnam",
    }
    ps["Country"] = ps["unit_name"].map(name_map).fillna(ps["unit_name"].str.title())
    return ps[["Country", "rp10_risk", "rp100_risk"]]


def _load_spectral_features() -> pd.DataFrame:
    """
    gfd_validation_points_2018_12_17.csv
    Pixel-level satellite observations (Landsat bands, MNDWI, NDVI).
    Aggregated per dfoID (= GFD flood ID).
    Only 4 ASEAN events have coverage — join fills missing with 0.
    """
    path = DATA_ROOT / "gfd_validation_points_2018_12_17.csv"
    if not path.exists():
        return pd.DataFrame(columns=["flood_id", "mean_MNDWI", "mean_NDVI",
                                      "flood_pixel_frac"])

    vp = pd.read_csv(path)

    # validation column has negative sentinels; clamp to [0,1]
    vp["validation"] = vp["validation"].clip(0, 1)

    agg = vp.groupby("dfoID").agg(
        mean_MNDWI      =("MNDWI",      "mean"),
        mean_NDVI       =("NDVI",       "mean"),
        mean_B4         =("B4",         "mean"),   # Red band
        mean_B5         =("B5",         "mean"),   # NIR band
        flood_pixel_frac=("validation", "mean"),   # fraction confirmed flooded
    ).reset_index().rename(columns={"dfoID": "flood_id"})

    return agg


# Rainfall feature columns added by _add_rainfall (kept here so all loaders agree)
RAIN_COLS = ["rain_mean", "rain_max", "rain_sum"]
_CELL = 0.5   # must match CELL in build_unified_dataset / fetch_rainfall.py


def _add_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    """
    Join monthly NASA POWER rainfall onto an event dataframe by snapping each
    event's LAT/LON to a 0.5° cell and matching its (year, month).

    Used for FLOOD and LANDSLIDE only — typhoon rainfall is handled via the
    OpenWeather pipeline, so typhoon rows keep rain_* = 0.
    Safe no-op (fills 0) if the rainfall parquet hasn't been fetched yet.
    """
    path = DATA_ROOT / "rainfall" / "rainfall_monthly.parquet"
    if not path.exists():
        print("    [rain] rainfall_monthly.parquet not found — run fetch_rainfall.py "
              "(filling rain_* with 0 for now)")
        for c in RAIN_COLS:
            df[c] = 0.0
        return df

    rain = pd.read_parquet(path)

    t = pd.to_datetime(df["time"], errors="coerce")
    df["_year"]     = t.dt.year
    df["_month"]    = t.dt.month
    df["_cell_lat"] = (df["LAT"] / _CELL).round() * _CELL
    df["_cell_lon"] = (df["LON"] / _CELL).round() * _CELL

    merged = df.merge(
        rain.rename(columns={"cell_lat": "_cell_lat", "cell_lon": "_cell_lon",
                             "year": "_year", "month": "_month"}),
        on=["_cell_lat", "_cell_lon", "_year", "_month"], how="left",
    )
    for c in RAIN_COLS:
        merged[c] = merged[c].fillna(0.0)
    merged.drop(columns=["_year", "_month", "_cell_lat", "_cell_lon"], inplace=True)

    hit = (merged["rain_mean"] > 0).sum()
    print(f"    [rain] rainfall joined: {hit}/{len(merged)} events enriched")
    return merged


# Per-event antecedent-rainfall + terrain feature columns (joined by event_id)
ANTECEDENT_COLS = ["rain_1d", "rain_3d", "rain_7d", "rain_30d", "rain_max3", "rain_api"]
TERRAIN_COLS    = ["elev", "slope_deg", "relief"]


def _add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Join per-event antecedent rainfall (data/rainfall/antecedent.parquet) and
    terrain (data/terrain.parquet) by event_id. These are the strongest physical
    predictors of floods/landslides — unlike the monthly cell rainfall, they are
    tied to the specific event and its location.
    Safe no-op (fills 0) if a parquet hasn't been fetched yet.
    """
    ante_path = DATA_ROOT / "rainfall" / "antecedent.parquet"
    terr_path = DATA_ROOT / "terrain.parquet"

    if ante_path.exists():
        ante = pd.read_parquet(ante_path)
        df = df.merge(ante, on="event_id", how="left")
        hit = df["rain_7d"].notna().sum()
        print(f"    [ante] antecedent rainfall joined: {hit}/{len(df)} events")
    else:
        print("    [ante] antecedent.parquet missing — run fetch_antecedent_rainfall.py")
    for c in ANTECEDENT_COLS:
        df[c] = df[c].fillna(0.0) if c in df.columns else 0.0

    if terr_path.exists():
        terr = pd.read_parquet(terr_path)
        df = df.merge(terr, on="event_id", how="left")
        hit = df["slope_deg"].notna().sum()
        print(f"    [terr] terrain joined: {hit}/{len(df)} events")
    else:
        print("    [terr] terrain.parquet missing — run fetch_terrain.py")
    for c in TERRAIN_COLS:
        df[c] = df[c].fillna(0.0) if c in df.columns else 0.0

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TYPHOON  (IBTrACS-style track data)
# ══════════════════════════════════════════════════════════════════════════════

def load_typhoon(split: str = "train") -> pd.DataFrame:
    path = DATA_ROOT / "typhoon" / f"{split}.csv"
    print(f"  Loading typhoon/{split}.csv …")
    df = pd.read_csv(path, parse_dates=["ISO_TIME"])

    df = df[
        df["LAT"].between(*ASEAN_LAT) &
        df["LON"].between(*ASEAN_LON)
    ].copy()

    for col in ["WMO_WIND", "WMO_PRES"]:
        df[col] = df.groupby("SID")[col].transform(
            lambda x: x.fillna(x.median())
        )
    df["WMO_WIND"] = df["WMO_WIND"].fillna(df["WMO_WIND"].median())
    df["WMO_PRES"] = df["WMO_PRES"].fillna(df["WMO_PRES"].median())

    df["hour"]      = df["ISO_TIME"].dt.hour
    df["month"]     = df["ISO_TIME"].dt.month
    df["dayofyear"] = df["ISO_TIME"].dt.dayofyear

    df["cat"] = pd.cut(
        df["WMO_WIND"],
        bins=[-1, 63, 82, 95, 112, 136, 999],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(float)

    df["risk_score"] = df["risk_score"].fillna(0.0)
    max_risk = df["risk_score"].max()
    df["risk_norm"]  = df["risk_score"] / max_risk if max_risk > 0 else 0.0

    df["label"] = df["event"].astype(int)

    feature_cols = [
        "WMO_WIND", "WMO_PRES", "STORM_SPEED", "STORM_DIR",
        "DIST2LAND", "elevation", "slope", "aspect",
        "hour", "month", "dayofyear", "cat", "risk_norm",
        "delta_lat", "delta_lon", "advection_intensity",
        "monsoon_proxy", "aerosol_proxy",
    ]
    df[feature_cols] = df[feature_cols].fillna(0.0)

    df["hazard_type"] = "typhoon"
    df["time"]        = df["ISO_TIME"]

    out = df[["SID", "time", "LAT", "LON", "label", "hazard_type"] + feature_cols].copy()
    out.rename(columns={"SID": "event_id"}, inplace=True)
    print(f"    → {len(out):,} rows  |  events: {out['label'].sum():,}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FLOOD  (GFD + enrichment layers)
# ══════════════════════════════════════════════════════════════════════════════

def load_flood() -> pd.DataFrame:
    """
    Base: gfd_qcdatabase_2019_08_01.csv
    + compiled_pop   → log_exposed, exposed_area  (event-level)
    + popsummary     → rp10_risk, rp100_risk       (country-level baseline)
    + val_points     → mean_MNDWI, mean_NDVI,
                       flood_pixel_frac, mean_B4/B5 (event-level satellite)

    Composite label:
        label = 1 if (Severity >= 2) OR (log_exposed > median_log_exposed)
        This captures events with large population exposure even when
        the original Severity score is low.
    """
    print("  Loading flood database …")
    df = pd.read_csv(DATA_ROOT / "gfd_qcdatabase_2019_08_01.csv",
                     parse_dates=["Began", "Ended"])
    df = df[df["Country"].isin(ASEAN_COUNTRIES)].copy()

    # ── Temporal ──
    df["time"]       = df["Began"]
    df["duration_d"] = (df["Ended"] - df["Began"]).dt.days.fillna(0).clip(0, 180)
    df["month"]      = df["Began"].dt.month
    df["dayofyear"]  = df["Began"].dt.dayofyear
    df["hour"]       = 0

    df["Severity"] = df["Severity"].fillna(df["Severity"].median())
    df["Dead"] = df["Dead"].fillna(0)
    df["Displaced"] = df["Displaced"].fillna(0)

    cause_map = {
        "Monsoonal rain": 0, "Heavy rain": 1, "Tropical cyclone": 2,
        "Snowmelt": 3, "Dam break": 4, "Tidal surge": 5,
        "Earthquake": 6, "Ice jam": 7,
    }
    df["cause_code"]    = df["MainCause"].map(cause_map).fillna(8)
    df["log_dead"]      = np.log1p(df["Dead"])
    df["log_displaced"] = np.log1p(df["Displaced"])

    # ── Enrichment 1: exposed population ──────────────────────────────────
    pop_df = _load_compiled_pop()
    df = df.merge(pop_df, left_on="ID", right_on="flood_id", how="left")
    df["log_exposed"]  = df["log_exposed"].fillna(0.0)
    df["exposed_area"] = df["exposed_area"].fillna(0.0)
    print(f"    [pop] exposed population joined: "
          f"{df['log_exposed'].gt(0).sum()}/{len(df)} events enriched")

    # ── Enrichment 2: return period flood risk (country baseline) ─────────
    rp_df = _load_return_period()
    df = df.merge(rp_df, on="Country", how="left")
    df["rp10_risk"]  = df["rp10_risk"].fillna(0.0)
    df["rp100_risk"] = df["rp100_risk"].fillna(0.0)
    print(f"    [rp]  return period risk joined: "
          f"{df['rp10_risk'].gt(0).sum()}/{len(df)} events enriched")

    # ── Enrichment 3: satellite spectral features ──────────────────────────
    spec_df = _load_spectral_features()
    df = df.merge(spec_df, left_on="ID", right_on="flood_id", how="left")
    spectral_cols = ["mean_MNDWI", "mean_NDVI", "flood_pixel_frac", "mean_B4", "mean_B5"]
    for c in spectral_cols:
        df[c] = df[c].fillna(0.0)
    print(f"    [sat] spectral features joined: "
          f"{df['mean_MNDWI'].ne(0).sum()}/{len(df)} events enriched")

    # ── Composite label ────────────────────────────────────────────────────
    median_exposed = df.loc[df["log_exposed"] > 0, "log_exposed"].median()
    median_exposed = median_exposed if pd.notna(median_exposed) else 0.0
    high_exposure  = df["log_exposed"] > median_exposed
    high_severity  = df["Severity"] >= 2
    df["label"]    = (high_severity | high_exposure).astype(int)
    print(f"    [label] composite label — positives: "
          f"{df['label'].sum()}/{len(df)} "
          f"(was {high_severity.sum()} with Severity>=2 only)")

    feature_cols = [
        "lat", "long", "month", "dayofyear", "hour",
        "duration_d", "cause_code", "log_dead", "log_displaced", "Severity",
        # enriched
        "log_exposed", "exposed_area",
        "rp10_risk", "rp100_risk",
        "mean_MNDWI", "mean_NDVI", "flood_pixel_frac", "mean_B4", "mean_B5",
    ]
    df[feature_cols] = df[feature_cols].fillna(0.0)

    df["hazard_type"] = "flood"
    df["event_id"]    = "FL_" + df["ID"].astype(str)
    df["LAT"]         = df["lat"]
    df["LON"]         = df["long"]

    out = df[["event_id", "time", "LAT", "LON", "label", "hazard_type"] + feature_cols].copy()
    out = _add_rainfall(out)         # monthly cell rainfall
    out = _add_event_features(out)   # antecedent rainfall + terrain (by event_id)
    print(f"    → {len(out):,} rows  |  events: {out['label'].sum():,}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3.  LANDSLIDE  (NASA GLC)
# ══════════════════════════════════════════════════════════════════════════════

def load_landslide() -> pd.DataFrame:
    print("  Loading landslide catalog …")
    df = pd.read_csv(
        DATA_ROOT / "landslide" / "Global_Landslide_Catalog_Export_rows.csv",
        parse_dates=["event_date"]
    )
    df = df[df["country_name"].isin(ASEAN_COUNTRIES)].copy()
    df = df.dropna(subset=["latitude", "longitude"])

    df["time"]      = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["time"])
    df["month"]     = df["time"].dt.month
    df["dayofyear"] = df["time"].dt.dayofyear
    df["hour"]      = 0

    trigger_map = {
        "rain": 0, "downpour": 1, "monsoon": 2, "earthquake": 3,
        "continuous_rain": 4, "flooding": 5, "tropical_cyclone": 6,
        "unknown": 7, "other": 8,
    }
    df["trigger_code"] = df["landslide_trigger"].str.lower().str.strip().map(
        trigger_map).fillna(8)

    cat_map = {
        "landslide": 0, "mudslide": 1, "rock_fall": 2,
        "debris_flow": 3, "lahar": 4, "complex": 5, "other": 6,
    }
    df["cat_code"] = df["landslide_category"].str.lower().str.strip().map(
        cat_map).fillna(6)

    df["fatality_count"] = df["fatality_count"].fillna(0)
    df["injury_count"] = df["injury_count"].fillna(0)
    df["log_fatality"] = np.log1p(df["fatality_count"])
    df["log_injury"]   = np.log1p(df["injury_count"])

    df["label"] = (df["fatality_count"] > 0).astype(int)

    feature_cols = [
        "latitude", "longitude", "month", "dayofyear", "hour",
        "trigger_code", "cat_code", "log_fatality", "log_injury",
    ]
    df[feature_cols] = df[feature_cols].fillna(0.0)

    df["hazard_type"] = "landslide"
    df["event_id"]    = "LS_" + df["event_id"].astype(str)
    df["LAT"]         = df["latitude"]
    df["LON"]         = df["longitude"]

    out = df[["event_id", "time", "LAT", "LON", "label", "hazard_type"] + feature_cols].copy()
    out = _add_rainfall(out)         # monthly cell rainfall
    out = _add_event_features(out)   # antecedent rainfall + terrain (by event_id)
    print(f"    → {len(out):,} rows  |  events: {out['label'].sum():,}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 4.  GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_spatial_graph(
    nodes_df: pd.DataFrame,
    radius_deg: float = 2.5,
    max_neighbors: int = 16,
):
    from sklearn.neighbors import BallTree

    if len(nodes_df) <= 1:
        return np.empty((2, 0), dtype=np.int64)

    coords     = nodes_df[["LAT", "LON"]].to_numpy(dtype=float)
    coords_rad = np.radians(coords)
    tree       = BallTree(coords_rad, metric="haversine")
    radius_rad = np.radians(radius_deg)

    k = min(max_neighbors + 1, len(nodes_df))
    distances, indices = tree.query(coords_rad, k=k)
    src_grid = np.broadcast_to(np.arange(len(nodes_df))[:, None], indices.shape)
    mask = (indices != src_grid) & (distances <= radius_rad)

    src = src_grid[mask].astype(np.int64)
    dst = indices[mask].astype(np.int64)

    return np.vstack((src, dst))


# ══════════════════════════════════════════════════════════════════════════════
# 5.  UNIFIED DATASET BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_unified_dataset(save_path: str = "processed/"):
    Path(save_path).mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Loading raw datasets …")
    frames = []
    # Typhoon is optional — skip cleanly if the CSVs aren't present
    # (we exclude it from rainfall/terrain work; OpenWeather handles it live).
    try:
        typh_all = pd.concat(
            [load_typhoon(s) for s in ("train", "val", "test")],
            ignore_index=True
        )
        frames.append(typh_all)
    except FileNotFoundError:
        print("  typhoon CSVs not found — skipping typhoon (flood + landslide only)")

    frames.append(load_flood())
    frames.append(load_landslide())

    # ── Align to common schema ──
    common_features = ["LAT", "LON", "month", "dayofyear", "hour"]
    for df in frames:
        for c in common_features:
            if c not in df.columns:
                df[c] = 0.0

    hazard_enc = {"typhoon": 0, "flood": 1, "landslide": 2}
    for df in frames:
        df["hazard_code"] = df["hazard_type"].map(hazard_enc)

    print("\n[2/5] Merging datasets …")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.fillna(0.0, inplace=True)
    combined.sort_values("time", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # ── Cross-hazard engineered features (non-zero for all rows) ──
    combined["lat_sin"] = np.sin(np.radians(combined["LAT"]))
    combined["lon_sin"] = np.sin(np.radians(combined["LON"]))
    combined["lat_cos"] = np.cos(np.radians(combined["LAT"]))
    combined["lon_cos"] = np.cos(np.radians(combined["LON"]))
    combined["month_sin"] = np.sin(2 * np.pi * combined["month"] / 12)
    combined["month_cos"] = np.cos(2 * np.pi * combined["month"] / 12)
    combined["day_sin"] = np.sin(2 * np.pi * combined["dayofyear"] / 365)
    combined["day_cos"] = np.cos(2 * np.pi * combined["dayofyear"] / 365)

    print(f"  Total rows: {len(combined):,}")
    print(f"  Label distribution:\n{combined['label'].value_counts().to_string()}")

    print("\n[3/5] Building spatial graph …")
    # ── Snap exact coordinates to grid CELLS so nodes recur over time ──
    # Each event used to create a brand-new node (median 2 events/node → no
    # temporal pattern to learn). Snapping to 0.5° cells turns nodes into
    # persistent geographic regions (median ~29 events/node), which is what
    # the ST-GNN actually needs and matches the "settlement" node concept.
    CELL = 0.5  # degrees (~55 km); lower = finer regions, fewer events/node
    combined["cell_lat"] = (combined["LAT"] / CELL).round() * CELL
    combined["cell_lon"] = (combined["LON"] / CELL).round() * CELL

    node_df = combined[["cell_lat", "cell_lon"]].drop_duplicates().reset_index(drop=True)
    node_df["node_id"] = node_df.index
    node_df["LAT"] = node_df["cell_lat"]   # node coords = cell center
    node_df["LON"] = node_df["cell_lon"]
    combined = combined.merge(
        node_df[["cell_lat", "cell_lon", "node_id"]],
        on=["cell_lat", "cell_lon"], how="left"
    )

    edge_index = build_spatial_graph(node_df, radius_deg=2.5, max_neighbors=16)
    print(f"  Nodes: {len(node_df):,}  |  Edges: {edge_index.shape[1]:,}  "
          f"(grid cells @ {CELL}°)")

    # Bug #1 fix: assert no NaN node_ids (would silently stack events onto node 0)
    assert combined["node_id"].notna().all(), \
        f"node_id merge produced NaNs for {combined['node_id'].isna().sum()} rows — " \
        "check for LAT/LON mismatches between combined and node_df"

    # ── Node-level historical features (vary between nodes, non-zero) ──
    print("  Computing node-level historical features …")
    combined = combined.sort_values("time").reset_index(drop=True)

    # node_degree: how connected this location is in the spatial graph
    degree = np.bincount(edge_index[0], minlength=len(node_df)).astype(np.float32)
    node_df["node_degree"] = degree
    combined = combined.merge(node_df[["node_id", "node_degree"]], on="node_id", how="left")

    # node_hist_count: how many past events occurred at this node (excludes current row)
    combined["node_hist_count"] = combined.groupby("node_id").cumcount().astype(np.float32)

    # node_hist_pos_rate: fraction of past events at this node that were positive
    combined["_cum_pos"] = combined.groupby("node_id")["label"].transform(
        lambda x: x.shift(1, fill_value=0).cumsum()
    )
    combined["node_hist_pos_rate"] = (
        combined["_cum_pos"] / combined["node_hist_count"].clip(lower=1)
    ).astype(np.float32)
    combined.drop(columns=["_cum_pos"], inplace=True)

    print(f"    node_degree range: {combined['node_degree'].min():.0f}–{combined['node_degree'].max():.0f}")
    print(f"    node_hist_count range: {combined['node_hist_count'].min():.0f}–{combined['node_hist_count'].max():.0f}")

    print("\n[4/5] Normalising features …")
    scale_cols = [
        c for c in combined.select_dtypes(include=np.number).columns
        if c not in ("label", "hazard_code", "node_id")
    ]

    # Bug #6 fix: compute split indices first, then fit scaler on TRAIN only
    n         = len(combined)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    scaler = StandardScaler()
    train_part = combined.iloc[:train_end]
    scaler.fit(train_part[scale_cols])
    combined[scale_cols] = scaler.transform(combined[scale_cols])

    print("\n[5/5] Saving processed data …")
    combined.to_parquet(f"{save_path}/combined.parquet", index=False)
    node_df.to_parquet(f"{save_path}/nodes.parquet",    index=False)
    np.save(f"{save_path}/edge_index.npy", edge_index)

    with open(f"{save_path}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Time-ordered splits — no leakage
    train_df = combined.iloc[:train_end]
    val_df   = combined.iloc[train_end:val_end]
    test_df  = combined.iloc[val_end:]

    # ── Oversample minority class in training set ──
    pos = train_df[train_df["label"] == 1]
    neg = train_df[train_df["label"] == 0]
    if len(pos) > 0 and len(neg) > len(pos):
        oversample_ratio = min(len(neg) // len(pos), 5)
        if oversample_ratio > 1:
            pos_upsampled = pos.sample(n=len(pos) * oversample_ratio,
                                       replace=True, random_state=42)
            train_df = pd.concat([neg, pos_upsampled], ignore_index=True)
            train_df = train_df.sort_values("time").reset_index(drop=True)
            print(f"  [oversample] positives upsampled {oversample_ratio}x "
                  f"→ {len(train_df):,} training rows")

    splits = {
        "train": train_df,
        "val":   val_df,
        "test":  test_df,
    }
    for name, df in splits.items():
        df.to_parquet(f"{save_path}/{name}.parquet", index=False)
        print(f"  {name}: {len(df):,} rows  (positives: {df['label'].sum():,})")

    print("\n✓ Dataset ready at:", save_path)
    return combined, node_df, edge_index, scaler


if __name__ == "__main__":
    build_unified_dataset()
