"""
Naga City, Bicol — real-event validation of the flood model.

Fetches NASA POWER antecedent rainfall at Naga (13.62N, 123.18E) for real Bicol
typhoon dates vs a calm day, then runs the trained flood model. Naga sits ~5 m
on the Bicol River floodplain — a true flood hotspot.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# torch-bearing import first (avoids numpy-first c10.dll init failure on Windows)
from predict import predict_flood
import pandas as pd
from fetch_antecedent_rainfall import _fetch_point, _features

NAGA = (13.62, 123.18)
ELEV, SLOPE, RELIEF = 5.0, 0.4, 15.0      # low-lying floodplain
RP10, RP100 = 0.07, 0.10                   # Philippines (Aqueduct)

EVENTS = [
    ("Typhoon Kristine/Trami (2024)", "2024-10-24"),
    ("Typhoon Goni/Rolly (2020)",     "2020-11-01"),
    ("Typhoon Usman (2018)",          "2018-12-29"),
    ("Calm dry day (Apr 2023)",       "2023-04-15"),
]

for name, date in EVENTS:
    d = pd.Timestamp(date)
    start = (d - pd.Timedelta(days=35)).strftime("%Y%m%d")
    series = _fetch_point(NAGA[0], NAGA[1], start, d.strftime("%Y%m%d"))
    if not series:
        print(f"{name:34s} | NASA POWER fetch failed (throttled?) — retry later")
        continue
    rf = _features(series, d)
    out = predict_flood(lat=NAGA[0], lon=NAGA[1], month=d.month,
                        elev=ELEV, slope_deg=SLOPE, relief=RELIEF,
                        rp10_risk=RP10, rp100_risk=RP100, **rf)
    print(f"{name:34s} | rain_7d={rf['rain_7d']:6.0f}mm rain_30d={rf['rain_30d']:6.0f}mm "
          f"-> P(flood)={out['flood_probability']:.3f}  {out['alert_level']}")
