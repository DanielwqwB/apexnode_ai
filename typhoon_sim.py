"""
SentryMesh — Typhoon Simulator ("fake weather feed" for demos)

Simulates a typhoon making landfall over Bicol and pushes synthetic heavy-rainfall
readings for real Naga-area barangays to the prediction API. Escalates intensity
over several ticks (approach -> landfall -> aftermath) so you can demo CRITICAL
alerts + equity-first rescue ranking on command — no real typhoon required.

Usage:
    python typhoon_sim.py                       # local AI service (:8000)
    python typhoon_sim.py https://apexnode-ai.onrender.com
    python typhoon_sim.py http://localhost:3000/predictions/flood  --backend

The override added to serve.py makes the API use these supplied rainfall values
instead of fetching live weather.
"""
import json
import sys
import time
import urllib.request

# Real Bicol / Naga-area points (lat, lon, elevation m)
BARANGAYS = [
    ("Naga City - Centro",     13.6218, 123.1948, 5),
    ("Naga - Triangulo",       13.6260, 123.1860, 4),
    ("Naga - Peñafrancia",     13.6300, 123.1820, 6),
    ("Calabanga (coastal)",    13.7050, 123.2210, 3),
    ("Camaligan (riverside)",  13.6230, 123.1660, 4),
    ("Pili (highland)",        13.5560, 123.2940, 45),
    ("Mt. Isarog slope",       13.6600, 123.3800, 320),
]

# Intensity ticks: (label, rainfall multiplier 0..1)
TICKS = [
    ("T-36h  calm before",   0.05),
    ("T-24h  outer bands",   0.18),
    ("T-12h  approaching",   0.45),
    ("LANDFALL  eyewall",    1.00),
    ("T+12h  aftermath",     0.55),
]


def rainfall_for(intensity: float) -> dict:
    """Scale a peak-typhoon rainfall profile by intensity."""
    s = intensity
    return {"rain_1d": 160 * s, "rain_3d": 360 * s, "rain_7d": 560 * s,
            "rain_30d": 820 * s, "rain_max3": 200 * s, "rain_api": 230 * s}


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def run_ai_direct(base: str):
    """Hit the AI service /predict/flood (batch). Shows model + equity ranking."""
    url = base if "/predict" in base else base.rstrip("/") + "/predict/flood"
    print(f"Typhoon simulator (AI-direct) -> {url}\n")
    for label, mult in TICKS:
        rain = rainfall_for(mult)
        nodes = [{"node_id": name, "lat": lat, "lon": lon, "elev": elev,
                  "slope_deg": 0.5, "relief": 12, **rain}
                 for name, lat, lon, elev in BARANGAYS]
        try:
            out = post(url, {"nodes": nodes})
        except Exception as e:
            print(f"  ! request failed: {e}"); return
        print(f"=== {label}  (rain_7d≈{rain['rain_7d']:.0f}mm) — "
              f"{out.get('alert_count', 0)}/{out.get('total_nodes', 0)} alerts ===")
        for n in out.get("nodes", []):
            ev = n.get("event_prob", n.get("probability"))
            print(f"  rank {n.get('rescue_rank','?'):>2}: {str(n.get('node_id')):24s} "
                  f"P={ev:.2f} equity={n.get('equity_score', 0):.2f} {n.get('alert_level')}")
        print()
        time.sleep(1.5)


def run_through_backend(base: str):
    """POST each barangay to the NestJS backend /predictions/flood so it saves to
    Postgres and the Flutter app sees it via GET /predictions. One call per node."""
    predict_url = base.rstrip("/") + "/predictions/flood"
    list_url = base.rstrip("/") + "/predictions?hazard_type=flood"
    print(f"Typhoon simulator (through backend) -> {predict_url}\n")
    for label, mult in TICKS:
        rain = rainfall_for(mult)
        print(f"=== {label}  (rain_7d≈{rain['rain_7d']:.0f}mm) ===")
        for name, lat, lon, elev in BARANGAYS:
            payload = {"location_label": name, "latitude": lat, "longitude": lon,
                       "elev": elev, "slope_deg": 0.5, "relief": 12, **rain}
            try:
                res = post(predict_url, payload)
            except Exception as e:
                print(f"  ! {name}: {e}"); continue
            print(f"  {name:24s} risk={res.get('risk_level')} conf={res.get('confidence')}")
        print()
        time.sleep(2)
    print("App reads these from GET /predictions. Latest saved rows:")
    try:
        items = get(list_url).get("items", [])[:7]
        for it in items:
            print(f"  {it.get('location_label')}: {it.get('risk_level')} "
                  f"({it.get('confidence')}) @ {it.get('created_at','')[:19]}")
    except Exception as e:
        print(f"  ! could not list: {e}")


def main():
    args = [a for a in sys.argv[1:]]
    backend = "--backend" in args
    http = [a for a in args if a.startswith("http")]
    base = http[0] if http else ("https://sentrymesh-backend.onrender.com" if backend
                                 else "http://localhost:8000")
    run_through_backend(base) if backend else run_ai_direct(base)


if __name__ == "__main__":
    main()
