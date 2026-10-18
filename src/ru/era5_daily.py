"""Daily ERA5 2-m temperature (mean/max/min) for the 11 ERCOT cities, 2005-2026,
from the free Open-Meteo archive API (CC BY 4.0), for monthly degree days in Module A2."""
import json, time, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd
from .wx import CITY_ZONE

COORD = {"houston": (29.76, -95.37), "dallas": (32.78, -96.80), "fortworth": (32.75, -97.33), "sanantonio": (29.42, -98.49),
         "austin": (30.27, -97.74), "corpus": (27.80, -97.40), "mcallen": (26.20, -98.23), "midland": (31.99, -102.08),
         "abilene": (32.45, -99.73), "tyler": (32.35, -95.30), "wichitafalls": (33.91, -98.49)}
OUT = Path("data/raw/era5_daily")


def main():
    for c, (lat, lon) in COORD.items():
        f = OUT / f"{c}.json"
        if f.exists():
            continue
        q = dict(latitude=lat, longitude=lon, start_date="2005-01-01", end_date="2026-09-30", models="era5",
                 daily="temperature_2m_mean,temperature_2m_max,temperature_2m_min", timezone="America/Chicago")
        url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(q)
        for i in range(6):
            try:
                with urllib.request.urlopen(url, timeout=180) as r:
                    f.write_bytes(r.read()); break
            except Exception as ex:
                print(c, "retry", i, ex); time.sleep(20 * (i + 1))
        print(c, f.stat().st_size, flush=True); time.sleep(5)
    rows = []
    for c in COORD:
        d = pd.DataFrame(json.loads((OUT / f"{c}.json").read_text())["daily"]); d["city"] = c; rows.append(d)
    d = pd.concat(rows); d["time"] = pd.to_datetime(d["time"])
    d.to_parquet("data/processed/era5_daily_cities.parquet", index=False)
    print(d.groupby("city")["time"].agg(["min", "max", "count"]).head(3))


if __name__ == "__main__":
    main()
