"""Zone-level, issue-time-safe weather for Module A1, reusing the weather caches
of the ERCOT load project (~/ML_forecast/data/processed/weather_*.parquet) and its vintage rule:
`<var>_previous_dayN` at valid time v comes from the run initialised at floor6(v - 24N h); a run is
usable PUB_DELAY hours after initialisation; use day1 if usable at the issue, else day2.
Zone value = mean over the zone's cities. Columns:
  wf_*  forecast (GFS), we_* forecast (ECMWF, 2024-02+), wb_* blend (mean where both exist)
  wo_*  ERA5 at target time  -> BACKCAST ONLY, never a forecast input
  wo2_dmean  ERA5 zone daily mean of day T-2 (observed past weather, allowed by the protocol)
"""
from pathlib import Path
import numpy as np
import pandas as pd

P1 = Path.home() / "ML_forecast/data/processed"
CITY_ZONE = {"houston": "COAST", "dallas": "NCENT", "fortworth": "NCENT", "sanantonio": "SCENT", "austin": "SCENT",
             "corpus": "SOUTH", "mcallen": "SOUTH", "midland": "FWEST", "abilene": "WEST", "tyler": "EAST",
             "wichitafalls": "NORTH"}
PUB_DELAY = {"gfs_global": 6, "ecmwf_ifs025": 8}
CACHE = Path("data/processed/wx_A1.parquet")
TZ = "America/Chicago"


def _vals(tab, city, times, col):
    s = tab[col].xs(city, level="city")
    a = s.reindex(times).values
    b = s.reindex(times + pd.Timedelta(hours=1)).values
    return (a + b) / 2                    # hourly-interval mean of instantaneous values at start/end


def forecast_city(tab, model, starts, issue, shift_h=0):
    st = starts - pd.Timedelta(hours=shift_h)
    ends = st + pd.Timedelta(hours=1)
    delay = pd.Timedelta(hours=PUB_DELAY[model])
    ok1 = (ends - pd.Timedelta(hours=24)).floor("6h") + delay <= issue
    ok2 = (ends - pd.Timedelta(hours=48)).floor("6h") + delay <= issue
    assert ok2.all()
    out = {}
    for c in CITY_ZONE:
        v1 = _vals(tab, c, st, "temperature_2m_previous_day1")
        v2 = _vals(tab, c, st, "temperature_2m_previous_day2")
        out[c] = np.where(ok1 & ~np.isnan(v1), v1, v2)
    return pd.DataFrame(out, index=starts)


def zone_mean(cdf):
    z = {}
    for zone in sorted(set(CITY_ZONE.values())):
        cs = [c for c, zz in CITY_ZONE.items() if zz == zone]
        z[zone] = cdf[cs].mean(axis=1, skipna=True)
    return pd.DataFrame(z)


def build(keys):
    """keys: DataFrame with unique (hour_utc, T); returns long table (hour_utc, T, zone, features)."""
    keys = keys.drop_duplicates().sort_values("hour_utc").reset_index(drop=True)
    starts = pd.DatetimeIndex(keys["hour_utc"])
    issue = pd.DatetimeIndex((keys["T"] - pd.Timedelta(days=1) + pd.Timedelta(hours=18)).dt.tz_localize(TZ).dt.tz_convert("UTC"))
    gfs = pd.read_parquet(P1 / "weather_fcst_gfs_global.parquet")
    ecm = pd.read_parquet(P1 / "weather_fcst_ecmwf_ifs025.parquet")
    era = pd.read_parquet(P1 / "weather_obs_era5.parquet")
    parts = {}
    for name, tab, model in [("wf", gfs, "gfs_global"), ("we", ecm, "ecmwf_ifs025")]:
        for sh in [0, 1, 2, 3, 6]:
            z = zone_mean(forecast_city(tab, model, starts, issue, sh))
            parts[(name, sh)] = z
    o = {c: _vals(era, c, starts, "temperature_2m") for c in CITY_ZONE}
    parts[("wo", 0)] = zone_mean(pd.DataFrame(o, index=starts))
    rows = []
    for zone in parts[("wf", 0)].columns:
        d = pd.DataFrame({"hour_utc": keys["hour_utc"], "T": keys["T"], "zone": zone})
        for (name, sh), z in parts.items():
            col = f"{name}_t" if sh == 0 else f"{name}_tlag{sh}"
            d[col] = z[zone].values
        b = d[["wf_t", "we_t"]].mean(axis=1, skipna=True)
        d["wb_t"] = np.where(d["wf_t"].isna(), np.nan, b)       # blend only where GFS exists
        for sh in [1, 2, 3, 6]:
            d[f"wb_tlag{sh}"] = np.where(d[f"wf_tlag{sh}"].isna(), np.nan, d[[f"wf_tlag{sh}", f"we_tlag{sh}"]].mean(axis=1, skipna=True))
        rows.append(d)
    W = pd.concat(rows, ignore_index=True)
    # day aggregates over the target local day (same issue -> allowed)
    for p in ["wf", "wb", "wo"]:
        g = W.groupby(["zone", "T"])[f"{p}_t"]
        W[f"{p}_dmax"], W[f"{p}_dmin"], W[f"{p}_dmean"] = g.transform("max"), g.transform("min"), g.transform("mean")
    # observed ERA5 daily mean of day T-2 (known at the issue)
    dm = W.groupby(["zone", "T"])["wo_t"].mean()
    W["wo2_dmean"] = dm.reindex(pd.MultiIndex.from_arrays([W["zone"], W["T"] - pd.Timedelta(days=2)])).values
    return W


def load_wx(feat, rebuild=False):
    if CACHE.exists() and not rebuild:
        return pd.read_parquet(CACHE)
    W = build(feat[["hour_utc", "T"]])
    W.to_parquet(CACHE, index=False)
    return W


if __name__ == "__main__":
    from .features import load_features
    f = load_features()
    W = load_wx(f, rebuild=True)
    print(W.shape)
    print(W.groupby(W["T"].dt.year)[["wf_t", "we_t", "wb_t", "wo_t", "wo2_dmean"]].apply(lambda x: x.notna().mean().round(3)))
    x = W.dropna(subset=["wf_t", "wo_t"]); print("GFS vs ERA5 zone temp MAE", round((x.wf_t - x.wo_t).abs().mean(), 2))
