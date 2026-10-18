"""Module A1 feature table (PROTOCOL.md): one row per (series, real UTC hour of
target day T), with availability-aware profile lags and issue-time-safe zone weather.

Availability: profile day d is known at the issue (18:00 on D = T-1) only if ADDTIME(d) < D.
Lags whose day is not known are set to NaN; `known_max_day` records the latest known day per row.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .calendar import holiday_table

TZ = "America/Chicago"
CLASSES = ["RESLOWR", "RESHIWR", "BUSNODEM", "BUSLOLF", "BUSMEDLF", "BUSHILF"]
ZONES = ["COAST", "EAST", "FWEST", "NCENT", "NORTH", "SCENT", "SOUTH", "WEST"]
SERIES = [f"{c}_{z}" for c in CLASSES for z in ZONES]
LAGS = [6, 7, 8, 13, 14, 21, 28, 364]
CACHE = Path("data/processed/features_A1.parquet")


def load_profiles():
    h = pd.read_parquet("data/processed/profiles_hourly.parquet")
    h = h[h["series"].isin(SERIES)].copy()
    loc = h["hour_utc"].dt.tz_convert(TZ)
    h["slot"] = loc.dt.hour + 1                       # hour-ending style 1..24 on local clock
    return h


def slot_matrix(h):
    """(series, op_date) x slot 1..24; fall-back duplicate hour averaged, spring gap interpolated."""
    m = h.groupby(["series", "op_date", "slot"])["kwh"].mean().unstack("slot")
    m = m.reindex(columns=range(1, 25))
    gap = m[3].isna() & m[2].notna() & m[4].notna()
    m.loc[gap, 3] = (m.loc[gap, 2] + m.loc[gap, 4]) / 2
    return m


def known_table():
    d = pd.read_parquet("data/processed/profile_days.parquet")
    d = d[d["series"].isin(SERIES)].copy()
    d["addtime"] = pd.to_datetime(d["addtime"]).dt.normalize()
    return d.drop_duplicates(["series", "op_date"], keep="last").set_index(["series", "op_date"])["addtime"]


def build(first="2017-02-01", last="2026-09-30"):
    h = load_profiles()
    M = slot_matrix(h)
    added = known_table()
    tgt = h[(h["op_date"] >= first) & (h["op_date"] <= last)].rename(columns={"op_date": "T", "kwh": "y"}).copy()
    tgt["issue_date"] = tgt["T"] - pd.Timedelta(days=1)
    tgt["issue_utc"] = (tgt["issue_date"] + pd.Timedelta(hours=18)).dt.tz_localize(TZ).dt.tz_convert("UTC")
    tgt["horizon_h"] = ((tgt["hour_utc"] - tgt["issue_utc"]).dt.total_seconds() / 3600).astype(int) + 1
    tgt["cls"] = tgt["series"].str.split("_").str[0]
    tgt["zone"] = tgt["series"].str.split("_").str[1]
    keys = list(zip(tgt["series"], tgt["T"]))
    for k in LAGS:
        d = tgt["T"] - pd.Timedelta(days=k)
        lag_keys = pd.MultiIndex.from_arrays([tgt["series"], d])
        stacked = M.stack()
        v = stacked.reindex(pd.MultiIndex.from_arrays([tgt["series"], d, tgt["slot"]])).values
        ok = added.reindex(lag_keys).values < tgt["issue_date"].values
        tgt[f"lag{k}"] = np.where(ok, v, np.nan)
    # daily means and a 7-day level from known days (T-12..T-6)
    dm = M.mean(axis=1)
    for k in [6, 7]:
        idx = pd.MultiIndex.from_arrays([tgt["series"], tgt["T"] - pd.Timedelta(days=k)])
        ok = added.reindex(idx).values < tgt["issue_date"].values
        tgt[f"dmean{k}"] = np.where(ok, dm.reindex(idx).values, np.nan)
    lv = []
    for k in range(6, 13):
        idx = pd.MultiIndex.from_arrays([tgt["series"], tgt["T"] - pd.Timedelta(days=k)])
        ok = added.reindex(idx).values < tgt["issue_date"].values
        lv.append(np.where(ok, dm.reindex(idx).values, np.nan))
    tgt["level7"] = np.nanmean(np.column_stack(lv), axis=1)
    # latest known day (for the cutoff test)
    kd = []
    for s, g in tgt.groupby("series")[["T", "issue_date"]]:
        a = added.loc[s].sort_index()
        # latest op_date whose addtime < issue_date: since addtime is nondecreasing-ish, use cummax trick
        ops = a.index.values
        adds = a.values
        order = np.argsort(adds)
        adds_s, ops_cummax = adds[order], np.maximum.accumulate(ops[order])
        pos = np.searchsorted(adds_s, g["issue_date"].values, side="left") - 1
        kd.append(pd.Series(np.where(pos >= 0, ops_cummax[np.clip(pos, 0, None)], np.datetime64("NaT")), index=g.index))
    tgt["known_max_day"] = pd.concat(kd).reindex(tgt.index)
    # calendar
    loc = tgt["hour_utc"].dt.tz_convert(TZ)
    tgt["hour"] = loc.dt.hour
    tgt["dow"] = tgt["T"].dt.dayofweek
    tgt["month"] = tgt["T"].dt.month
    tgt["doy"] = tgt["T"].dt.dayofyear
    hol = holiday_table("2016-01-01", "2027-01-01")
    hol.index = pd.to_datetime(hol.index)
    tgt["holiday"] = (hol["holiday"].reindex(tgt["T"]).values != "none").astype(int)
    tgt["xmas_week"] = hol["xmas_week"].reindex(tgt["T"]).values
    tgt["cls_id"] = tgt["cls"].map({c: i for i, c in enumerate(CLASSES)})
    tgt["zone_id"] = tgt["zone"].map({z: i for i, z in enumerate(ZONES)})
    return tgt.drop(columns=["n"]).reset_index(drop=True)


def load_features(rebuild=False):
    if CACHE.exists() and not rebuild:
        return pd.read_parquet(CACHE)
    df = build()
    df.to_parquet(CACHE, index=False)
    return df


if __name__ == "__main__":
    import time
    t = time.time(); df = load_features(rebuild=True)
    print(df.shape, round(time.time() - t), "s")
    print(df.groupby(df["T"].dt.year)[["lag6", "lag7", "lag364", "level7"]].apply(lambda x: x.isna().mean().round(3)))
    lagd = (df["T"] - df["known_max_day"]).dt.days
    print("T - known_max_day:", lagd.describe()[["min", "50%", "max"]].to_dict())
