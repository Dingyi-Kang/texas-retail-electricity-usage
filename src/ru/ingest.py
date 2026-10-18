"""ERCOT backcasted load profiles (yearly xlsx archives) -> tidy tables.

Each row of the archive = one profile (PType_WZ, e.g. RESLOWR_COAST) x one local operating day,
with 15-minute kWh per ESIID in int_kWh1..100 (92 on spring-forward days, 100 on fall-back days)
and ADDTIME (when ERCOT added the row; used to measure publication lag).
Rows are mapped in file order to the real UTC 15-minute intervals of that local day, then summed
to UTC hours (kWh per ESIID per hour).
Outputs: data/processed/profiles_hourly.parquet, data/processed/profile_days.parquet (ADDTIME),
reports/ingest_manifest.json.
"""
import glob, hashlib, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

TZ = "America/Chicago"
RAW = Path("data/raw/alp")
OUT = Path("data/processed")


def day_intervals_utc(day):
    """Real 15-minute UTC interval starts of a local operating day (92/96/100 values)."""
    start = pd.Timestamp(day).tz_localize(TZ).tz_convert("UTC")
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(TZ).tz_convert("UTC")
    return pd.date_range(start, end, freq="15min", inclusive="left")


def read_year(zpath):
    with zipfile.ZipFile(zpath) as z:
        name = [n for n in z.namelist() if n.endswith(".xlsx")][0]
        with z.open(name) as f:
            sheets = pd.read_excel(f, sheet_name=None)
    frames = []
    for sname, s in sheets.items():
        if "PType_WZ" not in s.columns:
            continue
        s = s.copy()
        s["sheet"] = sname
        frames.append(s)
    return pd.concat(frames, ignore_index=True), name


def to_long(df):
    cols = [c for c in df.columns if str(c).startswith("int_kWh")]
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.drop_duplicates(["PType_WZ", "Date"], keep="last")
    vals = df[cols].to_numpy(dtype=float)
    rows, issues = [], []
    cache = {}
    for i, (pt, d) in enumerate(zip(df["PType_WZ"].values, df["Date"])):
        if d not in cache:
            cache[d] = day_intervals_utc(d)
        idx = cache[d]
        v = vals[i]
        n_ok = int((~np.isnan(v)).sum())
        if n_ok < len(idx):
            issues.append((pt, str(d.date()), n_ok, len(idx)))
        v = v[:len(idx)]
        rows.append(pd.DataFrame({"series": pt, "op_date": d, "utc": idx, "kwh": v}))
    q = pd.concat(rows, ignore_index=True)
    q["hour_utc"] = q["utc"].dt.floor("h")
    h = q.groupby(["series", "op_date", "hour_utc"], as_index=False).agg(kwh=("kwh", "sum"), n=("kwh", "count"))
    return h, issues


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, H, D, all_issues = {"files": {}}, [], [], []
    for z in sorted(glob.glob(str(RAW / "alp_*.zip"))):
        raw, inner = read_year(z)
        h, issues = to_long(raw)
        H.append(h)
        D.append(raw[["PType_WZ", "Date", "ADDTIME", "sheet"]].assign(Date=lambda x: pd.to_datetime(x["Date"])))
        all_issues += issues
        manifest["files"][Path(z).name] = {"sha256": hashlib.sha256(Path(z).read_bytes()).hexdigest(),
                                           "inner": inner, "rows": len(raw), "sheets": sorted(raw["sheet"].unique().tolist()),
                                           "first_day": str(pd.to_datetime(raw["Date"]).min().date()),
                                           "last_day": str(pd.to_datetime(raw["Date"]).max().date())}
        print(Path(z).name, len(raw), manifest["files"][Path(z).name]["first_day"], manifest["files"][Path(z).name]["last_day"], flush=True)
    h = pd.concat(H, ignore_index=True).drop_duplicates(["series", "hour_utc"], keep="last")
    d = pd.concat(D, ignore_index=True).rename(columns={"PType_WZ": "series", "Date": "op_date", "ADDTIME": "addtime"})
    h.to_parquet(OUT / "profiles_hourly.parquet", index=False)
    d.to_parquet(OUT / "profile_days.parquet", index=False)
    # coverage: missing local days per series
    days = pd.date_range(h["op_date"].min(), h["op_date"].max(), freq="D")
    cov = h.groupby("series")["op_date"].nunique()
    manifest["series"] = int(h["series"].nunique())
    manifest["days_expected"] = len(days)
    manifest["series_with_missing_days"] = {k: int(len(days) - v) for k, v in cov.items() if v < len(days)}
    all_days = set(h["op_date"].unique())
    manifest["days_missing_all_series"] = [str(x.date()) for x in days if x not in all_days]
    manifest["incomplete_rows"] = len(all_issues)
    manifest["incomplete_examples"] = all_issues[:10]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/ingest_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, default=str)[:2000])


if __name__ == "__main__":
    main()
