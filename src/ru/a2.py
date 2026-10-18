"""Module A2 (PROTOCOL.md): monthly Texas retail sales (EIA-861M), issued on the
first day of month M for target months M, M+1, M+2. Sales of month m are known from the first day of
month m+3. Weather: monthly CDD/HDD (base 18.3 C) from ERA5 daily means of the 11 cities with fixed
weights (approximate metro populations); FORECASTS use climatology (mean
of the same calendar month over the 10 previous complete years), BACKCASTS use actual degree days.
Usage: python -m src.ru.a2 [--holdout]
"""
import argparse, datetime as dt, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from .wx import CITY_ZONE

REG = Path("experiments/registry_A2.jsonl")
VAL_ISSUES = pd.date_range("2021-01-01", "2023-12-01", freq="MS")
HOLD_ISSUES = pd.date_range("2024-01-01", "2026-05-01", freq="MS")
KNOWN_LAG = 3          # month m known from month m+3
BASE = 18.3


def degree_days():
    d = pd.read_parquet("data/processed/era5_daily_cities.parquet")
    # approximate metro populations (millions, ~2023, rounded; design choice): ERCOT count snapshots
    # under-cover municipal utilities (San Antonio, Austin), so they are not used for statewide weights
    pop = {"houston": 7.3, "dallas": 5.0, "fortworth": 3.1, "sanantonio": 2.7, "austin": 2.5, "corpus": 0.45,
           "mcallen": 0.9, "midland": 0.3, "abilene": 0.18, "tyler": 0.24, "wichitafalls": 0.15}
    tot = sum(pop.values()); w = {k: v / tot for k, v in pop.items()}
    d["w"] = d["city"].map(w)
    t = d.assign(tw=d["temperature_2m_mean"] * d["w"]).groupby("time")["tw"].sum()
    daily = pd.DataFrame({"t": t})
    daily["cdd"] = np.maximum(daily["t"] - BASE, 0); daily["hdd"] = np.maximum(BASE - daily["t"], 0)
    m = daily.resample("MS")[["cdd", "hdd"]].sum()
    return m, w


def climatology(dd, month_start, issue):
    """Same-calendar-month mean over the 10 complete years before the issue year (all known)."""
    yrs = range(issue.year - 10, issue.year)
    idx = [pd.Timestamp(y, month_start.month, 1) for y in yrs]
    return dd.reindex(idx).mean()


def rows_for_issue(sales, dd, issue, weather):
    out = []
    known = sales[sales["month"] <= issue - pd.DateOffset(months=KNOWN_LAG)]
    for sector, g in known.groupby("sector"):
        g = g.set_index("month").sort_index()
        last = g.index.max()
        for h in range(3):
            t = issue + pd.DateOffset(months=h)
            wx = dd.loc[t] if weather == "actual" else climatology(dd, t, issue)
            yoy = t - pd.DateOffset(years=1)
            cust_growth = g["customers"].iloc[-1] / g["customers"].iloc[-13] if len(g) > 13 else 1.0
            out.append(dict(issue=issue, sector=sector, target=t, h=h, lead=(t.year - last.year) * 12 + t.month - last.month,
                            cdd=wx["cdd"], hdd=wx["hdd"], sales_yoy=g["sales_mwh"].get(yoy, np.nan),
                            cdd_yoy=dd["cdd"].get(yoy, np.nan), hdd_yoy=dd["hdd"].get(yoy, np.nan),
                            cust_last=g["customers"].iloc[-1], cust_growth=cust_growth, last_known=last))
    return out


class SeasonalNaive:
    name = "seasonal_naive"
    def predict(self, r, hist): return r["sales_yoy"]


class YoYGrowth:
    name = "yoy_x_customer_growth"
    def predict(self, r, hist): return r["sales_yoy"] * r["cust_growth"]


class DDRegression:
    """kWh per customer ~ month dummies + month-specific CDD/HDD slopes (via season groups) + trend,
    fitted on known months (from `start`), times extrapolated customers."""
    def __init__(self, start="2010-01-01", alpha=1.0, per_customer=True, name="dd_regression"):
        self.start, self.alpha, self.per_customer, self.name = (pd.Timestamp(start) if start else None), alpha, per_customer, name

    @staticmethod
    def _X(month, cdd, hdd, t):
        X = np.zeros((len(month), 12 + 4 + 1))
        for i, (mo, c, hh, tt) in enumerate(zip(month, cdd, hdd, t)):
            X[i, mo - 1] = 1
            summer = mo in (5, 6, 7, 8, 9, 10)
            X[i, 12] = c / 100 if summer else 0; X[i, 13] = c / 100 if not summer else 0
            X[i, 14] = hh / 100 if not summer else 0; X[i, 15] = hh / 100 if summer else 0
            X[i, 16] = tt
        return X

    def predict(self, r, hist):
        g = hist[(hist["sector"] == r["sector"]) & (hist["month"] >= self.start) & (hist["month"] <= r["last_known"])]
        t0 = pd.Timestamp("2010-01-01")
        tt = lambda m: (m.year - t0.year) + (m.month - 1) / 12
        X = self._X(g["month"].dt.month.values, g["cdd"].values, g["hdd"].values, [tt(m) for m in g["month"]])
        y = g["kwh_per_customer"].values if self.per_customer else g["sales_mwh"].values / 1e6
        mdl = Ridge(alpha=self.alpha).fit(X, y)
        x = self._X([r["target"].month], [r["cdd"]], [r["hdd"]], [tt(r["target"])])
        if not self.per_customer:
            return float(mdl.predict(x)[0]) * 1e6
        months_ahead = r["lead"]
        cust = r["cust_last"] * r["cust_growth"] ** (months_ahead / 12)
        return float(mdl.predict(x)[0]) * cust / 1000


EXPERIMENTS = {
    "M0_seasonal_naive": (lambda: SeasonalNaive(), "clim", "Same month last year (raw) is the floor."),
    "M1_yoy_customer_growth": (lambda: YoYGrowth(), "clim", "Last-year same month scaled by 12-month customer growth beats M0."),
    "M2_dd_regression_clim": (lambda: DDRegression(), "clim", "Degree-day regression of usage per customer with climatological weather beats M1 by removing last year's weather anomaly."),
    "M3_dd_regression_5y_clim": (lambda: DDRegression(start=None), "clim", "Same as M2 but trained on the last 5 known years only (recent behaviour, e.g. solar/efficiency) - set in code."),
    "M4_dd_regression_3y_clim": (lambda: DDRegression(start=None), "clim", "Usage per customer is changing fast (commercial kWh/customer +19% 2021-2025), so a 3-year training window beats the 5-year window (M3)."),
    "M5_dd_total_5y_clim": (lambda: DDRegression(start=None, per_customer=False), "clim", "Modeling total sector sales (degree days + trend, 5-year window) instead of usage x extrapolated customers beats M3, because commercial customer counts do not track commercial energy."),
    "M6_dd_sector_windows_clim": (lambda: DDRegression(start=None), "clim", "Sector-specific windows chosen from M3/M4 validation (residential 5 y, commercial 3 y); expected ~= best of both per sector (selection made after seeing M3/M4, so mildly optimistic)."),
    "MX_dd_regression_actual": (lambda: DDRegression(), "actual", "BACKCAST reference (not a forecast): M2 with actual degree days measures the weather-normalized model error."),
}


def run(name, issues):
    factory, weather, hyp = EXPERIMENTS[name]
    sales = pd.read_parquet("data/processed/eia_tx_monthly.parquet")
    sales = sales[sales["sector"].isin(["residential", "commercial"])].copy()
    dd, w = degree_days()
    sales = sales.merge(dd, left_on="month", right_index=True, how="left")
    rows = []
    for issue in issues:
        hist = sales[sales["month"] <= issue - pd.DateOffset(months=KNOWN_LAG)]
        m = factory()
        if name in ("M3_dd_regression_5y_clim", "M5_dd_total_5y_clim"):
            m.start = issue - pd.DateOffset(years=5, months=KNOWN_LAG)
        if name == "M4_dd_regression_3y_clim":
            m.start = issue - pd.DateOffset(years=3, months=KNOWN_LAG)
        for r in rows_for_issue(sales, dd, issue, weather):
            if name == "M6_dd_sector_windows_clim":
                m.start = issue - pd.DateOffset(years=5 if r["sector"] == "residential" else 3, months=KNOWN_LAG)
            assert r["last_known"] <= issue - pd.DateOffset(months=KNOWN_LAG)
            r["pred"] = m.predict(r, hist)
            r["y"] = float(sales[(sales["sector"] == r["sector"]) & (sales["month"] == r["target"])]["sales_mwh"].iloc[0])
            rows.append(r)
    return pd.DataFrame(rows), hyp, weather


def summarize(df):
    df = df.assign(ape=100 * (df["pred"] - df["y"]).abs() / df["y"], err=df["pred"] - df["y"])
    out = {"mape": df["ape"].mean(), "mae_gwh": df["err"].abs().mean() / 1000, "bias_pct": 100 * df["err"].sum() / df["y"].sum()}
    for (s, h), g in df.groupby(["sector", "h"]):
        out[f"mape_{s[:3]}_h{h}"] = g["ape"].mean()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("exps", nargs="*"); ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args(argv)
    if a.holdout and not Path("configs/final_A2.yaml").exists():
        sys.exit("A2 holdout locked: create configs/final_A2.yaml first.")
    if a.holdout:
        import yaml
        allowed = yaml.safe_load(Path("configs/final_A2.yaml").read_text())["holdout_set"]
        done = {json.loads(l)["experiment"] for l in REG.open() if json.loads(l)["phase"] == "holdout"} if REG.exists() else set()
        bad = [e for e in a.exps if e not in allowed or e in done]
        if not a.exps or bad:
            sys.exit(f"Not allowed on holdout: {bad or 'empty list'}")
    REG.parent.mkdir(exist_ok=True)
    for name in a.exps or list(EXPERIMENTS):
        df, hyp, weather = run(name, HOLD_ISSUES if a.holdout else VAL_ISSUES)
        s = summarize(df)
        run_id = f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}{'_HOLDOUT' if a.holdout else ''}"
        rdir = Path("experiments/runs_A2") / run_id; rdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(rdir / "predictions.csv", index=False)
        rec = {"run_id": run_id, "experiment": name, "phase": "holdout" if a.holdout else "validation",
               "hypothesis": hyp, "weather": weather, "metrics": s}
        with REG.open("a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        print(f"{name:28s} MAPE={s['mape']:5.2f}%  res h0/h1/h2={s['mape_res_h0']:.1f}/{s['mape_res_h1']:.1f}/{s['mape_res_h2']:.1f}  "
              f"com h0/h1/h2={s['mape_com_h0']:.1f}/{s['mape_com_h1']:.1f}/{s['mape_com_h2']:.1f}  bias={s['bias_pct']:.2f}%")


if __name__ == "__main__":
    main()
