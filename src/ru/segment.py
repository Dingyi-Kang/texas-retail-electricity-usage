"""Module A3 (PROTOCOL.md): descriptive segmentation of ERCOT load-profile series.

Representations (fit period 2021-01-01..2024-12-31; descriptive only, never a forecasting input):
  shape   - mean daily shape (hour value / daily mean) for summer weekdays (Jun-Sep) and winter
            weekdays (Dec-Feb), 48 values;
  weather - daily kWh = a + b_c * CDD + b_h * HDD (zone ERA5 daily mean temperature, base 18.3 C);
            features b_c / a and b_h / a (cooling and heating sensitivity relative to base load);
  both    - standardized concatenation.
KMeans k=2..8 (20 inits); silhouette; stability = ARI across 10 seeds and between fits on
2021-2022 vs 2023-2024; agreement with ERCOT labels (group RES/BUS/NM, PV/DG/WD suffix) by ARI.
Also: ESIID growth of PV/DG profile types from the count snapshots.
Outputs: reports/A3_segmentation.md, reports/A3_clusters.csv, reports/figures/A3_*.png
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

TZ = "America/Chicago"
BASE = 18.3
FIG = Path("reports/figures")


def labels(series):
    pt = series.split("_")[0]
    grp = "RES" if pt.startswith("RES") else "BUS" if pt.startswith("BUS") else "NM"
    tech = next((t for t in ["PV", "DG", "WD"] if pt.endswith(t)), "none")
    return grp, tech, pt


def zone_daily_temp():
    from .era5_daily import COORD
    from .wx import CITY_ZONE
    d = pd.read_parquet("data/processed/era5_daily_cities.parquet")
    d["zone"] = d["city"].map(CITY_ZONE)
    return d.groupby(["zone", "time"])["temperature_2m_mean"].mean()


def representations(h, start, end):
    h = h[(h["op_date"] >= start) & (h["op_date"] <= end)].copy()
    loc = h["hour_utc"].dt.tz_convert(TZ)
    h["hour"] = loc.dt.hour
    h["month"] = h["op_date"].dt.month
    h["wd"] = h["op_date"].dt.dayofweek < 5
    dmean = h.groupby(["series", "op_date"])["kwh"].transform("mean")
    h["rel"] = h["kwh"] / dmean
    shp = []
    for name, months in [("sum", [6, 7, 8, 9]), ("win", [12, 1, 2])]:
        g = h[h["month"].isin(months) & h["wd"]].groupby(["series", "hour"])["rel"].mean().unstack("hour")
        g.columns = [f"{name}_h{c}" for c in g.columns]
        shp.append(g)
    shape = pd.concat(shp, axis=1).dropna()
    daily = h.groupby(["series", "op_date"])["kwh"].sum().reset_index()
    daily["zone"] = daily["series"].str.split("_").str[1]
    zt = zone_daily_temp()
    daily["t"] = zt.reindex(pd.MultiIndex.from_arrays([daily["zone"], daily["op_date"]])).values
    daily["cdd"] = np.maximum(daily["t"] - BASE, 0); daily["hdd"] = np.maximum(BASE - daily["t"], 0)
    rows = {}
    for s, g in daily.dropna().groupby("series"):
        X = np.column_stack([np.ones(len(g)), g["cdd"], g["hdd"], g["op_date"].dt.dayofweek >= 5])
        coef = np.linalg.lstsq(X, g["kwh"].values, rcond=None)[0]
        a = max(coef[0], 1e-6)
        rows[s] = {"cool_rel": coef[1] / a, "heat_rel": coef[2] / a, "weekend_rel": coef[3] / a, "base_kwh_day": coef[0]}
    weather = pd.DataFrame(rows).T
    return shape, weather


def cluster(X, k, seed=0):
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    h = pd.read_parquet("data/processed/profiles_hourly.parquet")
    h = h[h["kwh"].notna()]
    shape, weather = representations(h, "2021-01-01", "2024-12-31")
    common = shape.index.intersection(weather.index)
    shape, weather = shape.loc[common], weather.loc[common]
    reps = {"shape": StandardScaler().fit_transform(shape.values),
            "weather": StandardScaler().fit_transform(weather[["cool_rel", "heat_rel", "weekend_rel"]].values)}
    reps["both"] = np.hstack([reps["shape"] / np.sqrt(reps["shape"].shape[1]), reps["weather"] / np.sqrt(3)])
    lab = pd.DataFrame([labels(s) for s in common], index=common, columns=["group", "tech", "ptype"])
    # stability between periods: refit on 2021-22 and 2023-24
    s1, w1 = representations(h, "2021-01-01", "2022-12-31")
    s2, w2 = representations(h, "2023-01-01", "2024-12-31")
    res, chosen = {}, {}
    for name, X in reps.items():
        res[name] = {}
        for k in range(2, 9):
            km = cluster(X, k)
            seeds = [cluster(X, k, s).labels_ for s in range(1, 11)]
            res[name][k] = {"silhouette": float(silhouette_score(X, km.labels_)),
                            "ari_seeds_min": float(min(adjusted_rand_score(km.labels_, l) for l in seeds)),
                            "ari_vs_group": float(adjusted_rand_score(lab["group"], km.labels_)),
                            "ari_vs_tech": float(adjusted_rand_score(lab["tech"], km.labels_)),
                            "min_size": int(np.bincount(km.labels_).min())}
        # rule (set before looking): highest silhouette among k with seed-ARI >= 0.9 and min size >= 5
        ok = [k for k, r in res[name].items() if r["ari_seeds_min"] >= 0.9 and r["min_size"] >= 5]
        chosen[name] = max(ok, key=lambda k: res[name][k]["silhouette"]) if ok else 2
    # period stability for the chosen k
    per = {}
    for name, (A, B) in {"shape": (s1, s2), "weather": (w1[["cool_rel", "heat_rel", "weekend_rel"]], w2[["cool_rel", "heat_rel", "weekend_rel"]])}.items():
        idx = A.dropna().index.intersection(B.dropna().index)
        k = chosen[name]
        la = cluster(StandardScaler().fit_transform(A.loc[idx].values), k).labels_
        lb = cluster(StandardScaler().fit_transform(B.loc[idx].values), k).labels_
        per[name] = float(adjusted_rand_score(la, lb))
    out = lab.copy()
    for name, X in reps.items():
        out[f"cl_{name}"] = cluster(X, chosen[name]).labels_
    out = out.join(weather)
    Path("reports").mkdir(exist_ok=True)
    out.to_csv("reports/A3_clusters.csv")
    # PV / DG growth
    c = pd.read_parquet("data/processed/counts.parquet")
    c["tech"] = c["profile_type"].map(lambda p: next((t for t in ["PV", "DG", "WD"] if p.endswith(t)), "none"))
    grow = c.groupby(["snapshot", "tech"])["esiids"].sum().unstack("tech").fillna(0)
    yearly = grow.groupby(grow.index.year).last()
    # figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for j, season in enumerate(["sum", "win"]):
        cols = [c_ for c_ in shape.columns if c_.startswith(season)]
        for cl, g in out.groupby("cl_shape"):
            ax[j].plot(range(24), shape.loc[g.index, cols].mean().values, label=f"cluster {cl} (n={len(g)})")
        ax[j].set_title({"sum": "Summer weekdays", "win": "Winter weekdays"}[season]); ax[j].set_xlabel("local hour")
    ax[0].set_ylabel("kWh / daily mean"); ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "A3_shape_centroids.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for t in ["PV", "DG", "WD"]:
        if t in yearly:
            ax.plot(yearly.index, yearly[t] / 1000, marker="o", label=t)
    ax.set_ylabel("ESIIDs (thousands)"); ax.set_title("Profile-type ESIIDs with PV / DG / WD flags (year-end snapshot)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "A3_pv_growth.png", dpi=120); plt.close(fig)
    summ = {"chosen_k": chosen, "period_ari": per, "scores": res,
            "contingency_shape": pd.crosstab(out["ptype"], out["cl_shape"]).to_dict(),
            "pv_growth": yearly.to_dict()}
    Path("reports/A3_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps({"chosen_k": chosen, "period_ari": per}, default=float))
    for name in reps:
        k = chosen[name]; print(name, k, {kk: round(v, 3) for kk, v in res[name][k].items()})
    print(pd.crosstab(out["group"] + "_" + out["tech"], out["cl_shape"]))
    print(pd.crosstab(out["group"] + "_" + out["tech"], out["cl_weather"]))
    print(out.groupby("cl_weather")[["cool_rel", "heat_rel", "weekend_rel"]].mean().round(3))
    print(yearly.astype(int).tail(8))


if __name__ == "__main__":
    main()
