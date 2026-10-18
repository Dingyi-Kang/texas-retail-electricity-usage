"""Module A1 metrics (PROTOCOL.md): ESIID-weighted WAPE (primary), portfolio
error, per-class WAPE, bias, paired 7-day block bootstrap on daily weighted absolute errors."""
import numpy as np
import pandas as pd


def weighted(df):
    """df needs y, pred, w (ESIIDs of the series in the fold's count snapshot)."""
    ae = (df["pred"] - df["y"]).abs() * df["w"]
    return {"wape": 100 * ae.sum() / (df["y"] * df["w"]).sum(),
            "bias_pct": 100 * ((df["pred"] - df["y"]) * df["w"]).sum() / (df["y"] * df["w"]).sum()}


def portfolio(df):
    g = df.assign(Y=df["y"] * df["w"], P=df["pred"] * df["w"]).groupby("hour_utc")[["Y", "P"]].sum()
    return {"portfolio_mape": 100 * ((g["P"] - g["Y"]).abs() / g["Y"]).mean(),
            "portfolio_mae_mwh": (g["P"] - g["Y"]).abs().mean() / 1000}


def summarize(df):
    out = {"n": len(df)}
    out.update(weighted(df)); out.update(portfolio(df))
    for c, g in df.groupby("cls"):
        out[f"wape_{c}"] = weighted(g)["wape"]
    hot = df["extreme"] == 1
    if hot.any():
        out["wape_extreme_days"] = weighted(df[hot])["wape"]
    return out


def daily_err(df):
    d = df.assign(ae=(df["pred"] - df["y"]).abs() * df["w"], yw=df["y"] * df["w"])
    return d.groupby("T")[["ae", "yw"]].sum()


def block_bootstrap_diff(da, db, block=7, n=2000, seed=0):
    """WAPE(a) - WAPE(b) in percentage points with a paired moving-block bootstrap over days."""
    j = da.join(db, lsuffix="_a", rsuffix="_b", how="inner").sort_index()
    A, B, Y = j["ae_a"].values, j["ae_b"].values, j["yw_a"].values
    rng = np.random.default_rng(seed)
    m = len(j); nb = int(np.ceil(m / block)); stats = []
    for _ in range(n):
        st = rng.integers(0, m - block + 1, nb)
        idx = (st[:, None] + np.arange(block)).ravel()[:m]
        stats.append(100 * (A[idx].sum() - B[idx].sum()) / Y[idx].sum())
    d = 100 * (A.sum() - B.sum()) / Y.sum()
    return d, np.percentile(stats, [2.5, 97.5])
