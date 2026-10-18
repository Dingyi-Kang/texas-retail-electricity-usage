"""Module A1 rolling-origin backtest + append-only registry (PROTOCOL.md).
Usage: python -m src.ru.backtest EXP [EXP ...]          # validation 2024-2025
       python -m src.ru.backtest --holdout EXP [...]    # needs configs/final_A1.yaml
"""
import argparse, datetime as dt, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from .features import load_features
from .wx import load_wx
from .metrics import summarize

VAL_FOLDS = [str(d.date()) for d in pd.date_range("2024-01-01", "2025-10-01", freq="QS")]
VAL_END = "2025-12-31"
HOLDOUT_FOLDS, HOLDOUT_END = ["2026-01-01", "2026-04-01", "2026-07-01"], "2026-09-30"
REG = Path("experiments/registry.jsonl")
DATA_CACHE = Path("data/processed/A1_model_table.parquet")


def model_table(rebuild=False):
    if DATA_CACHE.exists() and not rebuild:
        return pd.read_parquet(DATA_CACHE)
    f = load_features()
    W = load_wx(f)
    df = f.merge(W, on=["hour_utc", "T", "zone"], how="left", validate="many_to_one")
    d = pd.read_parquet("data/processed/profile_days.parquet")
    d["addtime"] = pd.to_datetime(d["addtime"]).dt.normalize()
    d = d.drop_duplicates(["series", "op_date"], keep="last").rename(columns={"op_date": "T", "addtime": "y_addtime"})
    df = df.merge(d[["series", "T", "y_addtime"]], on=["series", "T"], how="left")
    ext = df.groupby(["zone", "T"])["wo_t"].agg(["max", "min"])
    flag = ((ext["max"] >= 38) | (ext["min"] <= -5)).astype(int).rename("extreme")
    df = df.merge(flag.reset_index(), on=["zone", "T"], how="left")
    df = df.merge(zone_load_features(), on=["zone", "T"], how="left")
    df.to_parquet(DATA_CACHE, index=False)
    return df


def zone_load_features():
    """ERCOT actual zone load (from the load project) known through day T-2 (its A0 assumption):
    zl2_rel = zone mean load of T-2 / mean of T-8..T-2; zl3_rel likewise for T-3; zl2_wk = T-2 / T-9."""
    l = pd.read_parquet(Path.home() / "ML_forecast/data/processed/load_hourly.parquet")
    l["op_date"] = pd.to_datetime(l["op_date"])
    zones = ["COAST", "EAST", "FWEST", "NCENT", "NORTH", "SCENT", "SOUTH", "WEST"]
    dm = l.groupby("op_date")[zones].mean()
    lvl = dm.rolling(7).mean()                       # mean of the 7 days ending at the index day
    rows = []
    days = pd.date_range("2017-02-01", "2026-09-30", freq="D")
    for z in zones:
        f = pd.DataFrame({"T": days, "zone": z})
        f["zl2_rel"] = (dm[z].reindex(days - pd.Timedelta(days=2)).values / lvl[z].reindex(days - pd.Timedelta(days=2)).values)
        f["zl3_rel"] = (dm[z].reindex(days - pd.Timedelta(days=3)).values / lvl[z].reindex(days - pd.Timedelta(days=3)).values)
        f["zl2_wk"] = (dm[z].reindex(days - pd.Timedelta(days=2)).values / dm[z].reindex(days - pd.Timedelta(days=9)).values)
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def fold_weights(S):
    c = pd.read_parquet("data/processed/counts.parquet")
    snap = c[c["snapshot"] < S]["snapshot"].max()
    w = c[c["snapshot"] == snap].groupby(["profile_type", "weather_zone"])["esiids"].sum()
    return {f"{p}_{z}": v for (p, z), v in w.items()}, snap


def make_folds(starts, end):
    return [(pd.Timestamp(s), pd.Timestamp(starts[i + 1]) - pd.Timedelta(days=1) if i + 1 < len(starts) else pd.Timestamp(end))
            for i, s in enumerate(starts)]


def run_backtest(df, factory, folds, train_start):
    preds = []
    for S, E in folds:
        first_issue_date = S - pd.Timedelta(days=1)
        tr = df[(df["T"] >= pd.Timestamp(train_start)) & (df["y_addtime"] < first_issue_date)]
        te = df[(df["T"] >= S) & (df["T"] <= E)].copy()
        # cutoff checks: every training target was published before the first issue of the fold,
        # and every test feature day was known at that row's issue
        assert (tr["y_addtime"] < first_issue_date).all() and tr["T"].max() < S
        assert (te["known_max_day"] < te["issue_date"]).all()
        m = factory()
        m.fit(tr)
        te["pred"] = m.predict(te)
        w, snap = fold_weights(S)
        te["w"] = te["series"].map(w).fillna(0)
        te["fold"] = str(S.date()); te["count_snapshot"] = str(snap.date())
        preds.append(te)
    return pd.concat(preds, ignore_index=True)


def git_commit():
    try:
        dirty = subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True).stdout.strip()
        c = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        return c + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main(argv=None):
    from .experiments import EXPERIMENTS
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+"); ap.add_argument("--holdout", action="store_true"); ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args(argv)
    if a.holdout:
        import yaml
        cfg = Path("configs/final_A1.yaml")
        if not cfg.exists():
            sys.exit("Holdout locked: create configs/final_A1.yaml first.")
        allowed = yaml.safe_load(cfg.read_text())["holdout_set"]
        done = {json.loads(l)["experiment"] for l in REG.open() if json.loads(l)["phase"] == "holdout"} if REG.exists() else set()
        bad = [e for e in a.exps if e not in allowed or e in done]
        if bad:
            sys.exit(f"Not allowed on holdout: {bad}")
    df = model_table(a.rebuild)
    folds = make_folds(HOLDOUT_FOLDS, HOLDOUT_END) if a.holdout else make_folds(VAL_FOLDS, VAL_END)
    REG.parent.mkdir(exist_ok=True)
    for name in a.exps:
        spec = EXPERIMENTS[name]
        t0 = time.time(); status, metrics = "ok", None
        run_id = f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}{'_HOLDOUT' if a.holdout else ''}"
        rdir = Path("experiments/runs") / run_id; rdir.mkdir(parents=True, exist_ok=True)
        try:
            out = run_backtest(df, spec["model"], folds, spec.get("train_start", "2017-02-01"))
            out[["series", "cls", "zone", "T", "hour_utc", "horizon_h", "fold", "y", "pred", "w", "extreme"]].to_parquet(rdir / "predictions.parquet", index=False)
            metrics = summarize(out)
        except Exception as ex:
            status = f"error: {ex!r}"
            import traceback; traceback.print_exc()
        rec = {"run_id": run_id, "experiment": name, "phase": "holdout" if a.holdout else "validation",
               "hypothesis": spec["hypothesis"], "config": spec.get("config", {}), "status": status,
               "git": git_commit(), "seconds": round(time.time() - t0, 1), "metrics": metrics}
        (rdir / "record.json").write_text(json.dumps(rec, indent=1, default=float))
        with REG.open("a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        m = metrics or {}
        print(f"{name:34s} {status:6s} WAPE={m.get('wape', np.nan):6.2f}% bias={m.get('bias_pct', np.nan):6.2f}% "
              f"portMAPE={m.get('portfolio_mape', np.nan):5.2f}% extreme={m.get('wape_extreme_days', np.nan):6.2f}% ({rec['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
