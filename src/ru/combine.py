"""Equal-weight combinations of saved run predictions (experiments.COMBOS), logged
to the same registry. Usage: python -m src.ru.combine NAME [--holdout]"""
import datetime as dt, json, sys
from pathlib import Path
import pandas as pd
from .experiments import COMBOS
from .compare import load
from .metrics import summarize
from .backtest import REG, git_commit


def main():
    name = sys.argv[1]; h = "--holdout" in sys.argv
    spec = COMBOS[name]
    if h:
        import yaml
        cfg = Path("configs/final_A1.yaml")
        if not cfg.exists() or name not in yaml.safe_load(cfg.read_text())["holdout_set"]:
            sys.exit(f"Holdout locked or {name} not in holdout_set.")
        if any(json.loads(l)["experiment"] == name and json.loads(l)["phase"] == "holdout" for l in REG.open()):
            sys.exit(f"{name} already run on holdout.")
    P = [load(m, h) for m in spec["members"]]
    key = ["series", "hour_utc"]
    out = P[0].copy()
    for i, p in enumerate(P[1:], 1):
        assert (p[key].values == out[key].values).all()
    out["pred"] = sum(p["pred"].values for p in P) / len(P)
    run_id = f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}{'_HOLDOUT' if h else ''}"
    rdir = Path("experiments/runs") / run_id; rdir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(rdir / "predictions.parquet", index=False)
    m = summarize(out)
    rec = {"run_id": run_id, "experiment": name, "phase": "holdout" if h else "validation", "hypothesis": spec["hypothesis"],
           "config": {"members": spec["members"], "weights": "equal"}, "status": "ok", "git": git_commit(), "metrics": m}
    with REG.open("a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    print(f"{name}: WAPE={m['wape']:.2f}% bias={m['bias_pct']:.2f}% portMAPE={m['portfolio_mape']:.2f}% extreme={m.get('wape_extreme_days', float('nan')):.2f}%")


if __name__ == "__main__":
    main()
