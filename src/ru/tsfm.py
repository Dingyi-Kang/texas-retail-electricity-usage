"""Chronos-2 for Module A1 (zero-shot and LoRA fine-tuned), adapted from the ERCOT
load project's tsfm.py (eval-mode fix and seed averaging included from the start).

For target day T (issue 18:00 on T-1), each of the 48 series is one item:
- context: hourly timestamps up to the end of day T-2 (last `ctx` hours). Target = kWh per ESIID,
  NaN for days after the series' known_max_day (profiles are published with a ~4-day lag).
- past-only covariate: actual ERCOT zone load (known through T-2, load-project A0 assumption).
- known covariates: zone forecast temperature (each local day's own issue-safe vintage; all issued
  before T's issue), local hour, day of week, holiday.
- forecast: days T-1 and T (~48 h); only day T is scored.
Fine-tuning (per fold) uses 2021-04 .. last training day, temperature = GFS forecast vintage
(the blend is used only at prediction time, as in the load project).
"""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from .calendar import holiday_table

TZ = "America/Chicago"
P1_LOAD = str(Path.home() / "ML_forecast/data/processed/load_hourly.parquet")


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class Chronos2A1:
    def __init__(self, ctx=2048, blend=True, finetune=False, steps=1000, lr=1e-5, seeds=(0,), ft_start="2021-04-01",
                 name=None):
        self.ctx, self.blend, self.ft, self.steps, self.lr, self.seeds, self.ft_start = ctx, blend, finetune, steps, lr, seeds, ft_start
        self.name = name or (f"chronos2_{'ft' if finetune else 'zero'}{'_blend' if blend else ''}_s{len(seeds)}")
        self.pipes = None

    # ----------------------------------------------------------------- data
    def _prep(self):
        if hasattr(self, "Y"):
            return
        from .backtest import model_table
        df = model_table()
        self.zone_of = df.drop_duplicates("series").set_index("series")["zone"].to_dict()
        self.Y = df.pivot_table(index="hour_utc", columns="series", values="y", aggfunc="first").sort_index()
        full = pd.date_range(self.Y.index.min(), self.Y.index.max(), freq="h")
        self.Y = self.Y.reindex(full)
        wx = df.drop_duplicates(["hour_utc", "zone"])
        self.TF = wx.pivot(index="hour_utc", columns="zone", values="wf_t").reindex(full)
        self.TB = wx.pivot(index="hour_utc", columns="zone", values="wb_t").reindex(full)
        self.opday = pd.Series(full.tz_convert(TZ).tz_localize(None).normalize(), index=full)
        l = pd.read_parquet(P1_LOAD).set_index("interval_start_utc")
        self.L = l[["COAST", "EAST", "FWEST", "NCENT", "NORTH", "SCENT", "SOUTH", "WEST"]].reindex(full)
        self.Lday = pd.Series(pd.to_datetime(l["op_date"]).values, index=l.index).reindex(full)
        hol = holiday_table("2016-01-01", "2027-01-01"); hol.index = pd.to_datetime(hol.index)
        loc = full.tz_convert(TZ)
        self.cal = pd.DataFrame({"hour": loc.hour, "dow": loc.dayofweek,
                                 "hol": (hol["holiday"].reindex(self.opday.values).values != "none").astype(int)}, index=full)

    def _temp(self, idx, zone, use_blend):
        t = self.TF[zone].reindex(idx).values
        if use_blend:
            b = self.TB[zone].reindex(idx).values
            t = np.where(np.isnan(b), t, b)
        return t

    def _item(self, s, T, kmax):
        z = self.zone_of[s]
        end_ctx = (T - pd.Timedelta(days=1)).tz_localize(TZ).tz_convert("UTC")
        end_tgt = (T + pd.Timedelta(days=1)).tz_localize(TZ).tz_convert("UTC")
        cidx = pd.date_range(end_ctx - pd.Timedelta(hours=self.ctx), end_ctx, freq="h", inclusive="left")
        fidx = pd.date_range(end_ctx, end_tgt, freq="h", inclusive="left")
        y = self.Y[s].reindex(cidx).values.copy()
        y[self.opday.reindex(cidx).values > np.datetime64(kmax)] = np.nan       # unpublished days
        load = self.L[z].reindex(cidx).values.copy()
        load[self.Lday.reindex(cidx).values > np.datetime64(T - pd.Timedelta(days=2))] = np.nan
        key = f"{s}|{T.date()}"
        past = pd.DataFrame({"item_id": key, "timestamp": cidx.tz_localize(None), "target": y, "load": load,
                             "temp": self._temp(cidx, z, self.blend)})
        fut = pd.DataFrame({"item_id": key, "timestamp": fidx.tz_localize(None), "temp": self._temp(fidx, z, self.blend)})
        for c in ["hour", "dow", "hol"]:
            past[c] = self.cal[c].reindex(cidx).values
            fut[c] = self.cal[c].reindex(fidx).values
        past["temp"] = past["temp"].interpolate(limit_direction="both")
        fut["temp"] = fut["temp"].interpolate(limit_direction="both")
        return past, fut, fidx, key

    # ----------------------------------------------------------------- fit
    def fit(self, tr):
        from chronos import Chronos2Pipeline
        self._prep()
        if not self.ft:
            return self
        import torch
        from chronos.chronos2.preprocess import from_data_frame
        last = pd.Timestamp(tr["T"].max())
        end = (last + pd.Timedelta(days=1)).tz_localize(TZ).tz_convert("UTC")
        start = pd.Timestamp(self.ft_start).tz_localize(TZ).tz_convert("UTC")
        idx = pd.date_range(start, end, freq="h", inclusive="left")
        parts = []
        for s in sorted(tr["series"].unique()):
            z = self.zone_of[s]
            parts.append(pd.DataFrame({"item_id": s, "timestamp": idx.tz_localize(None), "target": self.Y[s].reindex(idx).values,
                                       "load": self.L[z].reindex(idx).values,
                                       "temp": pd.Series(self._temp(idx, z, False)).interpolate(limit_direction="both").values,
                                       "hour": self.cal["hour"].reindex(idx).values, "dow": self.cal["dow"].reindex(idx).values,
                                       "hol": self.cal["hol"].reindex(idx).values}))
        data = pd.concat(parts, ignore_index=True)
        inputs = from_data_frame(data, target_columns=["target"], prediction_length=48,
                                 known_covariates_names=["temp", "hour", "dow", "hol"])
        self.pipes = []
        for seed in self.seeds:
            out = Path(f"{tempfile.gettempdir()}/chronos_ft_A1/{self.name}_seed{seed}_{last.date()}")
            ck = out / "finetuned-ckpt"
            if (ck / "config.json").exists():
                p = Chronos2Pipeline.from_pretrained(str(ck), device_map=_device())
            else:
                torch.manual_seed(seed)
                base = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=_device())
                p = base.fit(inputs, prediction_length=48, finetune_mode="lora", learning_rate=self.lr, num_steps=self.steps,
                             batch_size=32, context_length=self.ctx, output_dir=str(out), remove_printer_callback=True,
                             save_strategy="no", seed=seed)
            p.model.eval()                    # Chronos2Pipeline.fit leaves train mode on (dropout)
            self.pipes.append(p)
        return self

    # ----------------------------------------------------------------- predict
    def predict(self, df):
        from chronos import Chronos2Pipeline
        self._prep()
        if self.pipes is None:
            self.pipes = [Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=_device())]
        keys = df.drop_duplicates(["series", "T"])[["series", "T", "known_max_day"]]
        groups = {}
        for s, T, k in keys.itertuples(index=False):
            p, f, fidx, key = self._item(s, pd.Timestamp(T), pd.Timestamp(k))
            groups.setdefault(len(fidx), []).append((p, f, fidx, key, pd.Timestamp(T)))
        preds = {}
        for H, lst in groups.items():
            for i in range(0, len(lst), 256):
                ch = lst[i:i + 256]
                past = pd.concat([c[0] for c in ch], ignore_index=True)
                fut = pd.concat([c[1] for c in ch], ignore_index=True)
                outs = []
                for pipe in self.pipes:
                    r = pipe.predict_df(past, future_df=fut, prediction_length=H, quantile_levels=[0.5], context_length=self.ctx,
                                        id_column="item_id", timestamp_column="timestamp", target="target")
                    col = "predictions" if "predictions" in r.columns else "0.5"
                    outs.append(r.set_index(["item_id", "timestamp"])[col])
                avg = sum(outs) / len(outs)
                for (_, _, fidx, key, T) in ch:
                    v = avg.loc[key].reindex(fidx.tz_localize(None)).values
                    preds[key] = pd.Series(v, index=fidx)
        out = np.empty(len(df))
        for (s, T), g in df.groupby(["series", "T"]):
            sr = preds[f"{s}|{pd.Timestamp(T).date()}"]
            out[df.index.get_indexer(g.index)] = sr.reindex(pd.DatetimeIndex(g["hour_utc"])).values
        return out
