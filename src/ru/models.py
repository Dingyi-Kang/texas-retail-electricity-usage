"""Module A1 models. Relative target: y / level7 (level from known days)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

LAG_COLS = ["lag6", "lag7", "lag8", "lag13", "lag14", "lag21", "lag28", "lag364"]


def rel(df, cols):
    return df[cols].div(df["level7"], axis=0)


class Naive7:
    name = "naive_d7"
    def fit(self, tr): return self
    def predict(self, df): return df["lag7"].fillna(df["lag14"]).fillna(df["level7"]).values


class RidgeSeriesHour:
    """One Ridge per (series, local hour) on relative lags + calendar (+ optional weather hinges)."""
    def __init__(self, weather=None, alpha=1.0, knots=(0, 5, 10, 15, 20, 25, 28, 31, 34), name="ridge_sh"):
        self.weather, self.alpha, self.knots, self.name = weather, alpha, knots, name

    def _X(self, df):
        X = rel(df, LAG_COLS).fillna(1.0)
        for d in range(7):
            X[f"dow{d}"] = (df["dow"] == d).astype(float)
        X["holiday"] = df["holiday"]; X["xmas"] = df["xmas_week"]
        if self.weather:
            for col in self.weather:
                t = df[col]
                for k in self.knots:
                    X[f"{col}_h{k}"] = np.maximum(t - k, 0) / 10
                X[f"{col}_lin"] = t / 10
        return X.fillna(0.0)

    def fit(self, tr):
        tr = tr.dropna(subset=["y", "level7"])
        self.m = {}
        for key, g in tr.groupby(["series", "hour"]):
            self.m[key] = Ridge(alpha=self.alpha).fit(self._X(g), g["y"] / g["level7"])
        return self

    def predict(self, df):
        out = pd.Series(np.nan, index=df.index)
        for key, g in df.groupby(["series", "hour"]):
            if key in self.m:
                out.loc[g.index] = self.m[key].predict(self._X(g)) * g["level7"].values
        return out.values


class LGBM:
    """Global LightGBM over all series and hours, relative target, L1 objective."""
    def __init__(self, features=(), n=1500, lr=0.03, leaves=63, seed=0, name="lgbm"):
        self.features, self.n, self.lr, self.leaves, self.seed, self.name = list(features), n, lr, leaves, seed, name

    def _X(self, df):
        X = rel(df, LAG_COLS)
        for c in ["hour", "dow", "month", "doy", "holiday", "xmas_week", "cls_id", "zone_id", "horizon_h"]:
            X[c] = df[c]
        for c in self.features:
            X[c] = df[c]
        return X

    def fit(self, tr):
        import lightgbm as lgb
        tr = tr.dropna(subset=["y", "level7"])
        self.m = lgb.LGBMRegressor(objective="l1", n_estimators=self.n, learning_rate=self.lr, num_leaves=self.leaves,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=self.seed,
                                   verbose=-1, n_jobs=16)
        self.m.fit(self._X(tr), tr["y"] / tr["level7"])
        return self

    def predict(self, df):
        return self.m.predict(self._X(df)) * df["level7"].values


class WithFallback:
    """Weather model where its required weather exists, otherwise a weather-free fallback (e.g. the
    Jan-2024 GFS archive gap)."""
    def __init__(self, main, fallback, required, name=None):
        self.main, self.fb, self.req = main, fallback, list(required)
        self.name = name or f"{main.name}+fb"

    def fit(self, tr):
        ok = tr[self.req].notna().all(axis=1)
        self.main.fit(tr[ok]); self.fb.fit(tr)
        return self

    def predict(self, df):
        out = self.fb.predict(df)
        ok = df[self.req].notna().all(axis=1).values
        if ok.any():
            out[ok] = self.main.predict(df[ok])
        return out


class Ensemble:
    def __init__(self, members, name="ens"):
        self.members, self.name = members, name
    def fit(self, tr):
        for m in self.members: m.fit(tr)
        return self
    def predict(self, df):
        return np.mean([m.predict(df) for m in self.members], axis=0)


class BlendView:
    """Train on GFS columns (wf_*), predict with the GFS+ECMWF blend (wb_*) substituted where it
    exists (as in the ERCOT load project)."""
    def __init__(self, model, name=None):
        self.model, self.name = model, name or f"{model.name}_blend"
    def fit(self, tr):
        self.model.fit(tr); return self
    def predict(self, df):
        d = df.copy()
        for c in [c for c in df.columns if c.startswith("wb_")]:
            f = "wf_" + c[3:]
            if f in d.columns:
                d[f] = np.where(d[c].notna(), d[c], d[f])
        return self.model.predict(d)
