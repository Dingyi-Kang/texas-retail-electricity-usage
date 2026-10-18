"""Module A1 experiments; each hypothesis is written before the run."""
from .models import Naive7, RidgeSeriesHour, LGBM, WithFallback, Ensemble

W_START = "2021-04-01"
EXPERIMENTS = {
    "B0_naive_d7": dict(
        hypothesis="Same hour one week back (T-7, known) is the floor every model must beat.",
        model=lambda: Naive7()),
    "B1_ridge_series_hour": dict(
        hypothesis="Per series x hour Ridge on relative known lags + calendar beats B0 (no weather).",
        model=lambda: RidgeSeriesHour()),
    "B2_lgbm_noweather": dict(
        hypothesis="Global LightGBM on lags + calendar + class/zone ids is at least as good as B1 (no weather).",
        model=lambda: LGBM()),
}

from .models import BlendView
WF = ["wf_t", "wf_tlag1", "wf_tlag2", "wf_tlag3", "wf_tlag6", "wf_dmax", "wf_dmin", "wf_dmean", "wo2_dmean"]
WO = ["wo_t", "wo_dmax", "wo_dmin", "wo_dmean", "wo2_dmean"]
_FB = lambda: LGBM(name="lgbm_fb")
EXPERIMENTS.update({
    "W1_lgbm_gfs": dict(
        hypothesis="Adding issue-safe GFS zone temperature (+1-6 h lags, day max/min/mean) and observed T-2 temperature to B2 cuts weighted WAPE substantially (profiles are weather-driven); weather-free fallback in the GFS gap.",
        train_start=W_START, model=lambda: WithFallback(LGBM(features=WF, name="lgbm_gfs"), _FB(), ["wf_t"]),
        config={"weather": "GFS", "features": WF}),
    "W2_ridge_hinge_gfs": dict(
        hypothesis="Per series x hour Ridge with hinge terms of GFS hourly and daily-max zone temperature beats B1 and is competitive with W1 (GEFCom-vanilla-style regression).",
        train_start=W_START, model=lambda: WithFallback(RidgeSeriesHour(weather=["wf_t", "wf_dmax", "wf_tlag3"], name="ridge_hinge"), _FB(), ["wf_t", "wf_dmax", "wf_tlag3"]),
        config={"weather": "GFS", "hinge_cols": ["wf_t", "wf_dmax", "wf_tlag3"]}),
    "WX_lgbm_era5_backcast": dict(
        hypothesis="BACKCAST reference (not a forecast): with actual ERA5 weather at target times, LightGBM reproduces ERCOT's profile models far better than any forecast, showing the target is weather-model output.",
        train_start=W_START, model=lambda: LGBM(features=WO, name="lgbm_backcast"), config={"weather": "ERA5 actual (backcast)"}),
})

ZL = ["zl2_rel", "zl3_rel", "zl2_wk"]
EXPERIMENTS.update({
    "W3_lgbm_gfs_zoneload": dict(
        hypothesis="Daily profile residuals persist (lag-1 autocorr 0.68) but profiles are only known to T-6; actual ERCOT zone load known to T-2 (load-project A0 assumption) carries that recent anomaly, so adding zone-load anomaly features to W1 lowers weighted WAPE.",
        train_start=W_START, model=lambda: WithFallback(LGBM(features=WF + ZL, name="lgbm_gfs_zl"), LGBM(features=ZL, name="lgbm_fb_zl"), ["wf_t"]),
        config={"weather": "GFS", "features": WF + ZL}),
})
EXPERIMENTS.update({
    "W4_lgbm_gfs_zoneload_blend": dict(
        hypothesis="W3 trained on GFS but predicting with the GFS+ECMWF blend where ECMWF exists (2024-02+) beats W3, as blending did in the load project.",
        train_start=W_START, model=lambda: BlendView(WithFallback(LGBM(features=WF + ZL, name="lgbm_gfs_zl"), LGBM(features=ZL, name="lgbm_fb_zl"), ["wf_t"])),
        config={"weather": "train GFS, predict GFS+ECMWF blend", "features": WF + ZL}),
})

from .tsfm import Chronos2A1
EXPERIMENTS.update({
    "T1_chronos2_zero_blend": dict(
        hypothesis="Zero-shot Chronos-2 per series (context to T-2 with unpublished days masked, zone load as past covariate, forecast temperature + calendar as known covariates) is competitive with W4 (7.83%).",
        model=lambda: Chronos2A1(finetune=False), config={"ctx": 2048, "covariates": ["load (past)", "temp blend", "hour", "dow", "hol"]}),
    "T2_chronos2_lora_3seeds_blend": dict(
        hypothesis="LoRA fine-tuning on the 48 series per fold (1000 steps, lr 1e-5, eval mode, 3-seed average; settings from the load project) beats T1 and W4.",
        model=lambda: Chronos2A1(finetune=True, seeds=(0, 1, 2)), config={"finetune": "lora", "steps": 1000, "lr": 1e-5, "seeds": 3}),
})


class SavedAverage:
    """Equal-weight average of the saved out-of-sample predictions of registered runs (no weights are
    fitted, so it is equivalent to running the members side by side). Used via backtest --combine."""
    def __init__(self, members):
        self.members = members

COMBOS = {
    "E1_avg_W4_T1": dict(
        hypothesis="Equal average of W4 (LightGBM, bias ~0) and T1 (zero-shot Chronos-2, bias -1.3%) beats both: different model families, errors partly offset.",
        members=["W4_lgbm_gfs_zoneload_blend", "T1_chronos2_zero_blend"]),
    # registered 2026-10-14 after T2 (6.20%) was seen, before any combo with T2 was computed
    "E2_avg_W4_T2": dict(
        hypothesis="Equal average of W4 (7.83%) and T2 (6.20%) beats T2; uncertain because W4 is much worse and T2 has ~0 bias, so less error offset than in E1.",
        members=["W4_lgbm_gfs_zoneload_blend", "T2_chronos2_lora_3seeds_blend"]),
    "E3_avg_W4_T1_T2": dict(
        hypothesis="Equal average of W4, T1 and T2 beats T2 (expected not to: two of three members are much worse).",
        members=["W4_lgbm_gfs_zoneload_blend", "T1_chronos2_zero_blend", "T2_chronos2_lora_3seeds_blend"]),
}
