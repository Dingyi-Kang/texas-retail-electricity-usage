# PROTOCOL — Project A: Texas retail electricity usage (class-level forecasting + segmentation)

Frozen 2026-10-14 before any model was fit. Changes only as dated
amendments at the end. Personal study project on public data; not affiliated with ERCOT, EIA or Vistra.

## Data
- **ERCOT Backcasted (Actual) Load Profiles** 2017-01..2026-09 (`src/ru/ingest.py`): average kWh per
  ESIID per 15 min for 248 profile-type x weather-zone series. **These are ERCOT model estimates
  driven by actual weather, not meter readings**; since AMS rollout they settle only ~1.6% of ESIIDs
  (non-IDR) and fill missing interval data. Archive gaps: all of 2026-05 and 2026-08-31.
- **ERCOT Profile Type Counts** (142 snapshots 2013-04..2026-10): ESIIDs by zone x meter type x TDSP
  x profile type (`src/ru/counts.py`).
- **EIA-861M** monthly retail sales/customers/revenue by state and sector, 2010..2026-07 (real data;
  Texas includes non-ERCOT areas; recent months are preliminary and later revised).
- Weather: Open-Meteo GFS / ECMWF forecast vintages and ERA5, reused from the ERCOT load project
  (`~/ML_forecast`, same vintage rule; see its reports/weather_semantics.md). Zone temperature =
  mean of that zone's cities.

## Module A1 — day-ahead class-level hourly usage (ERCOT profiles)
- **Series:** 6 mass-market classes x 8 weather zones = 48 series: RESLOWR, RESHIWR (residential),
  BUSNODEM, BUSLOLF, BUSMEDLF, BUSHILF (small/medium business). Together ~95% of ESIIDs.
  BUSIDRRQ (large IDR-required business), PV/DG/WD variants, lighting and flat profiles excluded.
- **Target:** kWh per ESIID for every real UTC hour of local day T (23/24/25 hours).
- **Issue:** 18:00 America/Chicago on T-1. Horizon 7-30 h.
- **Availability rule (measured, conservative):** a profile day d is known at the issue on day D
  only if its archive ADDTIME date < D. Measured ADDTIME lag is 4 days for 2018-2026, so normally
  d <= D-5 (= T-6). ERCOT states the backcast for the previous day is available, but archived values
  may be later revisions, so the conservative rule is used.
- **Weather:** forecast vintages per the ERCOT project rule (run + publication delay <= issue);
  observed (ERA5) weather for days <= T-2 allowed; weather at target times from ERA5 = backcast only.
- **Training data:** weather-forecast models from 2021-04-01 (GFS temperature archive start);
  weather-free baselines from 2017-02-01.
- **Validation:** 8 quarterly folds, target days 2024-01-01..2025-12-31; refit at each fold start S
  on days known at the first issue of the fold (<= S-6 by the rule above); online state updates
  within a fold only with data known at each issue.
- **Holdout:** 2026-01-01..2026-09-30 (excluding archive gaps), folds starting 2026-01-01,
  04-01, 07-01; run once for a pre-declared set after selection. (The weather and ERCOT system load
  of 2026 were seen in the earlier project; these profile targets were not.)
- **Metrics:** primary = **ESIID-weighted WAPE** = sum_s w_s sum_h |e_sh| / sum_s w_s sum_h y_sh, with
  w_s = ESIID count of series s in the latest count snapshot before the fold start (i.e. absolute
  error in MWh of the whole mass market without cancellation across series). Secondary: portfolio
  error of the summed MWh (errors may cancel), per-class WAPE, bias, daily-peak error, extreme-weather
  days (zone daily max temp >= 38 C or min <= -5 C).
- **Statistics:** paired moving-block bootstrap (7-day blocks) on daily weighted absolute errors.
- **Baselines first:** B0 same weekday one week back (T-7, known), B1 per-series Ridge on calendar +
  known lags, B2 global LightGBM without weather. Weather models only after baselines are logged.
- **Budget:** 40 validation runs, all logged in experiments/registry.jsonl including failures.
- **Selection rule:** lowest validation weighted WAPE among forecast (non-backcast) runs; if the
  gain over the next simpler candidate has a 95% CI including 0, the simpler model is chosen.

## Module A2 — monthly retail sales (EIA-861M, real data)
- **Targets:** Texas residential and commercial retail sales (MWh) for calendar months.
- **Issue:** first day of month M. Horizons: months M, M+1, M+2 (M+1 = next full month, primary).
- **Availability:** month m known from the first day of month m+3 (EIA posts ~8 weeks after month
  end; conservative). The current file contains revised values, not the vintages seen at the time
  (limitation, stated in results).
- **Weather:** no archived monthly weather forecasts exist in our data, so forecasts use
  climatology (same-month mean of the previous 10 years of ERA5 degree days, computed from data
  known at the issue). A weather-normalized backcast with actual ERA5 degree days is reported
  separately and never as a forecast.
- **Validation:** issues 2021-01..2023-12. **Holdout:** issues 2024-01..2026-05 (targets to 2026-07),
  opened once. Metrics: MAPE and MAE (GWh) per sector and horizon. Baselines: same month last
  known year (scaled by customer growth), seasonal naive.

## Module A3 — segmentation (descriptive)
- Cluster the 248 profile series (and the 48 mass-market series) by normalized daily shape and
  weather sensitivity, fit on 2021-2024 data; compare with ERCOT's class labels; stability by ARI
  across seeds and years. Track PV/DG class growth from the count snapshots. Descriptive only: no
  cluster labels from future data are used as forecasting inputs.

## Limits and honesty rules
- No public leaderboard exists for these tasks; results are internal-protocol comparisons.
- A1 targets are modeled data; a near-perfect fit with actual weather is expected and is not a
  forecasting result. A1 must not be described as forecasting real customers' metered usage.
- Neither module uses Vistra or any company's customer data.

## Amendments

### P-A1 (2026-10-14 ~09:45, after B0-B2, W1-W4, WX of A1 and M0-M3, MX of A2 were run)
- A2 validation budget: 15 runs (5 used). Selection rule as in A1: lowest validation MAPE among
  forecast (non-backcast) runs averaged over both sectors and horizons; simpler model if the
  paired difference is not clearly better (A2 has only 36 issues, so the comparison is reported
  per sector and horizon, without a formal CI).
- Observed: A1 backcast with actual weather (WX) is 7.58% WAPE, not near-perfect; the PROTOCOL's
  expectation of a near-perfect backcast is falsified (ERCOT's profile models use more than our
  zone temperature). A1 W3/W4 use ERCOT actual zone load known through T-2 (load project A0
  assumption) as features.
- A2 weather weights: approximate metro populations (ERCOT count snapshots under-cover municipal
  utilities such as San Antonio and Austin).
