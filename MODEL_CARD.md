# Model card — Project A: Texas retail electricity usage

Written 2026-10-14. Public data only; not affiliated with ERCOT, EIA or Vistra. **No public
leaderboard exists for these tasks**; all numbers are internal-protocol comparisons
([PROTOCOL.md](PROTOCOL.md), locked choices in `configs/final_A1.yaml`, `configs/final_A2.yaml`).

## A1 — day-ahead hourly kWh per ESIID, 6 mass-market classes x 8 weather zones

**Selected model: T2** — Chronos-2 (amazon/chronos-2) LoRA fine-tuned per quarterly fold on the 48
series (1000 steps, lr 1e-5, eval mode, average of 3 seeds); context up to T-2 with unpublished
profile days masked, ERCOT zone load as past covariate, forecast temperature (GFS/ECMWF blend
vintages) + calendar as known covariates.

Target caveat: ERCOT backcasted profiles are **ERCOT model estimates driven by actual weather, not
meter readings**. A1 is not a forecast of real customers' metered usage.

ESIID-weighted WAPE (lower is better). Validation = 8 quarterly folds 2024-2025; holdout =
2026-01..09 (no 2026-05, archive gap), opened once after the choice was locked.

| Run | Validation | Holdout | Holdout portfolio MAPE |
|---|---|---|---|
| B0 same weekday last week | 15.81% | 12.98% | 14.31% |
| B1 per-series Ridge (no weather) | 13.45% | — | — |
| B2 LightGBM (no weather) | 12.77% | — | — |
| W1 LightGBM + GFS temperature | 8.79% | — | — |
| W4 LightGBM + GFS/ECMWF blend + zone load to T-2 | 7.83% | 6.50% | 7.10% |
| T1 Chronos-2 zero-shot | 8.25% | 6.53% | 5.02% |
| E1 average W4 + T1 | 7.16% | 5.65% | 6.66% |
| **T2 Chronos-2 LoRA, 3 seeds** | **6.20%** | **4.78%** | **3.12%** |
| E2 average W4 + T2 | 6.27% | 4.84% | 5.77% |
| WX LightGBM with *actual* weather (backcast reference, not a forecast) | 7.58% | not run | — |

Paired moving-block bootstrap (7-day blocks), T2 minus other:

| vs | Validation | Holdout |
|---|---|---|
| W4 | -1.63 pp [-1.97, -1.29], 8/8 folds | -1.72 pp [-2.61, -0.94], 3/3 |
| E1 | -0.96 pp [-1.20, -0.73], 8/8 | -0.87 pp [-1.51, -0.28], 3/3 |
| T1 | -2.05 pp [-2.45, -1.71], 8/8 | -1.75 pp [-2.43, -1.24], 3/3 |
| E2 | -0.08 pp [-0.25, +0.10], 4/8 | -0.06 pp [-0.53, +0.45], 1/3 |

T2 holdout per-class WAPE: RESHIWR 6.4%, RESLOWR 5.7%, BUSLOLF 4.1%, BUSNODEM 3.2%, BUSMEDLF 2.5%,
BUSHILF 1.5%; extreme-weather days 4.6%; bias +0.5%.

Findings
- Fine-tuning a time-series foundation model on all 48 series together beat a tuned global
  LightGBM by ~1.7 pp on both validation and holdout; zero-shot Chronos-2 alone did not.
- Averaging helped when members had opposite biases (E1) but not once the strong model had ~0 bias
  (E2, E3: E3 = W4+T1+T2 was worse than T2, +0.32 pp [+0.18, +0.46]).
- All models score lower on the 2026 holdout than on 2024-2025 validation (B0 also drops
  15.8% -> 13.0%): the holdout period is easier, so holdout levels are not comparable to validation
  levels; only the ranking is.
- The backcast with actual weather (WX, 7.58%) was not near-perfect, so ERCOT's profile models use
  more than our zone temperature; the zone-load features (known through T-2) captured part of that.

Limits: archived profiles may be revisions; residential classes dominate error; the 4-day
publication lag means the model never sees the last 5 days of the target series itself.

## A2 — monthly Texas residential + commercial retail sales (EIA-861M, real data)

Issue on the 1st of month M, horizons M, M+1, M+2; month m known from the 1st of m+3; weather =
10-year climatology of degree days (no archived monthly forecasts). MAPE averaged over sectors and
horizons. Validation issues 2021-01..2023-12; holdout issues 2024-01..2026-05 (opened once).

| Run | Validation | Holdout | Holdout residential h0/h1/h2 | Holdout commercial h0/h1/h2 |
|---|---|---|---|---|
| M0 same month last year | 7.88% | 5.42% | 6.1 / 5.8 / 6.0 | 4.8 / 4.9 / 4.9 |
| M1 M0 x customer growth | 7.81% | 5.78% | 6.0 / 5.8 / 5.9 | 5.4 / 5.5 / 5.9 |
| M3 degree-day regression per customer, 5-y window | 5.75% | **4.10%** | 4.5 / 4.2 / 4.2 | 3.7 / 3.9 / 4.0 |
| M4 same, 3-y window | 5.70% | 4.48% | 4.8 / 4.5 / 4.4 | 4.1 / 4.6 / 4.6 |
| **M6 (selected): residential 5 y, commercial 3 y** | **5.48%** | 4.37% | 4.5 / 4.2 / 4.2 | 4.1 / 4.6 / 4.6 |
| MX full-history regression with *actual* degree days (backcast reference) | 4.65% | 6.40% | 4.2 / 4.0 / 4.0 | 8.4 / 8.6 / 9.0 |

Findings
- **The selected model was not the best on holdout.** M6's commercial window was picked after
  seeing M3/M4 on validation (flagged as "mildly optimistic" when locked); on holdout the 5-year
  window (M3) was better for commercial too, so M6 lost 0.27 pp to the simpler M3. Reported as is;
  the holdout is not used to change the choice.
- Degree-day regression beats last-year baselines by ~1.3 pp on holdout; residential error is
  mostly weather (actual degree days cut it to ~4%), commercial error is mostly trend: the
  full-history model with actual weather (MX) is badly biased for commercial (8-9%), because
  commercial kWh per customer rose ~19% from 2021 to 2025.
- Limitation: the EIA file holds revised values, not the values available at each issue.

## A3 — segmentation (descriptive)
See [reports/A3_segmentation.md](reports/A3_segmentation.md): weather-sensitivity clusters
(k=4, stable across seeds and periods) reproduce ERCOT's RESHI / RESLO / BUSLO / other grouping;
PV-flagged ESIIDs grew from ~35k (2019) to ~218k (2026).
