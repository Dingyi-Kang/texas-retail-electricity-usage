# Texas retail electricity usage: class-level forecasting and segmentation (Project A)

Status: **A1-A3 complete (holdouts opened once, see [MODEL_CARD.md](MODEL_CARD.md))**; public data. Not affiliated with ERCOT, EIA or Vistra;
no company customer data is used. **No public leaderboard exists for these tasks.**

Frozen protocol: [PROTOCOL.md](PROTOCOL.md) (amendments at the end).

## Modules
| Module | Question | Data |
|---|---|---|
| A1 | Day-ahead hourly kWh per ESIID for 6 mass-market classes x 8 weather zones | ERCOT backcasted load profiles (**model estimates, not meter readings**), weather forecast vintages, ERCOT zone load |
| A2 | Monthly Texas residential / commercial retail sales, 0-2 months ahead (hedging horizon) | EIA-861M (real data), ERA5 degree days |
| A3 | Which customer classes behave alike (shape, weather sensitivity)? PV adoption | ERCOT profiles + profile type counts |

## Layout
```
src/ru/ingest.py, counts.py, eia.py, era5_daily.py   data ingestion
src/ru/features.py, wx.py                            A1 features (ADDTIME availability), zone weather
src/ru/models.py, tsfm.py                            A1 models (Ridge, LightGBM, Chronos-2)
src/ru/backtest.py, metrics.py, compare.py           A1 rolling-origin backtest, ESIID-weighted WAPE, bootstrap
src/ru/a2.py                                         A2 monthly module
src/ru/segment.py                                    A3 segmentation
reports/                                             results; experiments/ registries (all runs logged)
```
Reuses weather caches and ERCOT zone load from the ERCOT load project (`~/ML_forecast`).

## Reproduce
```bash
PY=~/ML_forecast/.venv/bin/python
$PY -m src.ru.ingest && $PY -m src.ru.counts && $PY -m src.ru.eia && $PY -m src.ru.era5_daily
$PY -m pytest -q tests
$PY -m src.ru.backtest B0_naive_d7 W4_lgbm_gfs_zoneload_blend
$PY -m src.ru.a2
$PY -m src.ru.segment
```
