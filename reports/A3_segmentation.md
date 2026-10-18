# A3 — Segmentation of ERCOT load-profile series (descriptive)

Written 2026-10-14. Data: 248 ERCOT backcasted profile series (profile type x
weather zone), fit period 2021-01..2024-12. **These series are ERCOT model estimates, not meter
readings**, so the clusters describe ERCOT's profile models, not individual customers. Descriptive
only; nothing here is used as a forecasting input.

## Method
- Representations: (1) **shape** = mean daily curve / daily mean for summer and winter weekdays
  (48 values); (2) **weather** = daily kWh regressed on CDD/HDD (zone ERA5, base 18.3 C) and a
  weekend flag, features divided by the base load (cooling, heating, weekend sensitivity);
  (3) both, standardized.
- KMeans k = 2..8, 20 inits. Rule fixed before looking at results: highest silhouette among k
  with min ARI across 10 seeds >= 0.9 and smallest cluster >= 5 series.

## Results

| Representation | k | Silhouette | Min ARI across seeds | ARI vs RES/BUS/NM | Smallest cluster |
|---|---|---|---|---|---|
| shape | 2 | 0.68 | 1.00 | 0.21 | 32 |
| weather | 4 | 0.66 | 1.00 | 0.51 | 32 |
| both | 6 | 0.60 | 0.98 | 0.49 | 16 |

Stability between periods (fit on 2021-22 vs 2023-24, same k): ARI shape 1.00,
weather 0.95.

**Weather clusters (k=4) reproduce ERCOT's own classes exactly:**

| Cluster | Members | Cooling / base | Heating / base | Weekend / base |
|---|---|---|---|---|
| 1 | all RESHI* (high winter ratio: electric heating) | 0.144 | 0.173 | +0.076 |
| 2 | all RESLO* (low winter ratio) | 0.186 | 0.032 | +0.069 |
| 3 | all BUSLO* (low load factor) | 0.054 | 0.019 | -0.425 |
| 0 | all other business, lighting, flat | 0.019 | 0.004 | -0.075 |

(Coefficients: extra kWh per degree-day as a fraction of base daily kWh.)

- Interpretation: heating sensitivity separates the two residential classes (0.17 vs 0.03);
  business low-load-factor premises drop ~43% on weekends (offices/shops), other business classes
  are nearly weather- and weekday-insensitive in these profiles.
- PV / DG / WD variants fall in the same weather cluster as their base type, which suggests they
  are derived from the base profile model (consistent with the data being model output).
- **Shape clusters (k=2)** isolate unusual daily shapes: residential PV variants (midday dip from
  rooftop solar), some business PV, and lighting (night-only); everything else shares one shape
  group. Shape alone separates technology, weather sensitivity separates customer type.

## PV / DG adoption (count snapshots, ESIIDs with the flag, last snapshot of each year)

| Year | PV | DG | WD | Other |
|---|---|---|---|---|
| 2016 | 12,055 | 20 | 399 | 7,242,772 |
| 2017 | 17,587 | 22 | 388 | 7,354,265 |
| 2018 | 23,708 | 23 | 390 | 7,465,624 |
| 2019 | 35,180 | 23 | 383 | 7,579,998 |
| 2020 | 55,056 | 27 | 376 | 7,714,124 |
| 2021 | 85,604 | 34 | 374 | 7,836,187 |
| 2022 | 130,619 | 54 | 370 | 7,913,717 |
| 2023 | 171,088 | 94 | 362 | 8,018,449 |
| 2024 | 193,377 | 290 | 364 | 8,251,654 |
| 2025 | 207,538 | 3,629 | 363 | 8,349,568 |
| 2026 | 218,293 | 15,834 | 361 | 8,436,589 |

PV-flagged ESIIDs grew ~6x from 2019 to 2026. The DG count jumps after 2024; the cause (new
assignments, rule change, batteries) was not investigated.

## Limitations
- Model-output data (see above). Clustering treats every series equally, regardless of how many
  ESIIDs it represents (a 6-ESIID class counts as much as a 3.7-million-ESIID class).
- Zone temperature from 1-2 cities per zone.
- Figures: `reports/figures/A3_shape_centroids.png`, `reports/figures/A3_pv_growth.png`.
