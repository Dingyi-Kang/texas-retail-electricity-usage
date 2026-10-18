"""Module A1 tests: DST day lengths, availability of lag features, weather
vintage timing, metric hand checks."""
import numpy as np
import pandas as pd
import pytest
from src.ru.ingest import day_intervals_utc
from src.ru.metrics import weighted, block_bootstrap_diff
from src.ru.wx import forecast_city, PUB_DELAY


def test_dst_interval_counts():
    assert len(day_intervals_utc("2025-03-09")) == 92
    assert len(day_intervals_utc("2025-11-02")) == 100
    assert len(day_intervals_utc("2025-06-01")) == 96


@pytest.fixture(scope="module")
def table():
    from src.ru.backtest import model_table
    return model_table()


def test_hours_per_day(table):
    n = table.groupby(["series", "T"]).size()
    assert set(n.unique()) <= {23, 24, 25}
    assert (n.xs(pd.Timestamp("2025-03-09"), level="T") == 23).all()
    assert (n.xs(pd.Timestamp("2025-11-02"), level="T") == 25).all()


def test_lags_only_from_published_days(table):
    d = pd.read_parquet("data/processed/profile_days.parquet")
    d["addtime"] = pd.to_datetime(d["addtime"]).dt.normalize()
    add = d.drop_duplicates(["series", "op_date"], keep="last").set_index(["series", "op_date"])["addtime"]
    s = table.sample(20000, random_state=0)
    for k in [6, 7, 14]:
        idx = pd.MultiIndex.from_arrays([s["series"], s["T"] - pd.Timedelta(days=k)])
        known = add.reindex(idx).values < s["issue_date"].values
        assert s[f"lag{k}"].isna().values[~known].all()      # unpublished lag days are masked
    assert (s["known_max_day"] < s["issue_date"]).all()


def test_weather_vintage_before_issue():
    starts = pd.date_range("2025-07-02 05:00", periods=30, freq="h", tz="UTC")   # local day 2025-07-02
    issue = pd.DatetimeIndex([pd.Timestamp("2025-07-01 18:00", tz="America/Chicago").tz_convert("UTC")] * 30)
    for model, d in PUB_DELAY.items():
        ends = starts + pd.Timedelta(hours=1)
        run1 = (ends - pd.Timedelta(hours=24)).floor("6h")
        used1 = run1 + pd.Timedelta(hours=d) <= issue
        run2 = (ends - pd.Timedelta(hours=48)).floor("6h")
        assert (run2 + pd.Timedelta(hours=d) <= issue).all()
        assert (run1[used1] + pd.Timedelta(hours=d) <= issue[used1]).all()


def test_weighted_wape_hand():
    df = pd.DataFrame({"y": [1.0, 2.0], "pred": [2.0, 2.0], "w": [10, 30]})
    assert weighted(df)["wape"] == pytest.approx(100 * 10 / 70)


def test_bootstrap_identical_is_zero():
    d = pd.DataFrame({"ae": np.arange(30.0) + 1, "yw": np.full(30, 100.0)}, index=pd.date_range("2025-01-01", periods=30))
    diff, ci = block_bootstrap_diff(d, d)
    assert diff == 0 and ci[0] == 0 and ci[1] == 0
