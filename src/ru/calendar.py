"""Holiday calendar copied from ~/ML_forecast/src/elf/features.py (same project author)."""
import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, Holiday, USLaborDay, USMartinLutherKingJr,
                                    USMemorialDay, USPresidentsDay, USThanksgivingDay, USColumbusDay,
                                    nearest_workday)



class LoadHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("new_year", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr, USPresidentsDay, USMemorialDay,
        Holiday("july4", month=7, day=4, observance=nearest_workday),
        USLaborDay, USColumbusDay,
        Holiday("veterans", month=11, day=11),
        USThanksgivingDay,
        Holiday("christmas", month=12, day=25, observance=nearest_workday),
    ]


def holiday_table(start, end):
    """Known-in-advance calendar: holiday id per date plus a few load-relevant special days."""
    cal = LoadHolidays().holidays(start, end, return_name=True)
    hol = pd.Series(cal.values, index=cal.index.date)
    days = pd.date_range(start, end, freq="D")
    out = pd.DataFrame(index=days.date)
    out["holiday"] = hol.reindex(out.index).fillna("none")
    tg = [d for d, n in hol.items() if "Thanksgiving" in n]
    for d in tg:
        f = d + pd.Timedelta(days=1)
        if f in out.index:
            out.loc[f, "holiday"] = "black_friday"
    for y in sorted({d.year for d in out.index}):
        for (m, dd, name) in [(12, 24, "christmas_eve"), (12, 26, "dec26"), (12, 31, "nye")]:
            d = pd.Timestamp(y, m, dd).date()
            if d in out.index and out.loc[d, "holiday"] == "none":
                out.loc[d, "holiday"] = name
    # Christmas-to-New-Year week is consistently low load
    idx = pd.DatetimeIndex(out.index)
    out["xmas_week"] = (((idx.month == 12) & (idx.day >= 24)) | ((idx.month == 1) & (idx.day <= 2))).astype(int)
    return out
