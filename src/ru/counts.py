"""ERCOT Profile Type Counts snapshots -> tidy table
(snapshot_date, weather_zone, meter_type, tdsp, profile_type, esiids). The file name carries the
snapshot date; ERCOT's own posting dates are not in the archive."""
import re, zipfile, json
from pathlib import Path
import pandas as pd

RAW = Path("data/raw/counts/counts.zip")
# typos in ERCOT file names; dates confirmed by neighbouring snapshots and zip timestamps
FIX = {"201500803": "20150803", "201605614": "20160614"}


def main():
    rows, bad = [], []
    with zipfile.ZipFile(RAW) as z:
        for n in z.namelist():
            b = n.split("/")[-1]
            m = re.search(r"Counts (\d{8,9})(_corrected)?\s*\.xlsx$", b)
            if not m:
                continue
            ds = FIX.get(m.group(1), m.group(1))
            if m.group(2) is None and any(x.endswith(m.group(1) + "_corrected.xlsx") for x in z.namelist()):
                continue                 # a corrected file exists for this snapshot
            try:
                d = pd.read_excel(z.open(n), sheet_name="Data")
            except Exception as ex:
                bad.append((n, repr(ex)[:80])); continue
            d.columns = [str(c).strip().upper() for c in d.columns]
            need = {"WEATHERZONE", "PROFILETYPE", "RECORDS"}
            if not need <= set(d.columns):
                bad.append((n, list(d.columns))); continue
            keep = [c for c in ["WEATHERZONE", "METERDATATYPE", "TDSPNAME", "PROFILETYPE", "RECORDS"] if c in d.columns]
            d = d.loc[:, ~d.columns.duplicated()][keep].copy()
            d["snapshot"] = pd.to_datetime(ds, format="%Y%m%d")
            rows.append(d)
    c = pd.concat(rows, ignore_index=True).rename(columns={"WEATHERZONE": "weather_zone", "METERDATATYPE": "meter_type",
                                                          "TDSPNAME": "tdsp", "PROFILETYPE": "profile_type", "RECORDS": "esiids"})
    c = c.reindex(columns=["snapshot", "weather_zone", "meter_type", "tdsp", "profile_type", "esiids"])
    c = c.dropna(subset=["weather_zone", "profile_type", "esiids"])
    c.to_parquet("data/processed/counts.parquet", index=False)
    print(len(c), c.snapshot.nunique(), c.snapshot.min().date(), c.snapshot.max().date(), "bad:", bad[:5])
    tot = c.groupby("snapshot")["esiids"].sum()
    print(tot.iloc[[0, len(tot)//2, -1]])


if __name__ == "__main__":
    main()
