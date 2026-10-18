"""EIA-861M sales_revenue.xlsx (Monthly-States) -> Texas monthly table:
month, sector (residential/commercial), sales_mwh, customers, revenue_k, price_c, data_status."""
import pandas as pd

def main():
    raw = pd.read_excel("data/raw/eia/sales_revenue.xlsx", sheet_name="Monthly-States", header=None)
    body = raw.iloc[3:].copy()
    tx = body[body[2] == "TX"]
    rows = []
    for sector, c0 in [("residential", 4), ("commercial", 8), ("industrial", 12)]:
        rows.append(pd.DataFrame({"month": pd.to_datetime(dict(year=tx[0].astype(int), month=tx[1].astype(int), day=1)),
                                  "sector": sector, "status": tx[3].values,
                                  "revenue_k": pd.to_numeric(tx[c0], errors="coerce").values,
                                  "sales_mwh": pd.to_numeric(tx[c0 + 1], errors="coerce").values,
                                  "customers": pd.to_numeric(tx[c0 + 2], errors="coerce").values,
                                  "price_c": pd.to_numeric(tx[c0 + 3], errors="coerce").values}))
    d = pd.concat(rows).sort_values(["sector", "month"]).reset_index(drop=True)
    d["kwh_per_customer"] = d["sales_mwh"] * 1000 / d["customers"]
    d.to_parquet("data/processed/eia_tx_monthly.parquet", index=False)
    print(d.groupby("sector")["month"].agg(["min", "max", "count"]))
    print(d[d.sector == "residential"].tail(4).to_string())
    print(d.groupby(["sector", "status"]).size())

if __name__ == "__main__":
    main()
