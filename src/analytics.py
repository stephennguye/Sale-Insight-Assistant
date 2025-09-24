"""Compute simple KPIs and write back to the SQLite DB."""
import sqlite3
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("analytics")
DB_PATH = Path("data/sales_insights.db")


def compute_kpis():
    if not DB_PATH.exists():
        logger.error("DB not found: %s", DB_PATH)
        raise FileNotFoundError("DB not found. Run ingest first.")

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM superstore", conn)
        # basic KPIs
        total_sales = float(df["Sales"].sum())
        avg_discount = float(df["Discount"].mean())
        orders = int(df.shape[0])

        kpis = pd.DataFrame([
            {"metric": "total_sales", "value": total_sales},
            {"metric": "avg_discount", "value": avg_discount},
            {"metric": "orders", "value": orders},
        ])

        kpis.to_sql("kpis", conn, if_exists="replace", index=False)
        logger.info("KPIs computed and stored (total_sales=%.2f, orders=%d)", total_sales, orders)
        return kpis
    finally:
        conn.close()


if __name__ == "__main__":
    print(compute_kpis().to_dict(orient="records"))
