"""Ingest raw CSV files into a SQLite database."""
import sqlite3
from pathlib import Path
import pandas as pd
from src.utils.logger import get_logger
from src.utils.exceptions import DataIngestionError

logger = get_logger("ingest")
RAW_DIR = Path("data/raw")
DB_PATH = Path("data/sales_insights.db")


def ingest_csv_to_table(csv_path: Path, table_name: str, conn: sqlite3.Connection):
    try:
        try:
            # Try default UTF-8
            df = pd.read_csv(csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback for Windows/Excel exports
            df = pd.read_csv(csv_path, encoding="latin1")

        df.to_sql(table_name, conn, if_exists="replace", index=False)
        logger.info(f"Ingested {csv_path} -> {table_name} ({len(df)} rows)")
    except Exception as e:
        logger.exception("Failed ingesting %s", csv_path)
        raise DataIngestionError(str(e))



def main():
    if not RAW_DIR.exists():
        logger.error("Raw data folder missing: %s", RAW_DIR)
        raise DataIngestionError("Raw data folder missing")

    conn = sqlite3.connect(DB_PATH)
    try:
        # Expected filenames
        files = {
            "superstore.csv": "superstore",
            "telco_churn.csv": "churn",
        }
        for fname, table in files.items():
            p = RAW_DIR / fname
            if not p.exists():
                logger.warning("File not found, skipping: %s", p)
                continue
            ingest_csv_to_table(p, table, conn)
    finally:
        conn.close()
        logger.info("Ingestion finished, DB at %s", DB_PATH)


if __name__ == "__main__":
    main()