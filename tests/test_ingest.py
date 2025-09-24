import os
from pathlib import Path
import sqlite3
import pytest

from src.ingest import main as ingest_main


def test_ingest_creates_db(tmp_path, monkeypatch):
    # point data/raw to fixture tmp files
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    # create tiny csvs
    (data_dir / 'superstore.csv').write_text('Order ID,Sales,Discount\n1,100,0.1\n2,200,0.0')
    (data_dir / 'telco_churn.csv').write_text('customerID,Churn\n1,No')

    monkeypatch.chdir(tmp_path)
    ingest_main()
    assert Path('data/sales_insights.db').exists()

    conn = sqlite3.connect('D:/GitHub/Sale Insight Assistant/data/sales_insights.db')
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    names = [r[0] for r in cur.fetchall()]
    assert 'superstore' in names
    assert 'churn' in names
    conn.close()