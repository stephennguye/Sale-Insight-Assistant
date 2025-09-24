import pytest
from pathlib import Path
from src.ingest import main as ingest_main
from src.analytics import compute_kpis


def test_kpis(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    (data_dir / 'superstore.csv').write_text('Order ID,Sales,Discount\n1,100,0.1\n2,200,0.0')
    (data_dir / 'telco_churn.csv').write_text('customerID,Churn\n1,No')

    monkeypatch.chdir(tmp_path)
    ingest_main()
    kpis = compute_kpis()
    metrics = {r['metric']: r['value'] for r in kpis.to_dict(orient='records')}
    assert metrics['total_sales'] == 300
    assert metrics['orders'] == 2