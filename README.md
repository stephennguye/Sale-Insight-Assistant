# Sales Insight Assistant  

An AI-powered assistant for analyzing sales data, predicting customer churn, and answering business questions through retrieval-augmented generation (RAG).  

**Purpose**: This project is designed as a **learning playground** to practice building an end-to-end AI/ML system — from **data ingestion → analytics → ML model training → RAG → serving APIs/UI → testing/CI/CD**. It’s not production-grade but demonstrates how different components fit together.

---

## ✨ Features
- **Data Ingestion**  
  - Load raw CSVs (Superstore sales, Telco churn) into a SQLite database.
- **Analytics KPIs**  
  - Compute metrics (total sales, average discount, total orders).  
  - Store KPIs back into the database.
- **Churn Prediction**  
  - Train an XGBoost model on the Telco churn dataset.  
  - Expose a `/predict` endpoint that returns churn probability for a customer.
- **Retrieval-Augmented Generation (RAG)**  
  - Index text/PDF docs into a FAISS vector store.  
  - Query with natural language and get context-aware answers.
- **API Service**  
  - FastAPI endpoints:  
    - `/kpis` → current KPIs  
    - `/predict` → churn prediction  
    - `/ask` → RAG-powered Q&A  
- **UI**  
  - Streamlit-based frontend  

---

## 📂 Project Structure
```

Sale Insight Assistant/
├── src/
│   ├── app.py              # FastAPI entrypoint
│   ├── ingest.py           # Ingest raw data into SQLite
│   ├── analytics.py        # Compute KPIs
│   ├── train\_model.py      # Train churn model
│   ├── rag\_index.py        # Build FAISS index
│   ├── rag\_query.py        # Retrieval + generation pipeline
│   └── utils/              # Logging & exceptions
├── tests/                  # Pytest-based tests
├── data/                   # Raw data + generated DB/indexes (ignored in git)
├── models/                 # Saved churn models (ignored in git)
├── ui.py                   # Streamlit frontend
├── requirements.txt
├── Makefile
├── ci.yml                  # GitHub Actions CI workflow
├── .gitignore
└── README.md               # This file

````

---

## ⚙️ Installation

Clone the repo:
```bash
git clone https://github.com/<your-username>/sale-insight-assistant.git
cd sale-insight-assistant
````

Create a virtual environment and install dependencies:

```bash
make setup
```

(or manually:)

```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
pip install -r requirements.txt
```

---

## 📊 Sample Datasets

To run this project, download the following datasets and place them under `data/raw/`:

* **Superstore Sales Dataset** (Kaggle):
  👉 [Download here](https://www.kaggle.com/datasets/juhi1994/superstore)
  Save as `data/raw/superstore.csv`.

* **Telco Customer Churn Dataset** (IBM Sample):
  👉 [Download here](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
  Save as `data/raw/telco_churn.csv`.

---

## ▶️ Usage

### 1. Ingest Data

```bash
make ingest
```

### 2. Compute KPIs

```bash
make analytics
```

### 3. Train Churn Model

```bash
make train
```

### 4. Build RAG Index

Place business documents in `data/docs/` (TXT or PDF → text). Then:

```bash
make rag-index
```

### 5. Run the API

```bash
make serve
```

Available endpoints:

* `GET /kpis`
* `POST /predict`
* `GET /ask?query=...`

Example churn prediction request:

```json
POST /predict
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "InternetService": "Fiber optic",
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 845.5
}
```

### 6. Optional UI

```bash
make ui
```

Launches a Streamlit dashboard.

---

## 🧪 Running Tests

Run unit tests:

```bash
make test
```

Lint code:

```bash
make lint
```

---

## 🤖 CI/CD

* **GitHub Actions (`ci.yml`)** runs on each push/PR to `main`:

  * Installs dependencies
  * Runs lint checks
  * Executes ingestion + analytics pipeline

---

## 🚀 Roadmap

* Add richer KPI analytics (monthly/quarterly trends).
* Improve churn model with hyperparameter tuning.
* Swap local generator (`distilgpt2`) with OpenAI.
* Containerize with Docker for deployment.

