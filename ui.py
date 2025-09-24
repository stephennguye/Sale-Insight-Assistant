import streamlit as st
import requests
import json

st.set_page_config(page_title="Sales Insights Assistant")
st.title("Sales Insights Assistant")

st.sidebar.markdown("## Controls")
if st.sidebar.button("Refresh KPIs"):
    st.experimental_rerun()

st.header("Key Performance Indicators")
if st.button("Load KPIs"):
    try:
        r = requests.get("http://localhost:8000/kpis", timeout=5)
        r.raise_for_status()
        kpis = r.json()
        st.write(kpis)
    except Exception as e:
        st.error(f"Failed to fetch KPIs: {e}")

st.header("Churn Prediction (demo)")
cust_json = st.text_area("Customer JSON", value='{"tenure": 1, "MonthlyCharges": 29.85}')
if st.button("Predict churn"):
    try:
        payload = json.loads(cust_json)
        r = requests.post("http://localhost:8000/predict", json=payload, timeout=5)
        r.raise_for_status()
        st.json(r.json())
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.header("Ask documents")
q = st.text_input("Your question")
if st.button("Ask"):
    try:
        r = requests.get("http://localhost:8000/ask", params={"query": q}, timeout=10)
        r.raise_for_status()
        st.write(r.json().get("answer"))
    except Exception as e:
        st.error(f"RAG query failed: {e}")