import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Fuel Price Optimization", layout="wide")
st.title("⛽ Fuel Price Optimization – ML Engineer Assignment")

st.markdown("""
This application recommends the **optimal daily fuel price** that maximizes profit  
based on historical demand, costs, and competitor pricing.
""")

# -------------------------------------------------
# FILE UPLOAD (CLOUD SAFE)
# -------------------------------------------------
st.sidebar.header("📂 Upload Input Files")

csv_file = st.sidebar.file_uploader(
    "Upload oil_retail_history.csv", type=["csv"]
)

json_file = st.sidebar.file_uploader(
    "Upload today_example.json", type=["json"]
)

if csv_file is None or json_file is None:
    st.warning("⬅️ Please upload both CSV and JSON files to continue.")
    st.stop()

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
d
