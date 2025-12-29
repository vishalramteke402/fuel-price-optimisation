import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fuel Price Optimization", layout="wide")
st.title("⛽ Fuel Price Optimization – ML Engineer Assignment")

# -----------------------------
# FILE UPLOADERS (CLOUD SAFE)
# -----------------------------
st.sidebar.header("📂 Upload Input Files")

csv_file = st.sidebar.file_uploader(
    "Upload oil_retail_history.csv", type=["csv"]
)

json_file = st.sidebar.file_uploader(
    "Upload today_example.json", type=["json"]
)

if csv_file is None or json_file is None:
    st.warning("Please upload both CSV and JSON files to continue.")
    st.stop()

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file, parse_dates=["date"])

def load_today(file):
    return json.load(file)

df = load_data(csv_file)
today = load_today(json_file)

# -----------------------------
# SIDEBAR: CONSTRAINTS
# -----------------------------
st.sidebar.header("⚙️ Business Constraints")

max_price_change = st.sidebar.slider(
    "Max Daily Price Change (₹)", 0.5, 5.0, 2.0
)

min_margin = st.sidebar.slider(
    "Minimum Margin (₹)", 0.5, 5.0, 1.0
)

# -----------------------------
# CLEAN DATA
# -----------------------------
df = df.dropna()
df = df[df["price"] > df["cost"]]
df = df[df["volume"] > 0]

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
df = df.sort_values("date")

df["avg_comp_price"] = df[
    ["comp1_price", "comp2_price", "comp3_price"]
].mean(axis=1)

df["price_diff"] = df["price"] - df["avg_comp_price"]
df["price_lag1"] = df["price"].shift(1)
df["volume_lag1"] = df["volume"].shift(1)
df["volume_ma7"] = df["volume"].rolling(7).mean()

df = df.dropna()

# -----------------------------
# MODEL TRAINING
# -----------------------------
X = df.drop(columns=["date", "volume"])
y = df["volume"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(
    ["📊 Data", "🧠 Model", "💰 Recommendation"]
)

with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.write("Rows:", df.shape[0])

with tab2:
    st.subheader("Model Performance")
    st.metric("MAE (Volume)", round(mae, 2))

    fig, ax = plt.subplots()
    ax.plot(y_test.values, label="Actual")
    ax.plot(preds, label="Predicted")
    ax.legend()
    st.pyplot(fig)

with tab3:
    st.subheader("Today's Input")
    st.json(today)

    if st.button("🚀 Recommend Price"):
        candidate_prices = np.linspace(
            today["cost"] + min_margin,
            today["cost"] + 10,
            30
        )

        best_price, best_profit, best_volume = None, -1, 0

        for p in candidate_prices:
            if abs(p - today["price"]) > max_price_change:
                continue

            features = pd.DataFrame([{
                "price": p,
                "cost": today["cost"],
                "comp1_price": today["comp1_price"],
                "comp2_price": today["comp2_price"],
                "comp3_price": today["comp3_price"],
                "avg_comp_price": np.mean([
                    today["comp1_price"],
                    today["comp2_price"],
                    today["comp3_price"]
                ]),
                "price_diff": p - today["comp1_price"],
                "price_lag1": today["price"],
                "volume_lag1": df["volume"].iloc[-1],
                "volume_ma7": df["volume_ma7"].iloc[-1]
            }])

            volume = model.predict(features)[0]
            profit = (p - today["cost"]) * volume

            if profit > best_profit:
                best_price = p
                best_profit = profit
                best_volume = volume

        st.success("Optimal Price Recommendation")
        c1, c2, c3 = st.columns(3)
        c1.metric("Recommended Price", f"₹ {round(best_price, 2)}")
        c2.metric("Expected Volume", int(best_volume))
        c3.metric("Expected Profit", f"₹ {int(best_profit)}")
