import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import json

st.set_page_config(page_title="Fuel Price Optimization", layout="wide")

st.title("⛽ Fuel Price Optimization – ML App")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("oil_retail_history.csv", parse_dates=["date"])
    return df

df = load_data()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Business Constraints")
max_price_change = st.sidebar.slider(
    "Max Daily Price Change", 0.5, 5.0, 2.0
)
min_margin = st.sidebar.slider(
    "Minimum Margin", 0.5, 5.0, 1.0
)

# -----------------------------
# DATA CLEANING
# -----------------------------
df = df.dropna()
df = df[df["price"] > df["cost"]]
df = df[df["volume"] > 0]

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
df = df.sort_values("date")
df["avg_comp_price"] = df[["comp1_price", "comp2_price", "comp3_price"]].mean(axis=1)
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
    n_estimators=300,
    max_depth=12,
    random_state=42
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(
    ["📊 Data Exploration", "🧠 Model Performance", "💰 Price Recommendation"]
)

# -----------------------------
# TAB 1: EDA
# -----------------------------
with tab1:
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write("Rows:", df.shape[0])

    st.subheader("Price vs Volume")
    fig, ax = plt.subplots()
    ax.scatter(df["price"], df["volume"], alpha=0.3)
    ax.set_xlabel("Price")
    ax.set_ylabel("Volume")
    st.pyplot(fig)

# -----------------------------
# TAB 2: MODEL PERFORMANCE
# -----------------------------
with tab2:
    st.subheader("Demand Model Evaluation")
    st.metric("Mean Absolute Error (Volume)", round(mae, 2))

    fig, ax = plt.subplots()
    ax.plot(y_test.values, label="Actual", alpha=0.7)
    ax.plot(preds, label="Predicted", alpha=0.7)
    ax.legend()
    ax.set_title("Actual vs Predicted Demand")
    st.pyplot(fig)

# -----------------------------
# TAB 3: OPTIMIZATION
# -----------------------------
with tab3:
    st.subheader("Today's Market Input")

    with open("today_example.json") as f:
        today = json.load(f)

    st.json(today)

    def recommend_price():
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
                best_profit = profit
                best_price = p
                best_volume = volume

        return best_price, best_volume, best_profit

    if st.button("🚀 Recommend Optimal Price"):
        price, volume, profit = recommend_price()

        st.success("Optimal Price Recommendation")
        col1, col2, col3 = st.columns(3)
        col1.metric("Recommended Price", f"₹ {round(price,2)}")
        col2.metric("Expected Volume", int(volume))
        col3.metric("Expected Profit", f"₹ {int(profit)}")
