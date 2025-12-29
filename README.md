⛽ Fuel Price Optimization using Machine Learning
📌 Project Overview

This project implements a machine learning–driven dynamic pricing system for a retail fuel company operating in a competitive market.
The system recommends an optimal daily fuel price that maximizes expected profit while respecting real-world business constraints.

The solution includes:

End-to-end data ingestion and feature engineering pipeline

Random Forest–based demand modeling

Profit-driven price optimization logic

A fully interactive Streamlit Cloud application

🎯 Business Objective

To recommend a daily retail fuel price that:

Maximizes total profit

Remains competitive with market prices

Enforces business rules such as price stability and minimum margins

🧠 Solution Approach
1️⃣ Data Understanding

The model learns demand behavior using ~2 years of historical data containing:

Company price & cost

Competitor prices

Daily sales volume

Temporal patterns

2️⃣ Data Engineering Pipeline

The pipeline performs:

Data ingestion (batch simulation via CSV upload)

Validation & cleaning

Feature computation:

Average competitor price

Price difference vs competitors

Lag features (price, volume)

Rolling averages (7-day demand trend)

All features are computed dynamically and cached for performance.

3️⃣ Machine Learning Model

Algorithm: Random Forest Regressor

Target Variable: Daily fuel volume sold

Why Random Forest?

Handles non-linear price-demand relationships

Robust to noise and multicollinearity

Strong baseline for tabular data

Evaluation Metric

Mean Absolute Error (MAE) on hold-out time-based validation

4️⃣ Price Optimization Strategy

For a new day:

Generate candidate prices within a business-allowed range

Predict expected demand for each price

Compute profit = (price − cost) × predicted volume

Select the price that yields maximum expected profit

5️⃣ Business Constraints Applied

Maximum daily price change

Minimum profit margin

Competitor price alignment

No negative-margin pricing

📊 Streamlit Application Features

Upload historical CSV data

Upload daily input JSON

Adjustable business constraints

Model performance visualization

One-click optimal price recommendation

Expected volume & profit estimation

📂 Project Structure
fuel-price-optimisation/
│
├── main.py              # Streamlit application
├── requirements.txt     # Dependencies
└── README.md            # Project documentation

▶️ How to Run the App (Local or Cloud)
Option 1: Streamlit Cloud (Recommended)

Push repository to GitHub

Go to https://streamlit.io/cloud

Select main.py and deploy

Upload dataset & JSON via UI

Option 2: Run Locally
pip install -r requirements.txt
streamlit run main.py

📥 Input Format
Historical Data (oil_retail_history.csv)
date, price, cost, comp1_price, comp2_price, comp3_price, volume

Daily Input (today_example.json)
{
  "date": "2025-01-01",
  "price": 105.0,
  "cost": 96.5,
  "comp1_price": 104.0,
  "comp2_price": 105.5,
  "comp3_price": 106.0
}

📤 Output

Recommended Price

Expected Sales Volume

Expected Daily Profit

🚀 Key Skills Demonstrated

Machine Learning (Random Forest, Regression)

Feature Engineering for Time-Series Data

Profit Optimization Logic

Business Rule Integration

Streamlit App Development

End-to-End ML Pipeline Design

Model Evaluation & Visualization

📈 Future Improvements

Reinforcement learning for long-term pricing strategy

Elasticity-based demand modeling

Automated retraining pipeline (Airflow/Prefect)

API deployment using FastAPI

Real-time competitor price ingestion

👤 Author

Vishal Ramteke
