import numpy as np
import pandas as pd
import streamlit as st
import joblib

@st.cache_resource
def load_artifacts():
    rf = joblib.load("rf_demand_model.pkl")
    features = joblib.load("model_features.pkl")
    encoders = joblib.load("label_encoders.pkl")
    return rf, features, encoders

rf, features, encoders = load_artifacts()

st.title("Demand Forecasting App")
st.divider()
st.header("Input Features")

price = st.number_input("Price", min_value=0.0, value=50.0)
discount = st.number_input("Discount (%)", min_value=0, max_value=100, value=0)
inventory_level = st.number_input("Inventory Level", min_value=0, value=100)
promotion = st.selectbox("Promotion", [0, 1])
competitor_pricing = st.number_input("Competitor Price", min_value=0.0, value=50.0)
category = st.selectbox("Category", encoders["Category"].classes_.tolist())
region = st.selectbox("Region", encoders["Region"].classes_.tolist())
weather = st.selectbox("Weather Condition", encoders["Weather Condition"].classes_.tolist())
seasonality = st.selectbox("Seasonality", encoders["Seasonality"].classes_.tolist())


category_avg = joblib.load("category_avg_demand.pkl")

if st.button("Predict Demand"):
      input_data = pd.DataFrame({
            "Price": [price],
            "Discount": [discount],
            "Inventory Level": [inventory_level],
            "Promotion": [promotion],
            "Competitor Pricing": [competitor_pricing],
            "Category_enc": [encoders["Category"].transform([category])[0]],
            "Region_enc": [encoders["Region"].transform([region])[0]],
            "Weather Condition_enc": [encoders["Weather Condition"].transform([weather])[0]],
            "Seasonality_enc": [encoders["Seasonality"].transform([seasonality])[0]],
        })[features] 
      tree_preds = np.array([tree.predict(input_data)[0] for tree in rf.estimators_])
      prediction = tree_preds.mean()
      lower, upper = np.percentile(tree_preds, [10, 90])

      avg_for_category = category_avg.get(category, None)
      delta = prediction - avg_for_category if avg_for_category else None
      st.metric(
        "Predicted Demand",
        f"{prediction:.0f} units",
        delta=f"{delta:+.0f} vs {category} avg" if delta is not None else None
    )
      st.caption(f"Estimated range: {lower:.0f} – {upper:.0f} units (based on model prediction spread across the ensemble)")
