# Demand Forecasting App
Predicting product demand from price, promotions, and market conditions using a Random Forest model — deployed as an interactive Streamlit app.

**🔗 Live app:** [salmalas-demand-forecasting.streamlit.app](https://salmalas-demand-forecasting.streamlit.app/)

![App input form](screenshot-inputs.png)
![App prediction output](screenshot-prediction.png)

## Overview

This project analyzes what actually drives product demand price, discounts, promotions, inventory levels, competitor pricing, category, region, weather, and seasonality. Then, it turns the resulting model into a live prediction tool. A Random Forest model beats a category mean baseline by roughly 35% (MAE), and feature importance analysis identifies price as the single strongest driver of demand.

## Dataset

- Source: [GitHub](https://github.com/Onurbltc/DemandForecastingDataset)
- 76,000 rows, spanning January 2022 – January 2024
- Multi-store, multi-product retail data across 5 categories (Electronics, Clothing, Groceries, Furniture, Toys) and 4 regions
- Columns include: `Price`, `Discount`, `Inventory Level`, `Promotion`, `Competitor Pricing`, `Category`, `Region`, `Weather Condition`, `Seasonality`, `Units Sold`, `Units Ordered`, `Epidemic`, `Demand`

## Approach

Two modeling directions were explored:
1. **Time-series forecasting** : predicting total daily demand from date/lag/rolling features
2. **Feature-based prediction** : predicting demand from price, promotion, and market conditions on a given row

The project settled on the **feature-based approach**, since the dataset's rich contextual columns (price, promotions, competitor pricing, weather, seasonality) carry more signal for this data than pure time trends, and the resulting model answers a more actionable business question: *what drives demand, and what happens if conditions change?*

## Results

### Baselines
| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Global mean | 36.78 | 46.73 | 56.17% |
| Category mean | 34.19 | 43.35 | 50.23% |

### Random Forest (9 features)
| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Random Forest | **22.32** | **31.81** | 30.94% |

The model reduces error by **~35% (MAE)** and **~27% (RMSE)** over the category-mean baseline.

**Note on MAPE:** MAPE remains high (~31%) even though MAE/RMSE improve substantially. A small number of rows with genuinely low demand (minimum = 4 units) produce very large percentage errors even when the absolute error is small. MAE and RMSE are the more representative metrics for this dataset.

### Hyperparameter tuning
A randomized search (`RandomizedSearchCV`, 3-fold CV) over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features` was run. The tuned model (MAE=23.51) did **not** outperform the default configuration (MAE=22.32), suggesting the defaults were already well-suited to this dataset.

### Feature importance
| Feature | Importance |
|---|---|
| Price | 33.0% |
| Category | 14.2% |
| Competitor Pricing | 13.3% |
| Inventory Level | 13.3% |
| Promotion | 8.0% |
| Region | 6.5% |
| Discount | 4.8% |
| Seasonality | 3.8% |
| Weather Condition | 3.2% |

**Price alone accounts for a third of the model's predictive signal**, by far the strongest single driver of demand in this dataset, followed by category and competitor pricing.

### Prediction confidence
Rather than a single point estimate, the app reports a range alongside each prediction (which is computed from the spread of predictions across the Random Forest's individual trees) plus a comparison to that category's average demand, so a prediction like "113 units" comes with context (e.g. *"98–128 units, +16 vs. Electronics avg"*) instead of an unqualified number.

## Deployment optimization
The initial model (300 trees, unbounded depth, trained on the full dataset) serialized to **1.82GB** which was far too large to deploy or version in GitHub. To ship a usable model:

- Reduced to **100 estimators, max depth 15**
- Compressed the serialized model with `joblib` (`compress=9`)
- Verified the smaller model on the real held-out test set before shipping: **MAE=23.01** vs. the original **MAE=22.32**  resulting in a **<3% difference**, confirming the size reduction cost negligible accuracy
This is a deliberate accuracy/deployability tradeoff, verified rather than assumed.

## Tech stack

- **Analysis:** Python, pandas, NumPy, scikit-learn, matplotlib, seaborn
- **App:** Streamlit
- **Model serialization:** joblib

## Repository structure

```
├── Demand_Forecasting.ipynb     # Full analysis: EDA, baselines, modeling, evaluation
├── Demand_Forecasting.py        # Streamlit app
├── demand_forecasting.csv       # Dataset
├── rf_demand_model.pkl          # Trained, compressed Random Forest model
├── model_features.pkl           # Feature column order used by the model
├── label_encoders.pkl           # Fitted LabelEncoders for categorical features
├── category_avg_demand.pkl      # Per-category average demand (for context in predictions)
└── requirements.txt
```

## Running locally

```bash
git clone https://github.com/salmalas/Demand-Forecasting-Predicting-Product-Demand-from-Price-Promotions-Market-Conditions.git
cd Demand-Forecasting-Predicting-Product-Demand-from-Price-Promotions-Market-Conditions
pip install -r requirements.txt
streamlit run Demand_Forecasting.py
```

## Future work

- A true multi-step time-series forecast (date-driven, recursive prediction) as a complementary view alongside the current conditions-based model
- A price/promotion "what-if" comparison view, letting users see how demand shifts across a range of prices rather than one prediction at a time
