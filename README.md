# CS506 Final Project  
**Stock Analysis and S&P 500 Trend Prediction**  
**Members**: Abidul, Ahemed, Abdullahi  


**Video Link**: [Watch here](https://www.youtube.com/watch?v=m6gIwRpMOj0&ab_channel=AbdullahiNur)
**[Midterm Report](https://github.com/ahemedbullo/CS506-Final-Project/blob/main/Midterm%20Report.pdf)


---

## Summary

The S&P 500 Index is a benchmark for U.S. stock market performance, influenced by the combined performance of its constituent companies. In this project, we aim to analyze how the daily movements of major stocks relate to the S&P 500 and develop a model that predicts the index’s next-day percentage change based on selected high-impact stocks. Our analysis includes identifying the most correlated stocks, understanding their influence on the index, and building a regression-based prediction model. The insights can help investors, analysts, and financial planners understand market behavior and evaluate predictive signals.

---

## Project Description

We investigate the predictive relationship between the daily returns of individual S&P 500 stocks and the index itself. Specifically, we:

- Analyze the correlation between each stock and the S&P 500’s daily returns.
- Select a subset of the most correlated stocks.
- Build a multivariate linear regression model to forecast the **next-day** percentage change of the S&P 500 based on historical daily returns of these stocks.

This project explores whether such a machine learning-based approach provides any advantage over static weighted averages or basic statistical techniques.

---

## Goals

- **Correlation Analysis**: Identify which S&P 500 stocks are most correlated with the index’s movement.
- **Predictive Modeling**: Train a regression model to forecast next-day S&P 500 percent change using selected stock data.
- **Evaluate Against Baselines**: Compare model performance with baselines like moving averages and static weighted combinations of major stocks (e.g., AAPL, MSFT, NVDA).
- **Model Interpretability**: Analyze which stocks contribute most to the model’s predictions.

---

## Dataset

We use historical daily stock data (2015–2025) including:

- **S&P 500 Index**: Daily close prices and percentage changes.
- **Constituent Stocks**: Including but not limited to AAPL, MSFT, NVDA, META, TSLA — companies that significantly impact the index.
- Data sources include [Yahoo Finance](https://finance.yahoo.com), [Alpha Vantage](https://www.alphavantage.co/), and Kaggle datasets.

---

## Data Preprocessing

- **Cleaning**: Removed NaNs, handled stock splits and non-trading days via forward-filling or interpolation.
- **Feature Engineering**:
  - Daily return = (Close - Previous Close) / Previous Close
  - Z-score normalization applied to all features
  - Lag features added to capture temporal patterns
- **Outlier Handling**: Returns exceeding ±15% are treated as outliers and smoothed via winsorization.
- **Static Weighted Baseline**: Added a baseline using market-cap-weighted average of selected stocks to evaluate model necessity.

---

## Feature Extraction

- **Lag Features**: Daily percent returns over past 3 days
- **Stock Features**: For each of the top-correlated stocks (e.g., AAPL, MSFT, NVDA)
- **Target**: Next-day percent change of S&P 500 index

---

## Modeling Approach

We implemented the following model pipeline:

- **Model**: Linear Regression (multi-feature)
- **Baselines**:
  - Static weighted average (e.g., using AAPL/MSFT/NVDA weights from current S&P 500)
  - 3-day moving average of the index
- **Data Split**:
  - 80% training (2015–2022), 20% testing (2023–2025)
  - No validation set used due to time-series nature; cross-validation not applied to avoid leakage
- **Overfitting Prevention**:
  - No peeking into test set
  - Model evaluated only once on the test set after final training
- **Loss Function**: Mean Absolute Error (MAE)
- **Model Evaluation**: Compared against baseline models

---

## Visualization

We present the following plots inline in the notebook:

- **Time-Series Line Plot**: Actual vs Predicted S&P 500 changes over test set
- **Feature Correlation Heatmap**: Stocks most correlated with the S&P 500
- **Residual Plot**: Model errors vs actual changes
- **Feature Importance**: Regression coefficients ranked by magnitude

---

## Reproducibility

To reproduce our results:

1. Download:
   - `stock_data.csv` — Cleaned and merged dataset
   - `stock_predictor.ipynb` — Main model notebook

2. Run:
   - Open `stock_predictor.ipynb` in Jupyter
   - Upload `stock_data.csv`
   - Execute all cells for preprocessing, modeling, and evaluation

---

## Performance

| Model                   | MAE (%) | Comments                                          |
|------------------------|---------|---------------------------------------------------|
| Static Weighted Avg    | 0.56    | Based on top 3 stocks' market-cap weights         |
| 3-Day Moving Average   | 0.62    | Simple baseline ignoring stock-level dynamics     |
| **Linear Regression**  | **0.47**| Trained on top 10 correlated stock returns        |

The regression model outperformed both baselines, indicating additional signal beyond static weights.

---

## Limitations & Discussion

- **Predictability Concern**: Some test cases may be “predicting” values that can be reconstructed with static rules. We addressed this by comparing to a market-cap weighted static baseline.
- **Overfitting Risk**: Time-based split used; no leakage confirmed. Future iterations may explore walk-forward validation.
- **Real-world Utility**: This model demonstrates predictability *within* known data bounds, but real-time trading would need forward-facing validation and robustness testing.

---

## Future Work

- Add RNN/GRU modeling to capture deeper time dependencies
- Incorporate macroeconomic features (e.g., interest rates, unemployment)
- Use rolling walk-forward validation to simulate live deployment

