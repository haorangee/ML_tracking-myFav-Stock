# ML_tracking-myFav-Stock

A machine-learning project that tracks and predicts **Nvidia (NVDA)** stock prices.

---

## Features

| Feature | Description |
|---|---|
| **Data fetching** | Downloads historical NVDA OHLCV data from Yahoo Finance via `yfinance` |
| **Feature engineering** | Moving averages (MA5/10/20/50), EMA, MACD, Bollinger Bands, RSI-14, lag prices, volume features |
| **ML model** | Random Forest *or* Gradient Boosting regressor (scikit-learn pipeline) |
| **Training** | Chronological 80/20 train-test split + time-series cross-validation |
| **Prediction** | Next-day close-price predictions with a directional BUY / SELL / HOLD signal |
| **Visualisation** | Actual vs predicted price chart with residuals, saved as PNG |

---

## Project structure

```
ML_tracking-myFav-Stock/
├── data_fetcher.py   # Download / load NVDA data
├── features.py       # Feature engineering pipeline
├── model.py          # Model building, training, evaluation, persistence
├── train.py          # End-to-end training script
├── predict.py        # Load saved model and predict next-day price
├── visualize.py      # Plot actual vs predicted prices
├── requirements.txt
├── tests/
│   └── test_features.py   # Unit tests for feature engineering
└── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

Downloads fresh NVDA data, engineers features, trains a Random Forest, and saves the model to `models/nvda_rf_model.joblib`.

```bash
python train.py
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--start YYYY-MM-DD` | `2015-01-01` | Historical data start date |
| `--model-type rf\|gbm` | `rf` | Random Forest or Gradient Boosting |
| `--no-fetch` | — | Use cached CSV instead of downloading |
| `--output PATH` | `models/nvda_rf_model.joblib` | Where to save the model |

Example with Gradient Boosting:

```bash
python train.py --model-type gbm --start 2018-01-01
```

### 3. Predict next-day price

```bash
python predict.py
```

Sample output:

```
Date         Actual Close  Predicted Next      Signal
-------------------------------------------------------
2024-12-18       131.3800       133.2156    🟢 BUY
2024-12-19       128.3100       130.0943    🟢 BUY
...
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--days N` | `10` | How many recent trading days to show |
| `--no-fetch` | — | Use cached data |
| `--model PATH` | `models/nvda_rf_model.joblib` | Path to saved model |

### 4. Visualise predictions

```bash
python visualize.py
```

Saves a chart to `data/nvda_predictions.png` showing actual vs predicted prices on the test set together with the per-day prediction error.

---

## Running the tests

```bash
python -m pytest tests/ -v
```

---

## How it works

1. **Data** – `yfinance` downloads daily OHLCV data for NVDA since 2015.
2. **Features** – 30 derived features capture trend, momentum, volatility, and volume behaviour.
3. **Target** – the next trading day's closing price.
4. **Model** – a scikit-learn `Pipeline` (StandardScaler → RandomForestRegressor) is trained on the first 80 % of dates and evaluated on the remaining 20 %.
5. **Prediction signal** – if the predicted next-day price is >0.5 % above today's close the script prints 🟢 **BUY**; >0.5 % below prints 🔴 **SELL**; otherwise 🟡 **HOLD**.

> **Disclaimer** – This project is for educational and research purposes only.  
> It does **not** constitute financial advice.  Past performance is no guarantee of future results.
