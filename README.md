# lstm-stock-forecasting
LSTM deep learning model for forecasting Tesla (TSLA) and EUR/USD stock prices


A deep learning project that uses a **Long Short-Term Memory (LSTM)** recurrent neural network to forecast the closing price of **Tesla (TSLA)** stock, with a comparative look at the **EUR/USD** exchange rate. Built as part of my Computer Engineering thesis.

> ⚠️ **Disclaimer:** This is a research and learning project. It is **not** financial advice and is **not** intended for real trading decisions.

---

## 📌 Overview

Financial time series are noisy, non-stationary, and hard to predict. The goal of this project was **not** to "beat the market" but to build an end-to-end deep learning forecasting pipeline and understand, hands-on, what LSTMs can and cannot do on real market data.

TSLA was chosen deliberately for its **high volatility and high beta** — a demanding test case for a forecasting model. EUR/USD was added as a contrast: a near-stationary series with very different statistical behaviour.

---

## 🧠 Methodology

| Step | Detail |
|------|--------|
| **Data source** | Daily TSLA history from Yahoo Finance (Open, High, Low, Close, Volume) |
| **Target** | Univariate forecasting on the `Close` price |
| **Cleaning** | `datetime` conversion, datetime index, numeric coercion, missing-value handling |
| **Scaling** | `StandardScaler` |
| **Windowing** | Sliding look-back window of **60 days** per input sequence |
| **Split** | Chronological 80% train / 20% test (no shuffling → no look-ahead bias) |
| **Reshape** | `(samples, timesteps, features)` as required by Keras LSTM |

### Model architecture

```
LSTM(128, return_sequences=True)
   → Dropout(0.2)
LSTM(64)
   → Dense(64, activation='relu')
   → Dropout(0.5)
Dense(1)          # single-value price prediction
```

- **Loss:** MAE   **Optimizer:** Adam   **Metric:** RMSE
- **Training:** batch size 64, 20 epochs

### Why LSTM?

A standard feed-forward network has no memory of previous inputs. LSTMs use gating mechanisms (input, forget, output gates) to **retain relevant information across time steps**, which makes them suited to sequential data where order matters — exactly the case with price series.

---

## 📊 Results

Evaluated on the held-out 20% test set, with predictions inverse-transformed back to real price scale:

| Metric | Value |
|--------|-------|
| MAE | $9.41 |
| RMSE | $12.67 |
| MAPE | 4.21% |
| R² | 0.9534 |

The model tracks the test-set price closely (see the predicted-vs-actual plot in the notebook), with an average absolute error around $9 and a mean percentage error under 5%.

**Honest interpretation:** an R² of 0.95 is strong, but part of it reflects the high autocorrelation of daily prices — tomorrow's close is usually near today's, so a model that follows the level will already score well. The more demanding test is predicting *direction and magnitude of change*, which is the natural next step (see below).

> 🐞 **Debugging note:** my first evaluation returned R² = −14.4. Tracing it revealed a **scale mismatch** — predictions were being scored in normalized space against raw dollar prices. Fixing the evaluation (inverse-transforming before scoring) brought R² to 0.95. The model was never the problem; the measurement was.

---

## 🛠️ Tech Stack

- **Python 3.11**
- **TensorFlow / Keras** — LSTM implementation
- **pandas / NumPy** — data manipulation
- **scikit-learn** — scaling and metrics
- **Matplotlib / Seaborn** — visualization
- **Google Colab** — development environment

---

## 🚀 How to Run

### Google Colab (recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Open the notebook in the `notebooks/` folder and run the cells top to bottom.

### Local

```bash
git clone https://github.com/Hyp3rion90/lstm-stock-forecasting.git
cd lstm-stock-forecasting
pip install -r requirements.txt
jupyter notebook
```

---

## 📁 Project Structure

```
lstm-stock-forecasting/
├── notebooks/
│   └── tsla_lstm_forecasting.ipynb
├── data/
│   └── TSLA_daily_history.csv
├── requirements.txt
└── README.md
```

---

## 💡 What I Learned

- **Sequence modeling:** how to turn a 1-D price series into `(samples, timesteps, features)` windows, and why the look-back length is a key hyperparameter.
- **No data leakage:** financial data must be split chronologically, never shuffled.
- **Scale consistency in evaluation:** metrics must be computed on the *same scale* as the ground truth. A scale mismatch made my R² read −14.4; fixing it (inverse-transforming predictions before scoring) revealed the true value of 0.95. The single most useful debugging lesson of the project — a model is only as trustworthy as its measurement.
- **The hard limits of forecasting:** absolute-price prediction is far harder than it looks; near-stationary series (EUR/USD) expose this dramatically.

## 🔭 Next Steps

- Predict **returns** (% change) instead of absolute prices — a harder, more meaningful target.
- Add walk-forward (rolling) validation.
- Fit the scaler on the training split only, to remove residual data leakage.
- Publish the EUR/USD notebook alongside TSLA.

---

## 👤 Author

**Davide Froiio** — Computer Engineering graduate, transitioning into AI / ML.

- GitHub: [@Hyp3rion90](https://github.com/Hyp3rion90)
- LinkedIn: `[link]`

---

*Built as part of my thesis in Computer Engineering, 2026.*
