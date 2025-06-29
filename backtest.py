import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix, classification_report
from train import model
from data_processing import DataProcessor

"""
Backtest configuration:
- CSV_PATH: Path to historical data
- N_DATAPOINTS: Number of rows to load for backtest
- N_TESTPOINTS: Size of test window
- WINDOW: Sequence window size for model input
- OFFSET: Row offset to start loading data from
- INITIAL_CAP: Starting capital for backtest
- LEVERAGE: Trade leverage
- RISK_PER_TRADE: % of capital risked per trade
- STOP_LOSS_PCT, TAKE_PROFIT_PCT: SL/TP levels (percent)
- QUANTIZER_PATH: Path to saved quantizer object
- USE_PRICE_CHANGES: Use price returns rather than close prices if True
"""

CSV_PATH        = "historical_data/XAUUSD_15m_historical_data.csv"
N_DATAPOINTS    = 5000
N_TESTPOINTS    = 500
WINDOW          = 3
OFFSET          = 90000

INITIAL_CAP     = 10_000.0
LEVERAGE        = 100
RISK_PER_TRADE  = 0.3
STOP_LOSS_PCT   = 1#0.002
TAKE_PROFIT_PCT = 1#0.002

USE_PRICE_CHANGES = False  # match setting used during training

QUANTIZER_PATH  = "quantizer.pkl"

"""
Load a specific window of historical OHLCV data into a DataFrame.
"""
dp = DataProcessor()
df = dp.read_csv(CSV_PATH, n_points=N_DATAPOINTS, offset=OFFSET)
closes        = df["Close"].values
price_changes = dp.get_price_changes()
highs         = df["High"].values
lows          = df["Low"].values

series = price_changes if USE_PRICE_CHANGES else closes

"""
Apply the saved quantizer to map the chosen target series to discrete labels.
Encode labels into one-hot format for model input.
"""
with open(QUANTIZER_PATH, "rb") as f:
    quantizer = pickle.load(f)

labels  = quantizer.transform(series)
one_hot = np.eye(quantizer.get_bits(), dtype=int)[labels]

TRAIN_SIZE = len(labels) - N_TESTPOINTS
start_idx  = TRAIN_SIZE + WINDOW

"""
Simulate a real-time backtest over the test set.
Predict signals bar by bar, execute trades with SL/TP logic.
Update capital and track trade returns.
"""
capital       = INITIAL_CAP
equity_curve  = [capital]
trade_returns = []
signals_list  = []
prev_label    = None

for t in range(start_idx, len(labels)-1):
    X_in   = one_hot[t-WINDOW:t][np.newaxis, :, :]
    y_prob = model.predict(X_in, verbose=0)[0]
    lbl    = int(np.argmax(y_prob))

    if USE_PRICE_CHANGES:
        return_val = quantizer.inverse_transform([lbl])[0]
        signal     = int(np.sign(return_val))
    else:
        if prev_label is None:
            signal = 0
        else:
            signal = +1 if lbl > prev_label else -1 if lbl < prev_label else 0
        prev_label = lbl

    signals_list.append(signal)

    if signal != 0:
        entry = closes[t]
        tp    = entry*(1+TAKE_PROFIT_PCT) if signal>0 else entry*(1-TAKE_PROFIT_PCT)
        sl    = entry*(1-STOP_LOSS_PCT)   if signal>0 else entry*(1+STOP_LOSS_PCT)

        hi, lo, nxt = highs[t+1], lows[t+1], closes[t+1]
        if   signal>0 and hi  >= tp: exit_price = tp
        elif signal<0 and lo  <= tp: exit_price = tp
        elif signal>0 and lo  <= sl: exit_price = sl
        elif signal<0 and hi  >= sl: exit_price = sl
        else:                          exit_price = nxt

        margin   = capital * RISK_PER_TRADE
        notional = margin * LEVERAGE
        units    = notional / entry
        pnl      = units * ((exit_price-entry) if signal>0 else (entry-exit_price))

        capital += pnl
        trade_returns.append(pnl / margin)

    equity_curve.append(capital)

equity_curve  = np.array(equity_curve)
trade_returns = np.array(trade_returns)

"""
Calculate directional accuracy metrics:
- Confusion matrix of predicted vs actual signals
- Classification report
- Cumulative return and max drawdown
"""
test_labels  = labels[TRAIN_SIZE:]
test_series  = series[TRAIN_SIZE:]
if USE_PRICE_CHANGES:
    true_dirs_full = np.sign(test_series)
else:
    true_dirs_full = np.sign(np.diff(test_series, prepend=test_series[0]))
true_dirs    = true_dirs_full[WINDOW:]
pred_dirs       = np.array(signals_list)

true_dirs = true_dirs[:len(pred_dirs)]

cm = confusion_matrix(true_dirs, pred_dirs, labels=[-1,0,1])
print("=== Directional Accuracy ===")
print("Confusion Matrix (-1,0,+1):")
print(cm)
print("\nClassification Report:")
print(classification_report(true_dirs, pred_dirs, labels=[-1,0,1]))

cum_return = equity_curve[-1]/INITIAL_CAP - 1
max_dd     = (equity_curve/np.maximum.accumulate(equity_curve) - 1).min()
print(f"\n=== Performance Summary ===")
print(f"Cumulative Return: {cum_return:.2%}")
print(f"Max Drawdown:     {max_dd:.2%}")




# Plot
fig, axs = plt.subplots(3, 1, figsize=(12, 16), sharex=False)

# Equity Curve
axs[0].plot(equity_curve, label="Equity Curve")
axs[0].set_title("Equity Curve")
axs[0].grid(True)
axs[0].legend()

# Drawdown
drawdown = equity_curve / np.maximum.accumulate(equity_curve) - 1
axs[1].plot(drawdown, color="red", label="Drawdown")
axs[1].set_title("Drawdown Curve")
axs[1].grid(True)
axs[1].legend()

# Trade Return Histogram
axs[2].hist(trade_returns, bins=50, edgecolor='black')
axs[2].set_title("Distribution of Trade Returns")
axs[2].set_xlabel("Trade Return (PnL / Risked Margin)")
axs[2].set_ylabel("Frequency")
axs[2].grid(True)


plt.show()
