# backtest_bin_sweep.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping

from data_processing import DataProcessor
from quantization import Quantization
from x_y_arrays import x_y_arrays
from model import create_model

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
CSV_PATH       = "historical_data/XAUUSD_15m_historical_data.csv"
N_TOTAL        = 5000      # total points (train + test)
N_TESTPOINTS   = 500
OFFSET         = 60000
WINDOW         = 3
INITIAL_CAP    = 10000.0

USE_PRICE_CHANGES = False
LEVERAGE          = 100
RISK_PER_TRADE    = 0.3
STOP_LOSS_PCT     = 1.0
TAKE_PROFIT_PCT   = 1.0

# Sweep over these bin sizes (in price units)
BIN_SIZES     = list(np.arange(0.05, 11, 0.05))  # 1,2,…,10
TEST_SPLIT    = 0.2
EPOCHS        = 100
BATCH_SIZE    = 64
PATIENCE      = 10

# ─── LOAD DATA ONCE ─────────────────────────────────────────────────────────────
dp    = DataProcessor()
df    = dp.read_csv(CSV_PATH, n_points=N_TOTAL, offset=OFFSET)
closes = df["Close"].values
highs  = df["High"].values
lows   = df["Low"].values

series = dp.get_price_changes() if USE_PRICE_CHANGES else closes
TRAIN_SIZE = len(series) - N_TESTPOINTS

results = []  # (bin_size, final_equity)

for bin_size in BIN_SIZES:
    print(f"\n>> Running for BIN_SIZE={bin_size}")

    # 1) Quantize
    quant = Quantization(bin_size=bin_size)
    quant.fit(series[:TRAIN_SIZE])
    labels    = quant.transform(series)
    n_classes = quant.get_bits()

    # 2) Build one-hot arrays
    one_hot_full  = np.eye(n_classes, dtype=int)[labels]
    one_hot_train = one_hot_full[:TRAIN_SIZE]

    # 3) Prepare X/Y
    xy = x_y_arrays(
        df=one_hot_train,
        target=labels[:TRAIN_SIZE],
        n=WINDOW,
        test_size=TEST_SPLIT,
        shuffle=True,
        random_state=42,
        num_classes=n_classes
    )
    X_train, X_val, Y_train, Y_val = xy.get_train_test()

    # 4) Train
    model = create_model(input_timesteps=WINDOW, n_classes=n_classes)
    early = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early],
        verbose=0
    )

    # 5) Backtest
    capital      = INITIAL_CAP
    equity_curve = [capital]
    prev_label   = None

    for t in range(TRAIN_SIZE + WINDOW, len(labels) - 1):
        X_in = one_hot_full[t-WINDOW:t][np.newaxis, ...]
        yp   = model.predict(X_in, verbose=0)[0]
        lbl  = int(np.argmax(yp))

        # signal
        if USE_PRICE_CHANGES:
            ret    = quant.inverse_transform([lbl])[0]
            signal = int(np.sign(ret))
        else:
            signal = 0 if prev_label is None else (1 if lbl>prev_label else -1 if lbl<prev_label else 0)
            prev_label = lbl

        # trade
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

        equity_curve.append(capital)

    final_equity = equity_curve[-1]
    results.append((bin_size, final_equity))
    print(f"   -> BIN_SIZE {bin_size}, Final Equity {final_equity:.2f}")

# ─── PLOT RESULTS ───────────────────────────────────────────────────────────────
bins, equities = zip(*results)
plt.figure(figsize=(10,6))
plt.plot(bins, equities, marker='o')
plt.xlabel("Bin Size")
plt.ylabel("Final Equity")
plt.title("Final Equity vs. Quantization Bin Size")
plt.grid(True)
plt.xticks(bins)
plt.tight_layout()
plt.show()
