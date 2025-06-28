# backtest_sweep.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping

from data_processing import DataProcessor
from quantization import Quantization
from x_y_arrays import x_y_arrays
from model import create_model

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
CSV_PATH        = "historical_data/XAUUSD_15m_historical_data.csv"
N_TESTPOINTS    = 500
INITIAL_OFFSET  = 80000
INITIAL_N_TOTAL = 1000   # total points (train + test) at first iteration
MAX_N_TOTAL     = 5500
STEP            = 50
WINDOW          = 3

# Backtest parameters
USE_PRICE_CHANGES = False
LEVERAGE          = 100
RISK_PER_TRADE    = 0.3
STOP_LOSS_PCT     = 0.0015    # 100%
TAKE_PROFIT_PCT   = 0.0015
INITIAL_CAP       = 10000.0

# Training parameters
BIN_SIZE    = 6
TEST_SPLIT  = 0.2
EPOCHS      = 100
BATCH_SIZE  = 64
PATIENCE    = 10

# ─── SWEEP LOOP ─────────────────────────────────────────────────────────────────
results = []   # will hold (training_size, final_equity)

for N_TOTAL in range(INITIAL_N_TOTAL, MAX_N_TOTAL + 1, STEP):
    offset = INITIAL_OFFSET - (N_TOTAL - INITIAL_N_TOTAL)
    print(f"\n>> N_TOTAL={N_TOTAL}, OFFSET={offset}")

    # 1) LOAD DATA
    dp     = DataProcessor()
    df     = dp.read_csv(CSV_PATH, n_points=N_TOTAL, offset=offset)
    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values

    series = dp.get_price_changes() if USE_PRICE_CHANGES else closes

    # 2) QUANTIZE
    TRAIN_SIZE = len(series) - N_TESTPOINTS
    quant      = Quantization(bin_size=BIN_SIZE)
    quant.fit(series[:TRAIN_SIZE])
    labels     = quant.transform(series)
    n_classes  = quant.get_bits()

    # build full one-hot for inference, and train-only one-hot for X/Y prep
    one_hot_full  = np.eye(n_classes, dtype=int)[labels]
    one_hot_train = one_hot_full[:TRAIN_SIZE]

    # 3) PREPARE X/Y
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

    # 4) TRAIN MODEL
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

    # 5) BACKTEST
    capital      = INITIAL_CAP
    equity_curve = [capital]
    prev_label   = None

    for t in range(TRAIN_SIZE + WINDOW, len(labels) - 1):
        X_in = one_hot_full[t-WINDOW:t][np.newaxis, ...]
        yp   = model.predict(X_in, verbose=0)[0]
        lbl  = int(np.argmax(yp))

        # signal logic
        if USE_PRICE_CHANGES:
            ret    = quant.inverse_transform([lbl])[0]
            signal = int(np.sign(ret))
        else:
            if prev_label is None:
                signal = 0
            else:
                signal = 1 if lbl > prev_label else -1 if lbl < prev_label else 0
            prev_label = lbl

        # trade execution
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

    final_equity   = equity_curve[-1]
    training_size  = N_TOTAL - N_TESTPOINTS
    results.append((training_size, final_equity))
    print(f"   -> Train size {training_size}, Final Equity {final_equity:.2f}")

# ─── PLOT ────────────────────────────────────────────────────────────────────────
sizes, equities = zip(*results)

plt.figure(figsize=(10,6))
plt.plot(sizes, equities, marker='o')
plt.xlabel("Training Size")
plt.ylabel("Final Equity")
plt.grid(True)

# x-axis ticks every 500: 500, 1000, …, 5000
plt.xticks(np.arange(500, 5001, 500))

plt.tight_layout()
plt.show()
