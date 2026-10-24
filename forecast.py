"""GARCH(1,1) volatility forecasts and forecast scoring.

AI-written (Claude). See the "Who built what" section of the README.

Every report logs two forecasts of the next FORECAST_HORIZON trading days'
annualized volatility:
  * GARCH(1,1), fit on up to 3 years of daily log returns
  * a naive benchmark: today's 20-day close-to-close volatility
Once those days have actually happened, each forecast is scored against the
realized volatility, so /accuracy can show whether GARCH beats the benchmark.
"""

import sqlite3
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

TRADING_DAYS = 252
FORECAST_HORIZON = 5  # trading days ahead


def init_tables(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS forecasts (
            ticker TEXT,
            made_on TEXT,          -- date of the last completed bar used
            horizon INTEGER,
            garch_vol REAL,        -- annualized, as a fraction (0.25 = 25%)
            naive_vol REAL,
            realized_vol REAL,     -- filled in once the horizon has passed
            PRIMARY KEY (ticker, made_on, horizon))""")


def log_returns(df):
    return np.log(df["Close"] / df["Close"].shift(1)).dropna()


def garch_forecast(df, horizon=FORECAST_HORIZON):
    """Average annualized volatility GARCH(1,1) expects over the next `horizon` days."""
    r = log_returns(df) * 100  # percent returns help the optimizer converge
    model = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t")  # Student-t: expects occasional extreme days
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # arch warns on slow convergence; the fit is still usable
        res = model.fit(disp="off")
    daily_var = res.forecast(horizon=horizon, reindex=False).variance.values[-1]
    params = res.params
    return {
        "vol": float(np.sqrt(daily_var.mean() * TRADING_DAYS) / 100),
        "persistence": float(params["alpha[1]"] + params["beta[1]"]),
    }


def realized_vol(returns):
    """Annualized realized volatility from daily log returns (zero-mean, standard for scoring)."""
    return float(np.sqrt((returns ** 2).mean() * TRADING_DAYS))


def log_forecast(db_path, ticker, made_on, garch_vol, naive_vol, horizon=FORECAST_HORIZON):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO forecasts (ticker, made_on, horizon, garch_vol, naive_vol) "
            "VALUES (?, ?, ?, ?, ?)",
            (ticker, made_on, horizon, garch_vol, naive_vol),
        )


def score_pending(db_path, ticker, df):
    """Fill in realized volatility for this ticker's forecasts whose horizon has passed."""
    r = log_returns(df)
    dates = pd.Index(r.index.strftime("%Y-%m-%d"))
    scored = 0
    with sqlite3.connect(db_path) as conn:
        pending = conn.execute(
            "SELECT made_on, horizon FROM forecasts WHERE ticker=? AND realized_vol IS NULL",
            (ticker,),
        ).fetchall()
        for made_on, horizon in pending:
            after = r[dates > made_on]
            if len(after) < horizon:
                continue  # not enough days have passed yet
            conn.execute(
                "UPDATE forecasts SET realized_vol=? WHERE ticker=? AND made_on=? AND horizon=?",
                (realized_vol(after.iloc[:horizon]), ticker, made_on, horizon),
            )
            scored += 1
    return scored


def accuracy(db_path, ticker=None):
    """Error stats for scored forecasts, GARCH vs. the naive benchmark. None if nothing is scored yet."""
    query = "SELECT garch_vol, naive_vol, realized_vol FROM forecasts WHERE realized_vol IS NOT NULL"
    params = ()
    if ticker:
        query += " AND ticker=?"
        params = (ticker,)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    if not rows:
        return None
    g, n, real = (np.array(col) for col in zip(*rows))
    return {
        "count": len(rows),
        "garch_rmse": float(np.sqrt(np.mean((g - real) ** 2))),
        "naive_rmse": float(np.sqrt(np.mean((n - real) ** 2))),
        "garch_mae": float(np.mean(np.abs(g - real))),
        "naive_mae": float(np.mean(np.abs(n - real))),
        "garch_win_rate": float(np.mean(np.abs(g - real) < np.abs(n - real))),
    }
