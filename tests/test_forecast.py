# AI-written (Claude).
import numpy as np
import pandas as pd
import pytest

import forecast
from conftest import simulate_ohlc


def test_garch_matches_constant_vol():
    df = simulate_ohlc(750, 0.25, seed=3)
    result = forecast.garch_forecast(df)
    assert result["vol"] == pytest.approx(0.25, rel=0.15)


def test_garch_reacts_to_a_vol_spike():
    calm = simulate_ohlc(700, 0.15, seed=4)
    stormy = simulate_ohlc(20, 0.60, seed=5, start=calm.index[-1].strftime("%Y-%m-%d"))
    stormy = stormy.iloc[1:] * (calm["Close"].iloc[-1] / stormy["Open"].iloc[1])
    df = pd.concat([calm, stormy])
    assert forecast.garch_forecast(df)["vol"] > 0.20


def make_returns_df(returns, start="2026-01-02"):
    close = 100 * np.exp(np.cumsum(np.r_[0, returns]))
    idx = pd.bdate_range(start, periods=len(close), tz="America/New_York")
    return pd.DataFrame({"Close": close}, index=idx)


def test_scoring_waits_for_horizon_then_scores(db):
    forecast.init_tables(db)
    df = make_returns_df([0.0] * 10 + [0.01, -0.01, 0.01, -0.01, 0.01])
    made_on = df.index[10].strftime("%Y-%m-%d")
    forecast.log_forecast(db, "TEST", made_on, garch_vol=0.15, naive_vol=0.30, horizon=5)

    assert forecast.score_pending(db, "TEST", df.iloc[:13]) == 0  # only 2 days after made_on
    assert forecast.accuracy(db) is None

    assert forecast.score_pending(db, "TEST", df) == 1
    realized = 0.01 * np.sqrt(252)  # every return is ±1%
    stats = forecast.accuracy(db, "TEST")
    assert stats["count"] == 1
    assert stats["garch_rmse"] == pytest.approx(abs(0.15 - realized))
    assert stats["naive_rmse"] == pytest.approx(abs(0.30 - realized))
    assert stats["garch_win_rate"] == 1.0


def test_duplicate_forecasts_ignored(db):
    forecast.init_tables(db)
    forecast.log_forecast(db, "TEST", "2026-01-02", 0.2, 0.2)
    forecast.log_forecast(db, "TEST", "2026-01-02", 0.9, 0.9)
    import sqlite3
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*), MAX(garch_vol) FROM forecasts").fetchone() == (1, 0.2)
