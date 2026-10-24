# AI-written (Claude). Shared test setup and simulated market data.
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("channel_id", "1")  # bot.py reads this at import time

STEPS_PER_DAY = 390  # one step per trading minute


def simulate_ohlc(days, annual_vol, overnight_share=0.0, seed=0, start="2023-01-03"):
    """Random-walk prices with a known volatility, sampled each minute and rolled into daily OHLC.

    overnight_share is the fraction of daily variance that happens between close and next open.
    """
    rng = np.random.default_rng(seed)
    daily_sd = annual_vol / np.sqrt(252)
    gap_sd = daily_sd * np.sqrt(overnight_share)
    intraday_sd = daily_sd * np.sqrt(1 - overnight_share) / np.sqrt(STEPS_PER_DAY)

    gaps = rng.normal(0, gap_sd, days)
    steps = rng.normal(0, intraday_sd, (days, STEPS_PER_DAY))
    log_open = np.log(100) + np.cumsum(gaps + np.r_[0, steps.sum(axis=1)[:-1]])
    path = log_open[:, None] + np.cumsum(steps, axis=1)
    path = np.hstack([log_open[:, None], path])

    idx = pd.bdate_range(start, periods=days, tz="America/New_York")
    return pd.DataFrame({
        "Open": np.exp(path[:, 0]),
        "High": np.exp(path.max(axis=1)),
        "Low": np.exp(path.min(axis=1)),
        "Close": np.exp(path[:, -1]),
    }, index=idx)


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "test.db")
