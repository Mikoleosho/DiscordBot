# AI-written (Claude). Checks Michael's four estimators recover a known volatility.
from datetime import datetime

import pytest

import bot
from conftest import simulate_ohlc

TRUE_VOL = 0.30


@pytest.mark.parametrize("name", list(bot.ESTIMATORS))
def test_estimators_recover_known_vol(name):
    df = simulate_ohlc(1500, TRUE_VOL, seed=1)
    assert bot.ESTIMATORS[name](df) == pytest.approx(TRUE_VOL, rel=0.10)


def test_yang_zhang_handles_overnight_gaps():
    # A third of the variance happens overnight; Yang-Zhang and close-to-close should still see all of it.
    df = simulate_ohlc(1500, TRUE_VOL, overnight_share=1 / 3, seed=2)
    assert bot.yang_zhang(df) == pytest.approx(TRUE_VOL, rel=0.10)
    assert bot.close_to_close(df) == pytest.approx(TRUE_VOL, rel=0.10)
    # Parkinson only sees the intraday range, so it should miss the overnight part.
    assert bot.parkinson(df) < 0.9 * TRUE_VOL


def test_unfinished_bar_dropped_during_trading_day():
    df = simulate_ohlc(10, TRUE_VOL)
    last_day = df.index[-1]
    midday = datetime(last_day.year, last_day.month, last_day.day, 13, 0, tzinfo=bot.ny_tz)
    after_close = midday.replace(hour=16, minute=15)
    assert len(bot.drop_unfinished_bar(df, now=midday)) == 9
    assert len(bot.drop_unfinished_bar(df, now=after_close)) == 10


def test_ordinal_suffixes():
    assert [bot.ordinal(x) for x in (1, 2, 3, 11, 12, 13, 21, 22, 90.4, 100)] == \
        ["1st", "2nd", "3rd", "11th", "12th", "13th", "21st", "22nd", "90th", "100th"]
