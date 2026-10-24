# AI-written (Claude). End-to-end check of a /vol report with Yahoo replaced by simulated data.
import asyncio

import bot
import implied
from conftest import simulate_ohlc


def test_full_report_builds(db, monkeypatch):
    monkeypatch.setattr(bot, "db_path", db)
    monkeypatch.setattr(bot, "fetch_history", lambda t: simulate_ohlc(750, 0.30, seed=7))
    monkeypatch.setattr(implied, "implied_vol", lambda t, spot: {"iv": 0.35, "expiry": "2026-11-20"})
    bot.init_db()

    a, embed, file = asyncio.run(bot.build_report("TEST"))

    names = [f.name for f in embed.fields]
    for expected in ["Yang-Zhang", "1-year context", "Regime", "GARCH 5-day forecast", "Implied vs realized"]:
        assert expected in names
    assert len(a["rolling_cc"]) <= bot.TRADING_DAYS + 1  # context stays at 1 year with 3y fetched
    assert file.fp.read(8) == b"\x89PNG\r\n\x1a\n"
