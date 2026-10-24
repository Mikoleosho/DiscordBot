# AI-written (Claude).
from datetime import date
from types import SimpleNamespace

import pandas as pd

import implied
import regime


def test_classify_boundaries():
    assert regime.classify(10) == "low"
    assert regime.classify(25) == "normal"
    assert regime.classify(74.9) == "normal"
    assert regime.classify(75) == "high"


def test_regime_update_reports_only_changes(db):
    regime.init_tables(db)
    assert regime.update(db, 1, "AAPL", "normal", "2026-10-01") is None  # first sighting
    assert regime.update(db, 1, "AAPL", "normal", "2026-10-02") is None  # unchanged
    assert regime.update(db, 1, "AAPL", "high", "2026-10-03") == "normal"
    assert regime.update(db, 2, "AAPL", "low", "2026-10-03") is None  # other servers are separate


def test_pick_expiry_prefers_about_a_month_out():
    today = date(2026, 10, 24)
    expiries = ["2026-10-27", "2026-11-06", "2026-11-20", "2027-01-15"]
    assert implied.pick_expiry(expiries, today) == "2026-11-20"
    assert implied.pick_expiry(["2026-10-27"], today) is None  # too close


def chain(strikes, ivs):
    return pd.DataFrame({"strike": strikes, "impliedVolatility": ivs})


def test_implied_vol_averages_atm_call_and_put():
    fake = SimpleNamespace(
        options=["2026-11-20"],
        option_chain=lambda exp: SimpleNamespace(
            calls=chain([95, 100, 105], [0.40, 0.30, 0.25]),
            puts=chain([95, 100, 105], [0.45, 0.34, 0.28]),
        ),
    )
    result = implied.implied_vol("X", spot=101, today=date(2026, 10, 24), yf_ticker=fake)
    assert result == {"iv": 0.32, "expiry": "2026-11-20"}


def test_implied_vol_skips_stale_quotes():
    fake = SimpleNamespace(
        options=["2026-11-20"],
        option_chain=lambda exp: SimpleNamespace(calls=chain([100], [0.00001]), puts=chain([100], [0.0])),
    )
    assert implied.implied_vol("X", spot=100, today=date(2026, 10, 24), yf_ticker=fake) is None
