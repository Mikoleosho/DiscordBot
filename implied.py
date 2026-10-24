"""At-the-money implied volatility from the options chain.

AI-written (Claude). See the "Who built what" section of the README.

Picks the listed expiry closest to TARGET_DAYS calendar days out (skipping
anything under MIN_DAYS, which is too noisy), takes the call and put whose
strikes are nearest the spot price, and averages their implied volatilities.
"""

from datetime import date

import yfinance as yf

TARGET_DAYS = 30
MIN_DAYS = 7
MIN_VALID_IV = 0.01  # Yahoo reports ~0 IV for stale quotes outside market hours


def pick_expiry(expiries, today):
    """Listed expiry (YYYY-MM-DD) closest to TARGET_DAYS out, or None."""
    candidates = []
    for e in expiries:
        days = (date.fromisoformat(e) - today).days
        if days >= MIN_DAYS:
            candidates.append((abs(days - TARGET_DAYS), e))
    return min(candidates)[1] if candidates else None


def atm_iv(chain_side, spot):
    """IV of the contract whose strike is nearest spot, or None if unusable."""
    if chain_side is None or chain_side.empty:
        return None
    row = chain_side.iloc[(chain_side["strike"] - spot).abs().argmin()]
    iv = row["impliedVolatility"]
    return float(iv) if iv == iv and iv >= MIN_VALID_IV else None  # iv == iv filters NaN


def implied_vol(ticker, spot, today=None, yf_ticker=None):
    """{'iv': 0.31, 'expiry': '2026-11-20'} or None if the ticker has no usable options."""
    t = yf_ticker or yf.Ticker(ticker)
    expiry = pick_expiry(t.options, today or date.today())
    if expiry is None:
        return None
    chain = t.option_chain(expiry)
    ivs = [v for v in (atm_iv(chain.calls, spot), atm_iv(chain.puts, spot)) if v is not None]
    if not ivs:
        return None
    return {"iv": sum(ivs) / len(ivs), "expiry": expiry}
