"""Volatility regimes and regime-change tracking.

AI-written (Claude). See the "Who built what" section of the README.

A ticker's regime comes from where today's 20-day volatility sits in its own
past year: low (bottom quarter), normal, or high (top quarter). The bot alerts
when a watched ticker moves from one regime to another.
"""

import sqlite3

LOW_PCT = 25
HIGH_PCT = 75

REGIME_EMOJI = {"low": "🟢", "normal": "🟡", "high": "🔴"}


def classify(percentile):
    if percentile < LOW_PCT:
        return "low"
    if percentile >= HIGH_PCT:
        return "high"
    return "normal"


def init_tables(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS regimes (
            guild_id INTEGER, ticker TEXT, regime TEXT, since TEXT,
            PRIMARY KEY (guild_id, ticker))""")


def update(db_path, guild_id, ticker, regime, day):
    """Store the current regime. Returns the previous regime if it changed, else None.

    The first time a ticker is seen there is nothing to compare against, so no change is reported.
    """
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT regime FROM regimes WHERE guild_id=? AND ticker=?", (guild_id, ticker)
        ).fetchone()
        if row and row[0] == regime:
            return None
        conn.execute(
            "INSERT OR REPLACE INTO regimes VALUES (?, ?, ?, ?)", (guild_id, ticker, regime, day)
        )
    return row[0] if row else None


def forget(db_path, guild_id, ticker):
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM regimes WHERE guild_id=? AND ticker=?", (guild_id, ticker))
