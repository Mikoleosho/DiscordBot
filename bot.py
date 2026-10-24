import discord 
import io
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal 
import yfinance as yf 
from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo
import os 
from dotenv import load_dotenv 
from discord import app_commands
from matplotlib.figure import Figure

import forecast  # AI-added module
import implied   # AI-added module
import regime    # AI-added module



import sqlite3 
load_dotenv()
token = os.getenv("Discord_Token")

import asyncio

import logging 
from discord.ext import commands, tasks 
# Set up logging to both console and file
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(filename="discord.log", encoding="utf-8", mode="w")
                    ])
log=logging.getLogger("volbot")

ny_tz = ZoneInfo("America/New_York")
target_time=time(hour=16, minute=15, second=0, tzinfo=ny_tz)  # 4:15 PM Eastern, after the close (AI change: was 2:00 PM)
# Set the channel ID for alerts
channel_id= int(os.getenv("channel_id") or 0)  # AI change: checked with a clear error at startup
ALERT_PERCENTILE=90 # Percentile for alerting
WINDOW=20 # Rolling window size for volatility calculations
GUILD_ID=os.getenv("guild_id")
TRADING_DAYS=252
db_path="bot.db"

#Instantiate the New York Stock Exchange calender 
nyse=mcal.get_calendar("NYSE")
#Initialize the database and let user modify the watchlist
def init_db():
    with sqlite3.connect(db_path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS watchlist (
            guild_id INTEGER, ticker TEXT, PRIMARY KEY (guild_id, ticker))""")
        conn.execute("""CREATE TABLE IF NOT EXISTS alerts (
            guild_id INTEGER, ticker TEXT, day TEXT, PRIMARY KEY (guild_id, ticker, day))""")
    forecast.init_tables(db_path)  # AI-added
    regime.init_tables(db_path)    # AI-added


def add_ticker(guild_id, ticker):
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT OR IGNORE INTO watchlist VALUES (?, ?)", (guild_id, ticker))


def remove_ticker(guild_id, ticker):
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("DELETE FROM watchlist WHERE guild_id=? AND ticker=?", (guild_id, ticker))
        return cur.rowcount > 0


def get_tickers(guild_id):
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT ticker FROM watchlist WHERE guild_id=? ORDER BY ticker", (guild_id,)).fetchall()
    return [r[0] for r in rows]


def already_alerted(guild_id, ticker, day):
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("INSERT OR IGNORE INTO alerts VALUES (?, ?, ?)", (guild_id, ticker, day))
        return cur.rowcount == 0

#Creator the estimator functions 
def close_to_close(df):
    r = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    return r.std() * np.sqrt(TRADING_DAYS)


def parkinson(df):
    hl = np.log(df["High"] / df["Low"])
    return np.sqrt((hl ** 2).mean() / (4 * np.log(2)) * TRADING_DAYS)


def garman_klass(df):
    hl = np.log(df["High"] / df["Low"])
    co = np.log(df["Close"] / df["Open"])
    var = (0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2).mean()
    return np.sqrt(var * TRADING_DAYS)


def yang_zhang(df):
    prev_close = df["Close"].shift(1)
    d = df.iloc[1:]
    overnight = np.log(d["Open"] / prev_close.iloc[1:])
    open_close = np.log(d["Close"] / d["Open"])
    rs = (np.log(d["High"] / d["Close"]) * np.log(d["High"] / d["Open"])
          + np.log(d["Low"] / d["Close"]) * np.log(d["Low"] / d["Open"]))
    n = len(d)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = overnight.var() + k * open_close.var() + (1 - k) * rs.mean()
    return np.sqrt(var * TRADING_DAYS)


ESTIMATORS = {
    "Close-to-Close": close_to_close,
    "Parkinson": parkinson,
    "Garman-Klass": garman_klass,
    "Yang-Zhang": yang_zhang,
}


def fetch_history(ticker):
    # AI change: 3 years of history (GARCH needs more than 1 year to fit well),
    # and today's bar is dropped until the close so estimators only see finished days.
    df = yf.Ticker(ticker).history(period="3y")
    df = drop_unfinished_bar(df)
    if df.empty or len(df) < WINDOW * 3:
        raise ValueError(f"Not enough data for {ticker}.")
    return df


def drop_unfinished_bar(df, now=None):
    """AI-added: during the trading day, yfinance's last daily bar is still forming. Drop it."""
    now = now or datetime.now(ny_tz)
    if not df.empty and df.index[-1].date() == now.date() and now.time() < time(16, 0):
        return df.iloc[:-1]
    return df


def analyze(ticker):
    df_full = fetch_history(ticker)
    df = df_full.tail(TRADING_DAYS + WINDOW + 1)  # AI change: keep the 1-year context below at 1 year
    recent = df.tail(WINDOW + 1)
    current = {name: f(recent) for name, f in ESTIMATORS.items()}

    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    rolling_cc = (log_ret.rolling(WINDOW).std() * np.sqrt(TRADING_DAYS)).dropna()
    rolling_yz = pd.Series(
        [yang_zhang(df.iloc[i - WINDOW:i + 1]) for i in range(WINDOW, len(df))],
        index=df.index[WINDOW:],
    )

    now = rolling_cc.iloc[-1]
    a = {
        "ticker": ticker,
        "price": df["Close"].iloc[-1],
        "as_of": df.index[-1],
        "current": current,
        "rolling_cc": rolling_cc,
        "rolling_yz": rolling_yz,
        "percentile": (rolling_cc <= now).mean() * 100,
        "zscore": (now - rolling_cc.mean()) / rolling_cc.std(),
    }

    # --- AI-added: regime, GARCH forecast + scoring, implied vol ---
    a["regime"] = regime.classify(a["percentile"])
    made_on = a["as_of"].strftime("%Y-%m-%d")
    try:
        a["garch"] = forecast.garch_forecast(df_full)
        forecast.log_forecast(db_path, ticker, made_on, a["garch"]["vol"], current["Close-to-Close"])
    except Exception:
        log.exception("GARCH forecast failed for %s", ticker)
        a["garch"] = None
    forecast.score_pending(db_path, ticker, df_full)
    try:
        a["implied"] = implied.implied_vol(ticker, a["price"])
    except Exception:
        log.exception("Implied vol lookup failed for %s", ticker)
        a["implied"] = None
    return a


def make_chart(a):
    fig = Figure(figsize=(8, 4), dpi=110)
    ax = fig.subplots()
    ax.plot(a["rolling_cc"].index, a["rolling_cc"] * 100, label="Close-to-Close", linewidth=1.6)
    ax.plot(a["rolling_yz"].index, a["rolling_yz"] * 100, label="Yang-Zhang", linewidth=1.2, alpha=0.8)
    line = np.percentile(a["rolling_cc"], ALERT_PERCENTILE) * 100
    ax.axhline(line, linestyle="--", color="red", linewidth=1, label=f"{ALERT_PERCENTILE}th pct")
    if a.get("garch"):  # AI-added: GARCH forecast drawn past the last bar
        start = a["rolling_cc"].index[-1]
        end = start + pd.tseries.offsets.BDay(forecast.FORECAST_HORIZON)
        g = a["garch"]["vol"] * 100
        ax.plot([start, end], [g, g], color="black", linewidth=2.2,
                label=f"GARCH {forecast.FORECAST_HORIZON}-day forecast")
    ax.set_title(f"{a['ticker']} — {WINDOW}-day rolling annualized volatility")
    ax.set_ylabel("Volatility (%)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf


def ordinal(x):
    """AI-added: 21 -> '21st', 12 -> '12th'."""
    n = int(round(x))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


async def build_report(ticker, title_prefix="Volatility report", a=None):
    if a is None:  # AI change: callers that already ran analyze() pass it in instead of fetching twice
        a = await asyncio.to_thread(analyze, ticker)
    buf = await asyncio.to_thread(make_chart, a)
    filename = f"vol_{ticker}.png"

    high = a["percentile"] >= ALERT_PERCENTILE
    embed = discord.Embed(
        title=f"{title_prefix}: {ticker}",
        description=f"Close ${a['price']:.2f} on {a['as_of']:%b %d} · {WINDOW}-day window, annualized",
        color=discord.Color.red() if high else discord.Color.blurple(),
        timestamp=datetime.now(ny_tz),
    )
    for name, value in a["current"].items():
        embed.add_field(name=name, value=f"{value * 100:.2f}%", inline=True)
    embed.add_field(
        name="1-year context",
        value=f"{ordinal(a['percentile'])} percentile · z = {a['zscore']:+.2f}",  # AI change: was always "th"
        inline=False,
    )
    # --- AI-added fields ---
    r = a["regime"]
    embed.add_field(name="Regime", value=f"{regime.REGIME_EMOJI[r]} {r.capitalize()}", inline=True)
    if a.get("garch"):
        g = a["garch"]
        embed.add_field(
            name=f"GARCH {forecast.FORECAST_HORIZON}-day forecast",
            value=f"{g['vol'] * 100:.2f}% · persistence {g['persistence']:.2f}",
            inline=True,
        )
    if a.get("implied"):
        iv = a["implied"]["iv"]
        realized = a["current"]["Close-to-Close"]
        exp = datetime.strptime(a["implied"]["expiry"], "%Y-%m-%d")
        embed.add_field(
            name="Implied vs realized",
            value=f"IV {iv * 100:.2f}% ({exp:%b %d} exp) · spread {(iv - realized) * 100:+.2f} pts",
            inline=False,
        )
    embed.set_image(url=f"attachment://{filename}")
    return a, embed, discord.File(buf, filename=filename)

def market_open_today():
    today = datetime.now(ny_tz).date()
    return not nyse.valid_days(start_date=today, end_date=today).empty


def market_open_now():
    now = datetime.now(ny_tz)
    return market_open_today() and time(9, 30) <= now.time() <= time(16, 0)


class VolatilityReporter(commands.Cog):
    watch=app_commands.Group(name="watch", description="Manage the watchlist for volatility reports", guild_only=True)
    def __init__(self,bot):
        self.bot=bot 
        #start the backgroud trask loop automatically when the cog loads
       
    async def cog_load(self):
        self.daily_report_loop.start()
        self.alert_check.start()

    async def cog_unload(self):
        #cancel the loop if the cog is removed/reloaded
        self.daily_report_loop.cancel()
        self.alert_check.cancel()
    @app_commands.command(name="vol", description="Volatility estimates and chart for a ticker")
    async def vol(self, interaction: discord.Interaction, ticker: str):
        await interaction.response.defer()
        ticker=ticker.upper()
        try:
             _, embed, file = await build_report(ticker)
             await interaction.followup.send(embed=embed, file=file)
        except Exception as e:
            log.exception("vol failed for %s", ticker)
            await interaction.followup.send(f"Error generating report: Couldn't get data for {ticker}")
    @watch.command(name="add", description="Add a ticker to the watchlist")
    async def watch_add(self, interaction: discord.Interaction, ticker: str):
        await interaction.response.defer()
        ticker=ticker.upper()
        try:
            await asyncio.to_thread(fetch_history, ticker)  # Check if the ticker is valid
        except Exception as e:
            log.exception("Failed to add %s to watchlist", ticker)
            await interaction.followup.send(f"Error adding to watchlist: {e}")
            return
        add_ticker(interaction.guild.id, ticker)
        await interaction.followup.send(f"Added {ticker} to the watchlist.")

    @watch.command(name="remove", description="Remove a ticker from the watchlist")
    async def watch_remove(self, interaction: discord.Interaction, ticker: str):
        await interaction.response.defer()
        ticker=ticker.upper()
        if remove_ticker(interaction.guild.id, ticker):
            regime.forget(db_path, interaction.guild.id, ticker)  # AI-added
            await interaction.followup.send(f"Removed {ticker} from the watchlist.")
        else:
            await interaction.followup.send(f"{ticker} is not in the watchlist.")
            
    @watch.command(name="list", description="List all tickers in the watchlist")
    async def watch_list(self, interaction: discord.Interaction):
        await interaction.response.defer()
        tickers = get_tickers(interaction.guild.id)
        await interaction.followup.send(f"Watchlist: {', '.join(tickers) if tickers else 'No tickers in the watchlist.'}")

    # AI-added command
    @app_commands.command(name="accuracy", description="How GARCH forecasts have scored against a naive benchmark")
    async def accuracy(self, interaction: discord.Interaction, ticker: Optional[str] = None):
        ticker = ticker.upper() if ticker else None
        stats = forecast.accuracy(db_path, ticker)
        scope = ticker or "all tickers"
        if stats is None:
            await interaction.response.send_message(
                f"No scored forecasts for {scope} yet. Each forecast is scored "
                f"{forecast.FORECAST_HORIZON} trading days after it's made."
            )
            return
        better = stats["garch_rmse"] < stats["naive_rmse"]
        embed = discord.Embed(
            title=f"Forecast accuracy: {scope}",
            description=(f"{stats['count']} scored {forecast.FORECAST_HORIZON}-day forecasts. "
                         f"Benchmark = today's {WINDOW}-day realized vol."),
            color=discord.Color.green() if better else discord.Color.orange(),
        )
        embed.add_field(name="RMSE (GARCH / naive)",
                        value=f"{stats['garch_rmse'] * 100:.2f} / {stats['naive_rmse'] * 100:.2f} pts")
        embed.add_field(name="MAE (GARCH / naive)",
                        value=f"{stats['garch_mae'] * 100:.2f} / {stats['naive_mae'] * 100:.2f} pts")
        embed.add_field(name="GARCH closer", value=f"{stats['garch_win_rate'] * 100:.0f}% of the time")
        await interaction.response.send_message(embed=embed)

    @tasks.loop(time=target_time)
    async def daily_report_loop(self):
        today=datetime.now(ny_tz).date() 
        if nyse.valid_days(start_date=today, end_date=today).empty:
            return  # Skip the report if today is not a trading day
        channel=self.bot.get_channel(channel_id)
        if not channel:
            log.error("Channel with ID %s not found.", channel_id)
            return
        for ticker in get_tickers(channel.guild.id):  # AI change: was GUILD_ID from .env
            try:
                _, embed, file = await build_report(ticker, title_prefix="Daily Volatility Report")
                await channel.send(embed=embed, file=file)
            except Exception as e:
                log.exception("Failed to send daily report for %s", ticker)
                await channel.send(f"Error generating report for {ticker}: {e}")
    
   

    @tasks.loop(minutes=30)
    async def alert_check(self):
        if not market_open_now():
            return  # Skip if the market is not open
        channel = self.bot.get_channel(channel_id)
        if not channel:
            log.error("Channel with ID %s not found.", channel_id)
            return
        today=datetime.now(ny_tz).date().isoformat()
        for ticker in get_tickers(channel.guild.id):
            try:
                a = await asyncio.to_thread(analyze, ticker)
                # AI-added: regime-change alert
                previous = regime.update(db_path, channel.guild.id, ticker, a["regime"], today)
                if previous:
                    _, embed, file = await build_report(ticker, "Regime change", a=a)
                    await channel.send(
                        f"**{ticker}** moved from {regime.REGIME_EMOJI[previous]} {previous} "
                        f"to {regime.REGIME_EMOJI[a['regime']]} {a['regime']} volatility.",
                        embed=embed, file=file,
                    )
                if a["percentile"] < ALERT_PERCENTILE:
                    continue
                if already_alerted(channel.guild.id, ticker, today):
                    continue
                _, embed, file = await build_report(ticker, "⚠️ Volatility alert", a=a)  # AI change: reuse a
                await channel.send(
                    f"**{ticker}** volatility is in the top {100 - ALERT_PERCENTILE}% of the past year.",
                    embed=embed, file=file,
                )
            except Exception:
                log.exception("Alert check failed for %s", ticker)
    @daily_report_loop.before_loop
    @alert_check.before_loop
    async def before_loop(self):
        await self.bot.wait_until_ready()
        logging.info("Volatility report loop is starting...")

# Global Gateway Handlers (Moved outside the class hierarchy)

class VolBot(commands.Bot):
    async def setup_hook(self):
        init_db()
        await self.add_cog(VolatilityReporter(self))
        if GUILD_ID:
            guild = discord.Object(id=int(GUILD_ID))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()

    async def on_ready(self):
        log.info("Logged in as %s (%s)", self.user, self.user.id)
bot = VolBot(command_prefix="!", intents=discord.Intents.default())

if __name__ == "__main__":
    # AI-added: fail fast with a clear message instead of a cryptic crash
    if not token or not channel_id:
        raise SystemExit("Set Discord_Token and channel_id in .env (see .env.example).")
    bot.run(token, log_handler=None)
