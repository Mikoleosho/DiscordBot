import discord 
import io
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal 
import yfinance as yf 
from datetime import datetime, time
from zoneinfo import ZoneInfo
import os 
from dotenv import load_dotenv 
from discord import app_commands
from matplotlib.figure import Figure



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
target_time=time(hour=14, minute=0, second=0, tzinfo=ny_tz)  # Set the target time to 2:00 PM Eastern Time
# Set the channel ID for alerts
channel_id= int(os.getenv("channel_id"))
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
    df = yf.Ticker(ticker).history(period="1y")
    if df.empty or len(df) < WINDOW * 3:
        raise ValueError(f"Not enough data for {ticker}.")
    return df


def analyze(ticker):
    df = fetch_history(ticker)
    recent = df.tail(WINDOW + 1)
    current = {name: f(recent) for name, f in ESTIMATORS.items()}

    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    rolling_cc = (log_ret.rolling(WINDOW).std() * np.sqrt(TRADING_DAYS)).dropna()
    rolling_yz = pd.Series(
        [yang_zhang(df.iloc[i - WINDOW:i + 1]) for i in range(WINDOW, len(df))],
        index=df.index[WINDOW:],
    )

    now = rolling_cc.iloc[-1]
    return {
        "ticker": ticker,
        "price": df["Close"].iloc[-1],
        "current": current,
        "rolling_cc": rolling_cc,
        "rolling_yz": rolling_yz,
        "percentile": (rolling_cc <= now).mean() * 100,
        "zscore": (now - rolling_cc.mean()) / rolling_cc.std(),
    }


def make_chart(a):
    fig = Figure(figsize=(8, 4), dpi=110)
    ax = fig.subplots()
    ax.plot(a["rolling_cc"].index, a["rolling_cc"] * 100, label="Close-to-Close", linewidth=1.6)
    ax.plot(a["rolling_yz"].index, a["rolling_yz"] * 100, label="Yang-Zhang", linewidth=1.2, alpha=0.8)
    line = np.percentile(a["rolling_cc"], ALERT_PERCENTILE) * 100
    ax.axhline(line, linestyle="--", color="red", linewidth=1, label=f"{ALERT_PERCENTILE}th pct")
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


async def build_report(ticker, title_prefix="Volatility report"):
    a = await asyncio.to_thread(analyze, ticker)
    buf = await asyncio.to_thread(make_chart, a)
    filename = f"vol_{ticker}.png"

    high = a["percentile"] >= ALERT_PERCENTILE
    embed = discord.Embed(
        title=f"{title_prefix}: {ticker}",
        description=f"Last close ${a['price']:.2f} · {WINDOW}-day window, annualized",
        color=discord.Color.red() if high else discord.Color.blurple(),
        timestamp=datetime.now(ny_tz),
    )
    for name, value in a["current"].items():
        embed.add_field(name=name, value=f"{value * 100:.2f}%", inline=True)
    embed.add_field(
        name="1-year context",
        value=f"{a['percentile']:.0f}th percentile · z = {a['zscore']:+.2f}",
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
            await interaction.followup.send(f"Removed {ticker} from the watchlist.")
        else:
            await interaction.followup.send(f"{ticker} is not in the watchlist.")
            
    @watch.command(name="list", description="List all tickers in the watchlist")
    async def watch_list(self, interaction: discord.Interaction):
        await interaction.response.defer()
        tickers = get_tickers(interaction.guild.id)
        await interaction.followup.send(f"Watchlist: {', '.join(tickers) if tickers else 'No tickers in the watchlist.'}")

    @tasks.loop(time=target_time)
    async def daily_report_loop(self):
        today=datetime.now(ny_tz).date() 
        if nyse.valid_days(start_date=today, end_date=today).empty:
            return  # Skip the report if today is not a trading day
        channel=self.bot.get_channel(channel_id)
        if not channel:
            log.error("Channel with ID %s not found.", channel_id)
            return
        for ticker in get_tickers(GUILD_ID):
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
                if a["percentile"] < ALERT_PERCENTILE:
                    continue
                if already_alerted(channel.guild.id, ticker, today):
                    continue
                _, embed, file = await build_report(ticker, "⚠️ Volatility alert")
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
    bot.run(token, log_handler=None)
