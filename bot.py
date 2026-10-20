import discord 
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal 
import yfinance as yf 
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo
import threading 
import os 
from dotenv import load_dotenv 

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
intents=discord.Intents.default()
intents.message_content=True
intents.members=True
ny_tz = ZoneInfo("America/New_York")
target_time=time(hour=14, minute=0, second=0, tzinfo=ny_tz)  # Set the target time to 2:00 PM Eastern Time
channel_id= int(os.getenv("channel_id"))
alert_percentile=90
window=20
guild_id=os.getenv("guild_id")
trading_days=252
db_path="bot.db"
ticker_symbol="NVDA" 
#Instantiate the bot 
bot=commands.Bot(command_prefix="!", intents=intents)
#Instantiate the New York Stock Exchange calender 
nyse=mcal.get_calendar("NYSE")

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

def calculate_volatility(ticker: str, window: int=30)-> float:
    # This fetches historical data and calulates the annualized historical volatility
    
    stock=yf.Ticker(ticker)
    df=stock.history(period=f"3mo")
    # Check if the DataFrame is empty or has insufficient data
    if df.empty or len(df)<window:
        raise ValueError(f"Not enough data to calculate volatility for {ticker}.")
    close_prices=df['Close'].tail(window)

    # Calculate the log returns
    log_returns=np.log(close_prices / close_prices.shift(1)).dropna()

    # Calculate standard deviation of daily log returns
    
    daily_volatility=log_returns.std()
    ## Annualize the volatility, we use 252 because there's typically 252 trading days in a year
    annualized_volatility=daily_volatility * np.sqrt(252)
    return float(annualized_volatility * 100) 



class VolatilityReporter(commands.Cog):
    def __init__(self,bot):
        self.bot=bot 
        #start the backgroud trask loop automatically when the cog loads
        self.daily_report_loop.start()
        
    def cog_unload(self):
        #cancel the loop if the cog is removed/reloaded
        self.daily_report_loop.cancel()
    @tasks.loop(time=target_time)
    async def daily_report_loop(self):
        await self.bot.wait_until_ready()
        today=datetime.now(ny_tz).date() 
        if nyse.valid_days(start_date=today, end_date=today).empty:
            return  # Skip the report if today is not a trading day
        
        if datetime.now(ny_tz).weekday()>=5:  # Check if today is Saturday or Sunday
            return  # Skip the report on weekends
        channel=self.bot.get_channel(channel_id)
        if channel:
            try:
                volatility = await asyncio.to_thread(calculate_volatility, ticker_symbol)
                await channel.send(f"Daily volatility report for {ticker_symbol}: {volatility:.2f}%")
            except Exception as e:
                logging.error(f"Error fetching volatility for {ticker_symbol}: {e}")
                await channel.send(f"Error fetching volatility data: {e}")
                
    @daily_report_loop.before_loop
    async def before_daily_report_loop(self):
        await self.bot.wait_until_ready()
        logging.info("Volatility report loop is starting...")
    
    
# Global Gateway Handlers (Moved outside the class hierarchy)
@bot.event 
async def on_ready():
    logging.info(f"Logged in as {bot.user.name} - {bot.user.id}")
    print("-------")

# Asynchronous runner pattern to properly load the cog before startup
async def main():
    async with bot:
        await bot.add_cog(VolatilityReporter(bot))
        await bot.start(token)

if __name__ == "__main__":
    asyncio.run(main())