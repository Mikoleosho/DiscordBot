import discord 
import pandas as pd
import numpy as np
import yfinance as yf 
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo
import threading 
import os 
from dotenv import load_dotenv 
load_dotenv()
token = os.getenv("Discord_Token")

import asyncio

import logging 
from discord.ext import commands, tasks 

logging.basicConfig(level=logging.INFO)

handler=logging.FileHandler(filename="discord.log", encoding="utf-8", mode="w")
intents=discord.Intents.default()
intents.message_content=True
intents.members=True
ny_tz = ZoneInfo("America/New_York")
target_time=time(hour=15, minute=14, second=6, tzinfo=ny_tz)  # Set the target time to 3:05 PM Eastern Time
channel_id= 1561431738708926517
ticker_symbol="NVDA" 
#Instantiate the bot 
bot=commands.Bot(command_prefix="!", intents=intents)




def calculate_volatility(ticker: str, window: int=30)-> float:
    # This fetches historical data and calulates the annualized historical volatility
    
    stock=yf.Ticker(ticker)
    df=stock.history(period=f"3mo")
    
    if df.empty or len(df)<window:
        raise ValueError(f"Not enough data to calculate volatility for {ticker}.")
    close_prices=df['Close'].tail(window)

    # Calculate the log returns
    log_returns=np.log(close_prices / close_prices.shift(1)).dropna()

    # Calculate standard deviation of daily log returns
    
    daily_volatility=log_returns.std()

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
        channel=self.bot.get_channel(channel_id)
        if channel:
            try:
                volatility=calculate_volatility(ticker_symbol)
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