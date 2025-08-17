import os
import sys
import numpy as np
import pandas as pd
import ccxt.async_support as ccxt_async
from scipy import stats
from statsmodels.tsa.stattools import coint
from telegram import Bot
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

async def initialize_bybit_async(self):
        """Initialize async Bybit exchange connection"""
        api_key = os.environ.get('BYBIT_API_KEY')
        api_secret = os.environ.get('BYBIT_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("Please set BYBIT_API_KEY and BYBIT_SECRET environment variables")
        
        exchange = ccxt_async.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'recvWindow': 10000,
                'adjustForTimeDifference': True
            }
        })
        
        await exchange.load_time_difference()
        await exchange.load_markets()
        
        # check deposit
        try:
            balance = await exchange.fetch_balance()
            usdt_balance = balance['USDT']['free'] if 'USDT' in balance else 0
            print(f"[ACCOUNT] Available USDT Balance: ${usdt_balance:,.2f}")
            
            # exit if not enough deposit
            if usdt_balance < 1000:
                print(f"[ERROR] Insufficient balance: ${usdt_balance:.2f} < $1000 minimum")
                print("[SHUTDOWN] Bot terminated due to insufficient balance")
                await exchange.close()
                sys.exit(1)
                
        except Exception as e:
            print(f"[WARNING] Could not check balance: {e}")
        
        return exchange