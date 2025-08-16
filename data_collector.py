import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from telegram import Bot
from typing import List, Dict, Tuple
from scipy import stats
from statsmodels.tsa.stattools import coint
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import requests
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from concurrent.futures import ThreadPoolExecutor

class FundamentalDataCollector:
    """펀더멘털 데이터 수집 클래스 (CoinGecko API)"""
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        
    def get_top_coins_by_market_cap(self, limit=200):
        """시가총액 상위 코인 목록 가져오기"""
        try:
            endpoint = f"{self.base_url}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': limit,
                'page': 1,
                'sparkline': False
            }
            
            response = self.session.get(endpoint, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching market data: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error in get_top_coins_by_market_cap: {e}")
            return []
    
    def get_coin_fundamentals(self, coin_id):
        """특정 코인의 상세 펀더멘털 데이터"""
        try:
            endpoint = f"{self.base_url}/coins/{coin_id}"
            params = {
                'localization': 'false',
                'tickers': False,
                'market_data': True,
                'community_data': False,
                'developer_data': False
            }
            
            response = self.session.get(endpoint, params=params)
            if response.status_code == 200:
                data = response.json()
                market_data = data.get('market_data', {})
                
                return {
                    'symbol': data.get('symbol', '').upper(),
                    'name': data.get('name', ''),
                    'market_cap': market_data.get('market_cap', {}).get('usd', 0),
                    'market_cap_rank': data.get('market_cap_rank', 0),
                    'fully_diluted_valuation': market_data.get('fully_diluted_valuation', {}).get('usd', 0),
                    'circulating_supply': market_data.get('circulating_supply', 0),
                    'total_supply': market_data.get('total_supply', 0),
                    'max_supply': market_data.get('max_supply', 0),
                    'circulation_ratio': self.calculate_circulation_ratio(
                        market_data.get('circulating_supply', 0),
                        market_data.get('total_supply', 0)
                    ),
                    'price_change_24h': market_data.get('price_change_percentage_24h', 0),
                    'price_change_7d': market_data.get('price_change_percentage_7d', 0),
                    'price_change_30d': market_data.get('price_change_percentage_30d', 0),
                    'ath': market_data.get('ath', {}).get('usd', 0),
                    'ath_change_percentage': market_data.get('ath_change_percentage', {}).get('usd', 0),
                    'volume_24h': market_data.get('total_volume', {}).get('usd', 0)
                }
            else:
                return None
        except Exception as e:
            print(f"Error fetching fundamentals for {coin_id}: {e}")
            return None
    
    def calculate_circulation_ratio(self, circulating, total):
        """유통률 계산"""
        if total and total > 0:
            return (circulating / total) * 100
        return 0
    
    def get_coin_categories(self, coin_id):
        """CoinGecko API에서 코인 카테고리 가져오기"""
        try:
            endpoint = f"{self.base_url}/coins/{coin_id}"
            params = {
                'localization': 'false',
                'tickers': False,
                'market_data': False,
                'community_data': False,
                'developer_data': False
            }
            
            response = self.session.get(endpoint, params=params)
            if response.status_code == 200:
                data = response.json()
                categories = data.get('categories', [])
                return categories if categories else ['Other']
            else:
                return ['Other']
        except Exception as e:
            print(f"Error fetching categories for {coin_id}: {e}")
            return ['Other']
    
    def get_sector_classification(self, coin_id, categories=None):
        """코인 섹터 분류 (API 기반 동적 분류)"""
        if not categories:
            categories = self.get_coin_categories(coin_id)
        
        # 카테고리를 섹터로 매핑
        sector_mapping = {
            'L1': ['Smart Contract Platform', 'Layer 1 (L1)', 'Ethereum Ecosystem', 
                   'Binance Smart Chain Ecosystem', 'Avalanche Ecosystem', 'Cosmos Ecosystem'],
            'L2': ['Layer 2 (L2)', 'Polygon Ecosystem', 'Arbitrum Ecosystem', 
                   'Optimism Ecosystem', 'Zero Knowledge (ZK)', 'Rollups'],
            'DeFi': ['Decentralized Finance (DeFi)', 'Decentralized Exchange (DEX)', 
                    'Automated Market Maker (AMM)', 'Yield Farming', 'Lending/Borrowing',
                    'Liquid Staking Derivatives', 'Derivatives', 'Synthetic Assets'],
            'Gaming': ['Gaming (GameFi)', 'Play To Earn', 'Metaverse', 'Move To Earn'],
            'AI': ['Artificial Intelligence (AI)', 'Generative AI', 'Machine Learning',
                  'Big Data', 'Analytics'],
            'Meme': ['Meme', 'Memes'],
            'Privacy': ['Privacy Coins', 'Zero Knowledge Proofs'],
            'Exchange': ['Exchange-based Tokens', 'Centralized Exchange (CEX) Token'],
            'Web3': ['Storage', 'Filesharing', 'Oracle', 'Web3', 'Internet of Things (IoT)',
                    'Interoperability'],
            'NFT': ['Non-Fungible Tokens (NFT)', 'NFT Marketplace', 'Collectibles'],
            'RWA': ['Real World Assets (RWA)', 'Tokenized Assets'],
            'SocialFi': ['Social Token', 'Social Money', 'Content Creation']
        }
        
        # 카테고리에서 섹터 결정
        for category in categories:
            for sector, keywords in sector_mapping.items():
                if any(keyword.lower() in category.lower() for keyword in keywords):
                    return sector
        
        # 기본값
        return 'Other'
    
    def get_all_categories(self):
        """모든 사용 가능한 카테고리 목록 가져오기"""
        try:
            endpoint = f"{self.base_url}/coins/categories/list"
            response = self.session.get(endpoint)
            if response.status_code == 200:
                return response.json()
            else:
                return []
        except Exception as e:
            print(f"Error fetching categories list: {e}")
            return []