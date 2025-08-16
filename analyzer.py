import ccxt
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import time
import json
import os
from telegram import Bot
import asyncio
import schedule
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
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from concurrent.futures import ThreadPoolExecutor
from data_collector import FundamentalDataCollector
from optimizer import ParameterOptimizer

class BybitPairTrendAnalyzer:
    def __init__(self, config_path='trend_config.json'):
        """페어 추세 분석기 초기화"""
        # 설정 파일 로드
        self.config = self.load_config(config_path)
        
        # Bybit 연결
        self.exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',  # USDT 무기한 선물
            }
        })
        
        # 텔레그램 봇 초기화
        if self.config.get('telegram_bot_token'):
            self.bot = Bot(token=self.config['telegram_bot_token'])
        else:
            self.bot = None
        
        # 펀더멘털 데이터 수집기 초기화
        self.fundamental_collector = FundamentalDataCollector()
        
        # 파라미터 최적화기 초기화
        self.parameter_optimizer = ParameterOptimizer()
        
        # 섹터 정의 - 시총 200위까지 포함하도록 확장
        self.sectors = {
            'L1': ['BTC', 'ETH', 'SOL', 'ADA', 'AVAX', 'DOT', 'NEAR', 'ATOM', 'ALGO', 'TON', 'APT', 'SUI', 'SEI', 
                   'TRX', 'EOS', 'XTZ', 'VET', 'HBAR', 'KLAY', 'EGLD', 'FTM', 'ONE', 'ZIL', 'WAVES', 'KAVA', 
                   'CELO', 'FLOW', 'QNT', 'KSM', 'MINA', 'XDC', 'CKB', 'ASTR', 'ROSE', 'KDA'],
            'L2': ['ARB', 'OP', 'MATIC', 'IMX', 'MANTA', 'STRK', 'ZK', 'METIS', 'BOBA', 'LRC', 'OMG', 'MODE', 
                   'CANTO', 'CELR', 'BASE', 'BLAST', 'SCROLL', 'LINEA'],
            'DeFi': ['UNI', 'AAVE', 'MKR', 'COMP', 'SNX', 'CRV', 'SUSHI', 'YFI', 'BAL', 'LDO', 'FXS', 'GMX', 
                     'DYDX', 'JOE', 'RDNT', 'CAKE', '1INCH', 'RUNE', 'OSMO', 'PENDLE', 'RPL', 'SPELL', 'CVX', 
                     'FLX', 'VELO', 'QI', 'ANC', 'ALPHA', 'BOND', 'OHM', 'TIME', 'KNC', 'BNT', 'PERP', 'DODO'],
            'Meme': ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'MEME', 'BABYDOGE', 'ELON', 'LADYS', 
                     'TURBO', 'AIDOGE', 'WOJAK', 'BOB', 'CHAD', 'PSYOP', 'BOBO', 'HEX', 'VOLT', 'SAMO', 
                     'KISHU', 'AKITA', 'PIT', 'LEASH', 'BONE', 'SAITAMA', 'CATE', 'DOGE2'],
            'AI': ['FET', 'AGIX', 'OCEAN', 'RNDR', 'TAO', 'GRT', 'THETA', 'AKT', 'GLM', 'NMR', 'CTXC', 'MDT',
                   'ARKM', 'WLD', 'PHB', 'AI', 'ORAI', 'SDAO', 'ALI', 'VAIOT', 'DBC', 'AIOZ', 'RSS3', 'CHAT'],
            'Gaming': ['AXS', 'SAND', 'MANA', 'GALA', 'ENJ', 'IMX', 'ALICE', 'TLM', 'ILV', 'MAGIC', 'GMT', 'GST',
                       'YGG', 'GHST', 'UFO', 'STARL', 'DPET', 'MBOX', 'MOBOX', 'SLP', 'PYR', 'GODS', 'SUPER', 
                       'HERO', 'ATLAS', 'POLIS', 'DAR', 'COCOS', 'PLA', 'LOKA', 'VEMP'],
            'Web3': ['FIL', 'AR', 'STX', 'ICP', 'HNT', 'ANKR', 'AUDIO', 'LPT', 'BAT', 'MASK', 'GNO', 'POWR', 
                     'OCEAN', 'NKN', 'CVC', 'REQ', 'RLC', 'TORN', 'KEEP', 'NU', 'TRAC', 'DGB'],
            'Oracle': ['LINK', 'BAND', 'API3', 'TRB', 'DIA', 'PYTH', 'UMA', 'NEST', 'DOS', 'ZAP'],
            'Privacy': ['XMR', 'ZEC', 'DASH', 'DCR', 'SCRT', 'ROSE', 'ARRR', 'BEAM', 'GRIN', 'XVG', 'ZEN', 'PIVX'],
            'Exchange': ['BNB', 'OKB', 'CRO', 'KCS', 'HT', 'FTT', 'LEO', 'GT', 'MX', 'DODO', 'SUSHI', 'JOE', 
                         'QUICK', 'PNG', 'BAL', 'BIT', 'MEXC', 'WOO', 'SRM'],
            'Payment': ['XRP', 'XLM', 'ALGO', 'HBAR', 'CELO', 'ACH', 'AMP', 'MTL', 'REQ', 'UTK', 'PMA', 'COTI'],
            'RWA': ['ONDO', 'MPL', 'TRU', 'GFI', 'CFG', 'RIO', 'POLYX', 'DUSK', 'PRO', 'PROPS', 'RSR', 'FRAX'],
            'Storage': ['FIL', 'AR', 'STORJ', 'SIA', 'BTT', 'HOT', 'OPCT', 'XCH', 'ALEPH', 'LAMB', 'ANKR'],
            'Metaverse': ['SAND', 'MANA', 'AXS', 'ENJ', 'ALICE', 'GALA', 'RACA', 'CEEK', 'HIGH', 'BLOK', 'SENSO',
                          'WILD', 'REVV', 'BOSON', 'TVK', 'WAXP', 'NAKA', 'VOXEL'],
            'Social': ['CYBER', 'FRIEND', 'DESO', 'LENS', 'FARCASTER', 'TORUM', 'RALLY', 'WHALE', 'MITH'],
            'Infrastructure': ['RNDR', 'FLUX', 'POKT', 'DVPN', 'NKN', 'HNT', 'IOTX', 'IOST', 'LSK', 'ARK']
        }
            
    def load_config(self, config_path):
        """설정 파일 로드 (환경변수 우선)"""
        # 기본 설정
        config = {
            'telegram_bot_token': os.environ.get('TELEGRAM_BOT_TOKEN', ''),
            'telegram_chat_id': os.environ.get('TELEGRAM_CHAT_ID', ''),
            'top_n_coins': 200,  # 시총 200위까지 확장
            'trend_strength_threshold': 60,  # 페어 추세 강도 임계값 (0-100)
            'max_pairs_per_sector': 3,  # 섹터별 상위 페어 수
            'timeframes': ['4h', '1d'],  # 분석할 시간대
        }
        
        # 설정 파일이 있으면 로드하고 병합
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                # 파일 설정으로 업데이트 (환경변수가 없는 경우에만)
                for key, value in file_config.items():
                    if key in ['telegram_bot_token', 'telegram_chat_id']:
                        # 환경변수가 비어있을 때만 파일 설정 사용
                        if not config[key]:
                            config[key] = value
                    else:
                        config[key] = value
        
        return config
    
    def get_top_coins_by_market_cap(self, limit=200):
        """시가총액 기준 상위 코인 가져오기 - 200위까지 확장"""
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 시총 상위 {limit}개 코인 수집 중...")
            
            # 모든 USDT 마켓 가져오기
            markets = self.exchange.load_markets()
            usdt_markets = []
            
            for symbol, market in markets.items():
                if market['quote'] == 'USDT' and market['type'] == 'swap' and market['active']:
                    # 스테이블코인 제외
                    if market['base'] not in ['USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'UST']:
                        usdt_markets.append({
                            'symbol': symbol,
                            'base': market['base'],
                            'id': market['id']
                        })
            
            # 24시간 거래량과 가격 데이터
            tickers = self.exchange.fetch_tickers()
            volume_data = []
            
            for market in usdt_markets:
                symbol = market['symbol']
                if symbol in tickers:
                    ticker = tickers[symbol]
                    if ticker['quoteVolume'] and ticker['quoteVolume'] > 0:
                        # 시가총액 대신 거래량 * 가격을 사용 (근사치)
                        market_cap_proxy = ticker['quoteVolume'] * ticker['last']
                        volume_data.append({
                            'symbol': market['id'],
                            'base': market['base'],
                            'volume_24h': ticker['quoteVolume'],
                            'price': ticker['last'],
                            'market_cap_proxy': market_cap_proxy,
                            'change_24h': ticker['percentage'] if ticker['percentage'] else 0
                        })
            
            # 시가총액 대용값 기준 정렬
            volume_data.sort(key=lambda x: x['market_cap_proxy'], reverse=True)
            
            # 상위 N개 선택
            top_coins = volume_data[:limit]
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {len(top_coins)}개 코인 선택 완료")
            
            return top_coins
            
        except Exception as e:
            print(f"코인 데이터 수집 오류: {e}")
            return []
    
    def identify_coin_sector(self, coin_symbol: str) -> str:
        """코인이 속한 섹터 식별"""
        for sector, coins in self.sectors.items():
            if coin_symbol.upper() in coins:
                return sector
        return 'Other'
    
    def calculate_correlation(self, symbol1: str, symbol2: str, days: int = 7) -> Dict:
        """두 코인 간의 상관관계 계산"""
        try:
            # 일봉 데이터 가져오기
            limit = days + 1
            ohlcv1 = self.exchange.fetch_ohlcv(symbol1, '1d', limit=limit)
            ohlcv2 = self.exchange.fetch_ohlcv(symbol2, '1d', limit=limit)
            
            if len(ohlcv1) < days or len(ohlcv2) < days:
                return {'correlation': 0, 'p_value': 1, 'valid': False}
            
            # 종가 데이터 추출
            closes1 = [x[4] for x in ohlcv1[-days:]]
            closes2 = [x[4] for x in ohlcv2[-days:]]
            
            # 로그 수익률 계산
            returns1 = np.diff(np.log(closes1))
            returns2 = np.diff(np.log(closes2))
            
            # Pearson 상관계수
            correlation, p_value = stats.pearsonr(returns1, returns2)
            
            # 공적분 검정 (Cointegration test)
            try:
                _, coint_p_value, _ = coint(closes1, closes2)
                cointegrated = coint_p_value < 0.05
            except:
                cointegrated = False
                coint_p_value = 1.0
            
            # 베타 계산 (symbol1이 symbol2 대비 얼마나 움직이는지)
            if np.std(returns2) > 0:
                beta = np.cov(returns1, returns2)[0, 1] / np.var(returns2)
            else:
                beta = 1.0
            
            # 스프레드 안정성 (비율의 표준편차)
            ratio = np.array(closes1) / np.array(closes2)
            spread_std = np.std(ratio) / np.mean(ratio)  # 변동계수
            
            return {
                'correlation': correlation,
                'p_value': p_value,
                'cointegrated': cointegrated,
                'coint_p_value': coint_p_value,
                'beta': beta,
                'spread_stability': spread_std,
                'valid': True
            }
            
        except Exception as e:
            print(f"상관관계 계산 오류 ({symbol1}/{symbol2}): {e}")
            return {'correlation': 0, 'p_value': 1, 'valid': False}
    
    def check_market_cap_similarity(self, coin1: Dict, coin2: Dict) -> Dict:
        """시가총액 유사성 체크"""
        try:
            # 시가총액 대용값 비율
            cap_ratio = coin1['market_cap_proxy'] / coin2['market_cap_proxy']
            
            # 거래량 비율
            volume_ratio = coin1['volume_24h'] / coin2['volume_24h']
            
            # 유사성 판단
            cap_similar = 0.2 <= cap_ratio <= 5.0  # 시총 5배 이내
            volume_similar = 0.1 <= volume_ratio <= 10.0  # 거래량 10배 이내
            
            # 유동성 체크 (최소 거래량)
            min_volume = 1000000  # 100만 USDT
            liquid1 = coin1['volume_24h'] > min_volume
            liquid2 = coin2['volume_24h'] > min_volume
            
            return {
                'cap_ratio': cap_ratio,
                'volume_ratio': volume_ratio,
                'cap_similar': cap_similar,
                'volume_similar': volume_similar,
                'both_liquid': liquid1 and liquid2,
                'suitable': cap_similar and volume_similar and liquid1 and liquid2
            }
            
        except Exception as e:
            print(f"시가총액 유사성 체크 오류: {e}")
            return {'suitable': False}
    
    def classify_pair_type(self, correlation_data: Dict) -> Dict:
        """페어 타입 분류 및 전략 추천"""
        corr = correlation_data.get('correlation', 0)
        coint = correlation_data.get('cointegrated', False)
        spread_std = correlation_data.get('spread_stability', 1.0)
        
        # 페어 타입 결정
        if abs(corr) > 0.7:
            if corr > 0:
                pair_type = 'STRONG_POSITIVE'
                confidence = 'A'
                strategy = 'Mean Reversion (스프레드 벌어질 때 진입)'
            else:
                pair_type = 'STRONG_NEGATIVE'
                confidence = 'A'
                strategy = 'Hedge (한쪽 롱, 한쪽 숏)'
        elif abs(corr) > 0.5:
            if corr > 0:
                pair_type = 'MODERATE_POSITIVE'
                confidence = 'B'
                strategy = 'Trend + Mean Reversion 혼합'
            else:
                pair_type = 'MODERATE_NEGATIVE'
                confidence = 'B'
                strategy = 'Partial Hedge'
        elif abs(corr) > 0.3:
            pair_type = 'WEAK_CORRELATION'
            confidence = 'C'
            strategy = '기술적 지표 위주'
        else:
            pair_type = 'NO_CORRELATION'
            confidence = 'D'
            strategy = '페어 트레이딩 비추천'
        
        # 공적분 보너스
        if coint and confidence in ['B', 'C']:
            confidence = chr(ord(confidence) - 1)  # 한 등급 상승
            strategy += ' + 공적분 확인'
        
        # 스프레드 안정성 체크
        if spread_std < 0.1:
            spread_quality = 'STABLE'
        elif spread_std < 0.2:
            spread_quality = 'NORMAL'
        else:
            spread_quality = 'VOLATILE'
        
        return {
            'pair_type': pair_type,
            'confidence': confidence,
            'strategy': strategy,
            'spread_quality': spread_quality,
            'tradeable': confidence in ['A', 'B', 'C']
        }
    
    def visualize_trade_details(self, long_symbol: str, short_symbol: str, timeframe: str = '1h', days: int = 7):
        """거래 상세 내역 시각화 (진입/출구 시점 표시)"""
        try:
            # 백테스팅 수행
            backtest_result = self.backtest_entry_strategies(long_symbol, short_symbol, timeframe, days)
            if not backtest_result or 'backtest_trades' not in backtest_result:
                print("거래 데이터가 없습니다.")
                return
            
            # 최고 전략의 거래 내역 가져오기
            best_strategy = backtest_result['best_strategy']
            trades = backtest_result['backtest_trades'].get(best_strategy, [])
            
            if not trades:
                print(f"{best_strategy} 전략의 거래가 없습니다.")
                return
            
            print(f"\n[{best_strategy} 전략 거래 상세 내역]")
            print("=" * 80)
            
            # 거래별 상세 정보 출력
            for i, trade in enumerate(trades[:20], 1):  # 최대 20개 거래만 표시
                print(f"\n거래 #{i}")
                print(f"  - 진입 시간: {pd.to_datetime(trade['timestamp']).strftime('%Y-%m-%d %H:%M')}")
                print(f"  - 진입 가격(비율): {trade['entry']:.6f}")
                
                # 보유 기간에 따른 출구 시간 계산
                hold_period = trade.get('hold_period', 1)
                if timeframe == '1h':
                    exit_time = pd.to_datetime(trade['timestamp']) + pd.Timedelta(hours=hold_period)
                elif timeframe == '4h':
                    exit_time = pd.to_datetime(trade['timestamp']) + pd.Timedelta(hours=hold_period*4)
                elif timeframe == '15m':
                    exit_time = pd.to_datetime(trade['timestamp']) + pd.Timedelta(minutes=hold_period*15)
                else:  # 5m
                    exit_time = pd.to_datetime(trade['timestamp']) + pd.Timedelta(minutes=hold_period*5)
                
                print(f"  - 출구 시간: {exit_time.strftime('%Y-%m-%d %H:%M')}")
                print(f"  - 출구 가격(비율): {trade['exit']:.6f}")
                print(f"  - 보유 기간: {hold_period} 캔들 ({hold_period if timeframe=='1h' else hold_period*4 if timeframe=='4h' else hold_period*0.25}시간)")
                print(f"  - 포지션 방향: {trade.get('type', 'LONG')}")
                print(f"  - 수익률: {trade['profit']:.2f}%")
                print(f"  - 수수료 차감 후: {trade['profit'] - 0.08:.2f}%")
                
                # 추가 정보가 있으면 표시
                if 'rsi' in trade:
                    print(f"  - 진입 시 RSI: {trade['rsi']:.1f}")
                if 'ema_diff' in trade:
                    print(f"  - EMA 차이: {trade['ema_diff']:.2f}%")
                if 'z_score' in trade:
                    print(f"  - Z-Score: {trade['z_score']:.2f}")
            
            # 통계 요약
            print("\n" + "=" * 80)
            print("[거래 통계 요약]")
            print(f"  총 거래 수: {len(trades)}회")
            print(f"  평균 수익률: {np.mean([t['profit'] for t in trades]):.2f}%")
            print(f"  평균 수익률(수수료 차감): {np.mean([t['profit'] - 0.08 for t in trades]):.2f}%")
            print(f"  최대 수익: {max([t['profit'] for t in trades]):.2f}%")
            print(f"  최대 손실: {min([t['profit'] for t in trades]):.2f}%")
            
            return trades
            
        except Exception as e:
            print(f"거래 상세 시각화 오류: {e}")
            return None
    
    def visualize_backtest_results(self, long_symbol: str, short_symbol: str, timeframe: str = '1h', days: int = 7, save_path: str = None):
        """백테스팅 결과 시각화"""
        try:
            # 심볼명 정리 (표시용)
            long_display = long_symbol.split('/')[0] if '/' in long_symbol else long_symbol
            short_display = short_symbol.split('/')[0] if '/' in short_symbol else short_symbol
            print(f"\n{long_display}/{short_display} 백테스팅 시각화 시작...")
            
            # 백테스팅 수행
            backtest_result = self.backtest_entry_strategies(long_symbol, short_symbol, timeframe, days)
            
            if not backtest_result or 'error' in backtest_result:
                print("백테스팅 데이터 부족")
                return
            
            # 데이터 가져오기
            limit = days * (24 if timeframe == '1h' else 6 if timeframe == '4h' else 1)
            limit = min(limit, 1000)
            
            long_ohlcv = self.exchange.fetch_ohlcv(long_symbol, timeframe, limit=limit)
            short_ohlcv = self.exchange.fetch_ohlcv(short_symbol, timeframe, limit=limit)
            
            df_long = pd.DataFrame(long_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_short = pd.DataFrame(short_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 비율 및 지표 계산
            df_long['ratio'] = df_long['close'] / df_short['close']
            ratio_values = df_long['ratio'].values
            
            df_long['sma_10'] = talib.SMA(ratio_values, timeperiod=10)
            df_long['sma_20'] = talib.SMA(ratio_values, timeperiod=20)
            df_long['ema_5'] = talib.EMA(ratio_values, timeperiod=5)
            df_long['ema_9'] = talib.EMA(ratio_values, timeperiod=9)
            df_long['rsi'] = talib.RSI(ratio_values, timeperiod=14)
            
            upper, middle, lower = talib.BBANDS(ratio_values, timeperiod=20)
            df_long['bb_upper'] = upper
            df_long['bb_lower'] = lower
            df_long['bb_middle'] = middle
            
            macd, signal, hist = talib.MACD(ratio_values)
            df_long['macd'] = macd
            df_long['macd_signal'] = signal
            df_long['macd_hist'] = hist
            
            # 타임스탬프 변환
            df_long['datetime'] = pd.to_datetime(df_long['timestamp'], unit='ms')
            
            # Plotly 차트 생성
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(
                    f'{long_display}/{short_display} 비율 차트',
                    'RSI (14)',
                    'MACD',
                    '전략별 진입점'
                ),
                row_heights=[0.5, 0.15, 0.15, 0.2]
            )
            
            # 1. 비율 차트
            fig.add_trace(
                go.Scatter(x=df_long['datetime'], y=df_long['ratio'],
                          mode='lines', name='비율',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
            
            # SMA 추가
            fig.add_trace(
                go.Scatter(x=df_long['datetime'], y=df_long['sma_10'],
                          mode='lines', name='SMA10',
                          line=dict(color='orange', width=1)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df_long['datetime'], y=df_long['sma_20'],
                          mode='lines', name='SMA20',
                          line=dict(color='red', width=1)),
                row=1, col=1
            )
            
            # 볼린저 밴드
            fig.add_trace(
                go.Scatter(x=df_long['datetime'], y=df_long['bb_upper'],
                          mode='lines', name='BB Upper',
                          line=dict(color='gray', width=0.5, dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df_long['datetime'], y=df_long['bb_lower'],
                          mode='lines', name='BB Lower',
                          line=dict(color='gray', width=0.5, dash='dash')),
                row=1, col=1
            )
            
            # 2. RSI
            fig.add_trace(
                go.Scatter(x=df_long['datetime'], y=df_long['rsi'],
                          mode='lines', name='RSI',
                          line=dict(color='purple', width=1)),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # 3. MACD
            fig.add_trace(
                go.Scatter(x=df_long['datetime'], y=df_long['macd'],
                          mode='lines', name='MACD',
                          line=dict(color='blue', width=1)),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=df_long['datetime'], y=df_long['macd_signal'],
                          mode='lines', name='Signal',
                          line=dict(color='red', width=1)),
                row=3, col=1
            )
            fig.add_trace(
                go.Bar(x=df_long['datetime'], y=df_long['macd_hist'],
                      name='Histogram', marker_color='gray'),
                row=3, col=1
            )
            
            # 4. 진입점 표시
            if 'backtest_results' in backtest_result:
                colors = {
                    'ratio_momentum': 'purple',
                    'mean_reversion': 'green',
                    'trend_following': 'blue',
                    'bb_squeeze': 'orange',
                    'macd_cross': 'red',
                    'volatility_breakout': 'brown'
                }
                
                # 각 전략별 진입점 표시
                for strategy_name, strategy_data in backtest_result['backtest_trades'].items():
                    if len(strategy_data) > 0:
                        entry_times = [pd.to_datetime(t['timestamp'], unit='ms') for t in strategy_data[:20]]  # 최대 20개
                        entry_ratios = [t['entry'] for t in strategy_data[:20]]
                        
                        # 진입점 마커
                        fig.add_trace(
                            go.Scatter(x=entry_times, y=entry_ratios,
                                      mode='markers',
                                      name=f'{strategy_name} 진입',
                                      marker=dict(
                                          color=colors.get(strategy_name, 'black'),
                                          size=10,
                                          symbol='triangle-up' if strategy_data[0].get('type', 'LONG') == 'LONG' else 'triangle-down'
                                      )),
                            row=1, col=1
                        )
            
            # 레이아웃 설정
            fig.update_layout(
                title=f'{long_display}/{short_display} 백테스팅 분석 ({days}일)',
                xaxis_title='Time',
                height=1000,
                showlegend=True,
                hovermode='x unified'
            )
            
            # 축 레이블 설정
            fig.update_yaxes(title_text="Ratio", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)
            fig.update_yaxes(title_text="MACD", row=3, col=1)
            fig.update_yaxes(title_text="Strategy", row=4, col=1)
            
            # 저장 또는 표시
            if save_path:
                fig.write_html(save_path)
                print(f"\n차트 저장 완료: {save_path}")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # 파일명에서 특수문자 제거 (Windows 호환)
                long_clean = long_symbol.replace('/', '_').replace(':', '')
                short_clean = short_symbol.replace('/', '_').replace(':', '')
                filename = f"backtest_chart_{long_clean}_{short_clean}_{timestamp}.html"
                fig.write_html(filename)
                print(f"\n차트 저장 완료: {filename}")
            
            # 성과 요약 출력
            self.print_backtest_summary(backtest_result)
            
            return fig
            
        except Exception as e:
            print(f"시각화 오류: {e}")
            return None
    
    def print_backtest_summary(self, backtest_result):
        """백테스팅 요약 출력"""
        print("\n" + "="*60)
        print("📊 백테스팅 결과 요약")
        print("="*60)
        
        if 'best_strategy' in backtest_result:
            print(f"\n🏆 최적 전략: {backtest_result['best_strategy']}")
            
            if 'best_performance' in backtest_result:
                perf = backtest_result['best_performance']
                print(f"  • 거래 횟수: {perf.get('total_trades', 0)}건")
                print(f"  • 승률: {perf.get('win_rate', 0):.1f}%")
                print(f"  • 평균 수익: {perf.get('avg_profit', 0):.2f}%")
                print(f"  • 최대 수익: {perf.get('max_profit', 0):.2f}%")
                print(f"  • 최대 손실: {perf.get('max_loss', 0):.2f}%")
                print(f"  • 샤프 비율: {perf.get('sharpe_ratio', 0):.2f}")
                print(f"  • Profit Factor: {perf.get('profit_factor', 0):.2f}")
        
        print("\n📈 전략별 성과:")
        if 'backtest_results' in backtest_result:
            for strategy, perf in backtest_result['backtest_results'].items():
                if perf['total_trades'] > 0:
                    print(f"\n  [{strategy}]")
                    print(f"    거래: {perf['total_trades']}건 | 승률: {perf['win_rate']:.1f}% | 평균: {perf['avg_profit']:.2f}%")
    
    def calculate_zscore(self, ratio_values, lookback=30):
        """Z-Score 계산 (페어 트레이딩 핵심 지표)"""
        if len(ratio_values) < lookback:
            return 0
        
        recent_values = ratio_values[-lookback:]
        mean = np.mean(recent_values)
        std = np.std(recent_values)
        
        if std == 0:
            return 0
        
        zscore = (ratio_values[-1] - mean) / std
        return zscore
    
    def calculate_cointegration(self, price1, price2):
        """두 가격 시리즈의 공적분 테스트"""
        try:
            from statsmodels.tsa.stattools import coint
            score, pvalue, _ = coint(price1, price2)
            return {
                'score': score,
                'pvalue': pvalue,
                'is_cointegrated': pvalue < 0.05
            }
        except Exception as e:
            print(f"Cointegration test error: {e}")
            return {'score': 0, 'pvalue': 1, 'is_cointegrated': False}
    
    def calculate_hurst_exponent(self, ratio_values):
        """Hurst Exponent 계산 (추세 지속성 측정)"""
        if len(ratio_values) < 100:
            return 0.5
        
        lags = range(2, min(100, len(ratio_values) // 2))
        tau = [np.sqrt(np.std(np.subtract(ratio_values[lag:], ratio_values[:-lag]))) for lag in lags]
        
        try:
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return 0.5
    
    def calculate_half_life(self, ratio_values):
        """평균 회귀 반감기 계산"""
        if len(ratio_values) < 2:
            return np.inf
        
        lag = np.roll(ratio_values, 1)[1:]
        lag = np.column_stack([lag, np.ones(len(lag))])
        current = ratio_values[1:]
        
        try:
            beta = np.linalg.lstsq(lag, current, rcond=None)[0][0]
            half_life = -np.log(2) / np.log(beta) if beta > 0 and beta < 1 else np.inf
            return half_life
        except:
            return np.inf
    
    def calculate_advanced_indicators(self, df, ratio_values):
        """고급 기술적 지표 계산"""
        indicators = {}
        
        # 기존 지표들
        indicators['sma_20'] = talib.SMA(ratio_values, timeperiod=20)
        indicators['ema_9'] = talib.EMA(ratio_values, timeperiod=9)
        indicators['rsi'] = talib.RSI(ratio_values, timeperiod=14)
        
        # 새로운 지표들
        # 1. Stochastic Oscillator
        high = df['high'].values if 'high' in df.columns else ratio_values
        low = df['low'].values if 'low' in df.columns else ratio_values
        close = ratio_values
        
        slowk, slowd = talib.STOCH(high, low, close,
                                   fastk_period=14,
                                   slowk_period=3,
                                   slowk_matype=0,
                                   slowd_period=3,
                                   slowd_matype=0)
        indicators['stoch_k'] = slowk
        indicators['stoch_d'] = slowd
        
        # 2. MFI (Money Flow Index)
        if 'volume' in df.columns:
            mfi = talib.MFI(high, low, close, df['volume'].values, timeperiod=14)
            indicators['mfi'] = mfi
        
        # 3. CCI (Commodity Channel Index)
        indicators['cci'] = talib.CCI(high, low, close, timeperiod=20)
        
        # 4. Williams %R
        indicators['willr'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # 5. OBV (On Balance Volume)
        if 'volume' in df.columns:
            indicators['obv'] = talib.OBV(close, df['volume'].values)
        
        # 6. Ichimoku Cloud
        period9_high = talib.MAX(high, timeperiod=9)
        period9_low = talib.MIN(low, timeperiod=9)
        tenkan_sen = (period9_high + period9_low) / 2
        
        period26_high = talib.MAX(high, timeperiod=26)
        period26_low = talib.MIN(low, timeperiod=26)
        kijun_sen = (period26_high + period26_low) / 2
        
        indicators['tenkan_sen'] = tenkan_sen
        indicators['kijun_sen'] = kijun_sen
        
        # 7. Supertrend
        atr = talib.ATR(high, low, close, timeperiod=10)
        hl2 = (high + low) / 2
        
        multiplier = 3
        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)
        
        indicators['supertrend_upper'] = upperband
        indicators['supertrend_lower'] = lowerband
        
        return indicators
    
    def calculate_comprehensive_score(self, pair_data):
        """종합 스코어 계산 (100점 만점)"""
        score = 0
        score_details = {}
        
        # 1. 기술적 점수 (30점)
        technical_score = 0
        
        # ADX 추세 강도 (10점)
        if 'adx' in pair_data:
            adx = pair_data['adx']
            if adx > 40:
                technical_score += 10
            elif adx > 30:
                technical_score += 7
            elif adx > 25:
                technical_score += 5
            elif adx > 20:
                technical_score += 3
        
        # Z-Score (10점)
        if 'zscore_30' in pair_data:
            zscore = abs(pair_data['zscore_30'])
            if zscore > 2:
                technical_score += 10  # 극단적 편차 = 기회
            elif zscore > 1.5:
                technical_score += 7
            elif zscore > 1:
                technical_score += 5
        
        # Volume Profile (10점)
        if 'volume_increase' in pair_data:
            if pair_data['volume_increase'] > 2:  # 200% 증가
                technical_score += 10
            elif pair_data['volume_increase'] > 1.5:
                technical_score += 7
            elif pair_data['volume_increase'] > 1.2:
                technical_score += 5
        
        score_details['technical'] = technical_score
        score += technical_score
        
        # 2. 펀더멘털 점수 (30점)
        fundamental_score = 0
        
        # 시가총액 순위 차이 (10점)
        if 'market_cap_rank_diff' in pair_data:
            rank_diff = abs(pair_data['market_cap_rank_diff'])
            if rank_diff < 10:
                fundamental_score += 10  # 비슷한 시총 = 좋음
            elif rank_diff < 20:
                fundamental_score += 7
            elif rank_diff < 50:
                fundamental_score += 5
        
        # 유통률 (10점)
        if 'circulation_ratio_avg' in pair_data:
            ratio = pair_data['circulation_ratio_avg']
            if ratio > 80:
                fundamental_score += 10  # 높은 유통률 = 안정적
            elif ratio > 70:
                fundamental_score += 7
            elif ratio > 60:
                fundamental_score += 5
        
        # 섹터 매칭 (10점)
        if 'same_sector' in pair_data and pair_data['same_sector']:
            fundamental_score += 10
        elif 'related_sector' in pair_data and pair_data['related_sector']:
            fundamental_score += 5
        
        score_details['fundamental'] = fundamental_score
        score += fundamental_score
        
        # 3. 상관관계 점수 (20점)
        correlation_score = 0
        
        # 30일 상관계수 (10점)
        if 'correlation_30' in pair_data:
            corr = pair_data['correlation_30']
            if 0.3 <= corr <= 0.7:
                correlation_score += 10  # 적절한 상관관계
            elif 0.2 <= corr <= 0.8:
                correlation_score += 7
            elif 0.1 <= corr <= 0.9:
                correlation_score += 5
        
        # Cointegration (10점)
        if 'is_cointegrated' in pair_data and pair_data['is_cointegrated']:
            correlation_score += 10
        elif 'cointegration_pvalue' in pair_data and pair_data['cointegration_pvalue'] < 0.1:
            correlation_score += 5
        
        score_details['correlation'] = correlation_score
        score += correlation_score
        
        # 4. 시장 상황 점수 (20점)
        market_score = 0
        
        # 거래량 증가 (10점)
        if 'volume_24h_change' in pair_data:
            vol_change = pair_data['volume_24h_change']
            if vol_change > 100:  # 100% 증가
                market_score += 10
            elif vol_change > 50:
                market_score += 7
            elif vol_change > 20:
                market_score += 5
        
        # 변동성 적정성 (10점)
        if 'spread_volatility' in pair_data:
            volatility = pair_data['spread_volatility']
            if 0.02 <= volatility <= 0.05:  # 2-5% 변동성
                market_score += 10
            elif 0.01 <= volatility <= 0.07:
                market_score += 7
            elif 0.005 <= volatility <= 0.1:
                market_score += 5
        
        score_details['market'] = market_score
        score += market_score
        
        return {
            'total_score': score,
            'score_details': score_details,
            'grade': self.get_grade_from_score(score)
        }
    
    def get_grade_from_score(self, score):
        """점수를 등급으로 변환"""
        if score >= 80:
            return 'A+'
        elif score >= 70:
            return 'A'
        elif score >= 60:
            return 'B+'
        elif score >= 50:
            return 'B'
        elif score >= 40:
            return 'C+'
        elif score >= 30:
            return 'C'
        else:
            return 'D'
    
    def calculate_pair_specific_indicators(self, long_prices, short_prices, ratio_values):
        """페어 트레이딩 전용 지표"""
        indicators = {}
        
        # 1. Z-Score (여러 기간)
        indicators['zscore_20'] = self.calculate_zscore(ratio_values, 20)
        indicators['zscore_30'] = self.calculate_zscore(ratio_values, 30)
        indicators['zscore_60'] = self.calculate_zscore(ratio_values, 60)
        
        # 2. Cointegration
        coint_result = self.calculate_cointegration(long_prices, short_prices)
        indicators['cointegration_pvalue'] = coint_result['pvalue']
        indicators['is_cointegrated'] = coint_result['is_cointegrated']
        
        # 3. Correlation (여러 기간)
        if len(long_prices) >= 30:
            indicators['correlation_30'] = np.corrcoef(long_prices[-30:], short_prices[-30:])[0, 1]
        if len(long_prices) >= 60:
            indicators['correlation_60'] = np.corrcoef(long_prices[-60:], short_prices[-60:])[0, 1]
        
        # 4. Hurst Exponent
        indicators['hurst'] = self.calculate_hurst_exponent(ratio_values)
        
        # 5. Half-life
        indicators['half_life'] = self.calculate_half_life(ratio_values)
        
        # 6. Spread Volatility
        indicators['spread_volatility'] = np.std(ratio_values[-30:]) if len(ratio_values) >= 30 else 0
        
        return indicators
    
    def backtest_entry_strategies(self, long_symbol: str, short_symbol: str, timeframe: str = '1h', days: int = 30) -> Dict:
        """진입 전략 백테스팅 (개선버전)"""
        try:
            # 타임프레임별 데이터 개수 계산
            timeframe_minutes = {
                '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720, '1d': 1440
            }
            
            minutes_per_period = timeframe_minutes.get(timeframe, 60)
            periods_per_day = 1440 / minutes_per_period
            limit = int(days * periods_per_day)
            limit = min(limit, 1000)  # 최대 1000개
            
            # 데이터 가져오기 (since 파라미터 추가)
            since = self.exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
            long_ohlcv = self.exchange.fetch_ohlcv(long_symbol, timeframe, since=since, limit=limit)
            short_ohlcv = self.exchange.fetch_ohlcv(short_symbol, timeframe, since=since, limit=limit)
            
            if len(long_ohlcv) < 20 or len(short_ohlcv) < 20:
                return {'error': f'Not enough data: long={len(long_ohlcv)}, short={len(short_ohlcv)}'}
            
            df_long = pd.DataFrame(long_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_short = pd.DataFrame(short_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 비율 계산
            df_long['ratio'] = df_long['close'] / df_short['close']
            df_long['ratio_high'] = df_long['high'] / df_short['low']
            df_long['ratio_low'] = df_long['low'] / df_short['high']
            ratio_values = df_long['ratio'].values
            
            # 기술적 지표 계산 (최적화)
            df_long['sma_5'] = talib.SMA(ratio_values, timeperiod=5)
            df_long['sma_10'] = talib.SMA(ratio_values, timeperiod=10)
            df_long['sma_20'] = talib.SMA(ratio_values, timeperiod=20)
            df_long['ema_5'] = talib.EMA(ratio_values, timeperiod=5)
            df_long['ema_9'] = talib.EMA(ratio_values, timeperiod=9)
            df_long['rsi'] = talib.RSI(ratio_values, timeperiod=14)
            
            # 볼린저 밴드 (더 짧은 기간)
            upper, middle, lower = talib.BBANDS(ratio_values, timeperiod=20, nbdevup=2, nbdevdn=2)
            df_long['bb_upper'] = upper
            df_long['bb_lower'] = lower
            df_long['bb_middle'] = middle
            df_long['bb_width'] = (upper - lower) / middle * 100  # BB 폭
            
            # MACD
            macd, signal, hist = talib.MACD(ratio_values, fastperiod=12, slowperiod=26, signalperiod=9)
            df_long['macd'] = macd
            df_long['macd_signal'] = signal
            df_long['macd_hist'] = hist
            
            # ATR (변동성)
            df_long['atr'] = talib.ATR(df_long['ratio_high'].values, 
                                       df_long['ratio_low'].values, 
                                       ratio_values, timeperiod=14)
            
            # 비율 변화율
            df_long['ratio_change'] = df_long['ratio'].pct_change() * 100
            df_long['ratio_change_ma'] = df_long['ratio_change'].rolling(5).mean()
            
            # 여러 진입 전략 테스트 (개선)
            strategies = {
                'ratio_momentum': [],      # 비율 모멘텀
                'mean_reversion': [],      # 평균 회귀
                'trend_following': [],     # 추세 추종
                'bb_squeeze': [],          # 볼린저 스퀴즈
                'macd_cross': [],          # MACD 크로스
                'volatility_breakout': []  # 변동성 돌파
            }
            
            # 백테스팅 시작 (전체 데이터의 50% 이후부터)
            start_idx = max(50, len(df_long) // 2)  # 최소 50개 캔들 필요
            
            for i in range(start_idx, len(df_long) - 5):  # 5개 캔들 후 결과 확인
                if pd.isna(df_long['sma_20'].iloc[i]):
                    continue
                    
                current_ratio = ratio_values[i]
                
                # 타임프레임에 따른 보유 기간 설정
                if timeframe == '5m':
                    hold_periods = [24, 48, 96]  # 2시간, 4시간, 8시간
                elif timeframe == '15m':
                    hold_periods = [8, 16, 32]  # 2시간, 4시간, 8시간
                elif timeframe == '1h':
                    hold_periods = [6, 12, 24]  # 6시간, 12시간, 24시간
                elif timeframe == '4h':
                    hold_periods = [6, 12, 18]  # 24시간, 48시간, 72시간
                else:
                    hold_periods = [12, 24, 48]  # 기본값
                
                # 다양한 보유 기간 테스트
                for hold_period in hold_periods:
                    if i + hold_period >= len(df_long):
                        break
                    
                    exit_ratio = ratio_values[i + hold_period]
                    # LONG 포지션 기본 수익 계산
                    long_profit = (exit_ratio - current_ratio) / current_ratio * 100
                    # SHORT 포지션 수익 계산 (반대)
                    short_profit = (current_ratio - exit_ratio) / current_ratio * 100
                
                    # 1. 비율 모멘텀 전략 (변화율 기반)
                    if abs(df_long['ratio_change'].iloc[i]) > 0.3:  # 0.3% 이상 변화 (완화)
                        if df_long['ratio_change'].iloc[i] > 0 and df_long['rsi'].iloc[i] < 75:  # RSI 조건 완화
                            strategies['ratio_momentum'].append({
                                'entry': current_ratio,
                                'exit': exit_ratio,
                                'profit': long_profit,  # LONG 수익
                                'hold_period': hold_period,
                                'rsi': df_long['rsi'].iloc[i],
                                'timestamp': df_long['timestamp'].iloc[i]
                            })
                
                    # 2. 평균 회귀 전략 (개선)
                    deviation_20 = (current_ratio - df_long['sma_20'].iloc[i]) / df_long['sma_20'].iloc[i] * 100
                    
                    if abs(deviation_20) > 1.0:  # 1.0% 이상 이탈 (완화)
                        # 하락 후 반등 기대
                        if deviation_20 < -1.0 and df_long['rsi'].iloc[i] < 45:  # 조건 완화
                            strategies['mean_reversion'].append({
                                'entry': current_ratio,
                                'exit': exit_ratio,
                                'profit': long_profit,  # LONG 수익
                                'hold_period': hold_period,
                                'type': 'LONG',
                                'deviation': deviation_20,
                                'timestamp': df_long['timestamp'].iloc[i]
                            })
                        # 상승 후 하락 기대
                        elif deviation_20 > 1.0 and df_long['rsi'].iloc[i] > 55:  # 조건 완화
                            strategies['mean_reversion'].append({
                                'entry': current_ratio,
                                'exit': exit_ratio,
                                'profit': short_profit,  # SHORT 수익 (올바른 계산)
                                'hold_period': hold_period,
                                'type': 'SHORT',
                                'deviation': deviation_20,
                                'timestamp': df_long['timestamp'].iloc[i]
                            })
                
                    # 3. 추세 추종 전략 (개선된 버전)
                    if i > 20:  # 충분한 데이터 필요
                        # 추세 강도 계산 (간단한 방법)
                        sma_slope = (df_long['sma_10'].iloc[i] - df_long['sma_10'].iloc[i-5]) / df_long['sma_10'].iloc[i-5] * 100
                        ema_diff = (df_long['ema_5'].iloc[i] - df_long['ema_9'].iloc[i]) / df_long['ema_9'].iloc[i] * 100
                        
                        # 강한 상승 추세: EMA가 벌어지고 SMA 기울기가 양수
                        if ema_diff > 0.2 and sma_slope > 0.1:  # 상승 추세 (조건 완화)
                            # RSI가 과매수 아니고, 볼린저 밴드 중상단
                            if df_long['rsi'].iloc[i] < 80 and df_long['rsi'].iloc[i] > 30:  # RSI 범위 확대
                                bb_position = (current_ratio - df_long['bb_lower'].iloc[i]) / (df_long['bb_upper'].iloc[i] - df_long['bb_lower'].iloc[i])
                                if bb_position > 0.2 and bb_position < 0.95:  # BB 범위 확대
                                    strategies['trend_following'].append({
                                        'entry': current_ratio,
                                        'exit': exit_ratio,
                                        'profit': long_profit,
                                        'hold_period': hold_period,
                                        'type': 'LONG',
                                        'ema_diff': ema_diff,
                                        'sma_slope': sma_slope,
                                        'timestamp': df_long['timestamp'].iloc[i]
                                    })
                        
                        # 강한 하락 추세: EMA가 역전되고 SMA 기울기가 음수
                        elif ema_diff < -0.5 and sma_slope < -0.2:  # 하락 추세
                            # RSI가 과매도 아니고, 볼린저 밴드 중하단
                            if df_long['rsi'].iloc[i] > 25 and df_long['rsi'].iloc[i] < 60:
                                bb_position = (current_ratio - df_long['bb_lower'].iloc[i]) / (df_long['bb_upper'].iloc[i] - df_long['bb_lower'].iloc[i])
                                if bb_position > 0.1 and bb_position < 0.7:  # BB 중간에서 하단
                                    strategies['trend_following'].append({
                                        'entry': current_ratio,
                                        'exit': exit_ratio,
                                        'profit': short_profit,
                                        'hold_period': hold_period,
                                        'type': 'SHORT',
                                        'ema_diff': ema_diff,
                                        'sma_slope': sma_slope,
                                        'timestamp': df_long['timestamp'].iloc[i]
                                    })
                
                    # 4. 볼린저 밴드 반등 전략 (하단/상단 터치)
                    if i > 20:
                        bb_position = (current_ratio - df_long['bb_lower'].iloc[i]) / (df_long['bb_upper'].iloc[i] - df_long['bb_lower'].iloc[i])
                        
                        # 볼린저 하단 터치 후 반등
                        if bb_position < 0.05 and df_long['rsi'].iloc[i] < 35:  # BB 하단 + RSI 과매도
                            strategies['bb_squeeze'].append({
                                'entry': current_ratio,
                                'exit': exit_ratio,
                                'profit': long_profit,
                                'hold_period': hold_period,
                                'type': 'LONG',
                                'bb_position': bb_position,
                                'timestamp': df_long['timestamp'].iloc[i]
                            })
                        # 볼린저 상단 터치 후 하락
                        elif bb_position > 0.95 and df_long['rsi'].iloc[i] > 65:  # BB 상단 + RSI 과매수
                            strategies['bb_squeeze'].append({
                                'entry': current_ratio,
                                'exit': exit_ratio,
                                'profit': short_profit,
                                'hold_period': hold_period,
                                'type': 'SHORT',
                                'bb_position': bb_position,
                                'timestamp': df_long['timestamp'].iloc[i]
                            })
                
                    # 5. MACD 크로스 전략
                    if i > 0 and not pd.isna(df_long['macd'].iloc[i]):
                        # MACD 골든 크로스
                        if df_long['macd'].iloc[i] > df_long['macd_signal'].iloc[i] and \
                           df_long['macd'].iloc[i-1] <= df_long['macd_signal'].iloc[i-1]:
                            strategies['macd_cross'].append({
                                'entry': current_ratio,
                                'exit': exit_ratio,
                                'profit': long_profit,  # LONG 수익
                                'hold_period': hold_period,
                                'type': 'LONG',
                                'timestamp': df_long['timestamp'].iloc[i]
                            })
                    
                    # 6. 변동성 돌파 전략 (개선)
                    if not pd.isna(df_long['atr'].iloc[i]) and i > 10:
                        atr_ratio = df_long['atr'].iloc[i] / current_ratio * 100
                        
                        # 변동성이 증가하고 추세가 형성될 때
                        if atr_ratio > 0.8:  # 높은 변동성
                            # 상승 돌파
                            if current_ratio > df_long['sma_20'].iloc[i] * 1.01 and df_long['volume'].iloc[i] > df_long['volume'].iloc[i-10:i].mean() * 1.2:
                                strategies['volatility_breakout'].append({
                                    'entry': current_ratio,
                                    'exit': exit_ratio,
                                    'profit': long_profit,
                                    'hold_period': hold_period,
                                    'type': 'LONG',
                                    'atr': atr_ratio,
                                    'timestamp': df_long['timestamp'].iloc[i]
                                })
                            # 하락 돌파
                            elif current_ratio < df_long['sma_20'].iloc[i] * 0.99 and df_long['volume'].iloc[i] > df_long['volume'].iloc[i-10:i].mean() * 1.2:
                                strategies['volatility_breakout'].append({
                                    'entry': current_ratio,
                                    'exit': exit_ratio,
                                    'profit': short_profit,
                                    'hold_period': hold_period,
                                    'type': 'SHORT',
                                    'atr': atr_ratio,
                                    'timestamp': df_long['timestamp'].iloc[i]
                                })
                    
                    # 7. 페어 차익거래 전략 (새로운 전략)
                    if i > 30:  # 충분한 데이터 필요
                        # 30일 이동평균과의 편차
                        ma_30 = df_long['sma_20'].iloc[i-10:i].mean() if i > 10 else df_long['sma_20'].iloc[i]
                        deviation = (current_ratio - ma_30) / ma_30 * 100
                        
                        # Z-score 계산 (표준편차 기준)
                        std_30 = ratio_values[i-30:i].std()
                        mean_30 = ratio_values[i-30:i].mean()
                        z_score = (current_ratio - mean_30) / std_30 if std_30 > 0 else 0
                        
                        # 극단적 편차에서 평균회귀 기대
                        if abs(z_score) > 2:  # 2 표준편차 이상
                            if z_score < -2:  # 과매도 -> 반등 기대
                                strategies['pair_arbitrage'] = strategies.get('pair_arbitrage', [])
                                strategies['pair_arbitrage'].append({
                                    'entry': current_ratio,
                                    'exit': exit_ratio,
                                    'profit': long_profit,
                                    'hold_period': hold_period,
                                    'type': 'LONG',
                                    'z_score': z_score,
                                    'timestamp': df_long['timestamp'].iloc[i]
                                })
                            elif z_score > 2:  # 과매수 -> 하락 기대
                                strategies['pair_arbitrage'] = strategies.get('pair_arbitrage', [])
                                strategies['pair_arbitrage'].append({
                                    'entry': current_ratio,
                                    'exit': exit_ratio,
                                    'profit': short_profit,
                                    'hold_period': hold_period,
                                    'type': 'SHORT',
                                    'z_score': z_score,
                                    'timestamp': df_long['timestamp'].iloc[i]
                                })
            
            # 실제 투자 시뮬레이션 설정
            initial_capital = 100000  # $100,000 초기 자본
            trading_fee = 0.04  # 0.04% 거래 수수료 (진입 + 청산)
            
            # 각 전략별 성과 계산 (개선)
            results = {}
            for strategy_name, trades in strategies.items():
                if len(trades) > 5:  # 최소 5개 거래 필요
                    # 실제 수익 계산 (수수료 포함)
                    capital = initial_capital
                    trade_results = []
                    
                    for trade in trades:
                        gross_profit = trade['profit']  # 총 수익률 (%)
                        # 수수료 차감 (진입 0.04% + 청산 0.04% = 0.08%)
                        net_profit = gross_profit - (trading_fee * 2)
                        trade_results.append(net_profit)
                        
                        # 복리 계산
                        capital = capital * (1 + net_profit / 100)
                    
                    profits = trade_results
                    
                    # 타임프레임별 승/패 기준 조정 (수수료 포함 후)
                    # 이미 net_profit에 수수료가 차감되어 있으므로 0을 기준으로 판단
                    if timeframe in ['1h', '4h']:
                        win_threshold = 0.0  # 수수료 차감 후 이익
                        loss_threshold = 0.0  # 수수료 차감 후 손실
                    else:
                        win_threshold = 0.0  # 수수료 차감 후 이익
                        loss_threshold = 0.0  # 수수료 차감 후 손실
                    
                    winning_trades = [p for p in profits if p > win_threshold]
                    losing_trades = [p for p in profits if p <= loss_threshold]
                    
                    # 보유 기간별 평균 수익 (타임프레임에 따라 다름)
                    if timeframe == '5m':
                        hold_short = [t['profit'] for t in trades if t.get('hold_period') == 24]
                        hold_medium = [t['profit'] for t in trades if t.get('hold_period') == 48]
                        hold_long = [t['profit'] for t in trades if t.get('hold_period') == 96]
                    elif timeframe == '15m':
                        hold_short = [t['profit'] for t in trades if t.get('hold_period') == 8]
                        hold_medium = [t['profit'] for t in trades if t.get('hold_period') == 16]
                        hold_long = [t['profit'] for t in trades if t.get('hold_period') == 32]
                    elif timeframe == '1h':
                        hold_short = [t['profit'] for t in trades if t.get('hold_period') == 6]
                        hold_medium = [t['profit'] for t in trades if t.get('hold_period') == 12]
                        hold_long = [t['profit'] for t in trades if t.get('hold_period') == 24]
                    elif timeframe == '4h':
                        hold_short = [t['profit'] for t in trades if t.get('hold_period') == 6]
                        hold_medium = [t['profit'] for t in trades if t.get('hold_period') == 12]
                        hold_long = [t['profit'] for t in trades if t.get('hold_period') == 18]
                    else:
                        hold_short = [t['profit'] for t in trades if t.get('hold_period') == 12]
                        hold_medium = [t['profit'] for t in trades if t.get('hold_period') == 24]
                        hold_long = [t['profit'] for t in trades if t.get('hold_period') == 48]
                    
                    avg_profit = np.mean(profits) if profits else 0
                    std_profit = np.std(profits) if len(profits) > 1 else 1
                    
                    # 실제 투자 수익 계산
                    final_capital = capital
                    total_return = (final_capital - initial_capital) / initial_capital * 100
                    
                    # 평균 보유 기간 계산
                    avg_hold_period = np.mean([t.get('hold_period', 0) for t in trades])
                    if timeframe == '1h':
                        avg_hold_hours = avg_hold_period
                    elif timeframe == '4h':
                        avg_hold_hours = avg_hold_period * 4
                    elif timeframe == '15m':
                        avg_hold_hours = avg_hold_period * 0.25
                    else:  # 5m
                        avg_hold_hours = avg_hold_period * (5/60)
                    
                    # 연환산 수익률 계산 (30일 기준)
                    if days > 0:
                        annualized_return = (total_return / days) * 365
                    else:
                        annualized_return = 0
                    
                    results[strategy_name] = {
                        'total_trades': len(trades),
                        'avg_profit': avg_profit,
                        'total_profit': np.sum(profits),
                        'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
                        'win_count': len(winning_trades),
                        'loss_count': len(losing_trades),
                        'max_profit': max(profits) if profits else 0,
                        'max_loss': min(profits) if profits else 0,
                        'sharpe_ratio': avg_profit / std_profit if std_profit > 0 else 0,
                        'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else 999,
                        'avg_profit_short': np.mean(hold_short) if hold_short else 0,
                        'avg_profit_medium': np.mean(hold_medium) if hold_medium else 0,
                        'avg_profit_long': np.mean(hold_long) if hold_long else 0,
                        # 실제 투자 결과
                        'initial_capital': initial_capital,
                        'final_capital': final_capital,
                        'total_return_pct': total_return,
                        'total_return_usd': final_capital - initial_capital,
                        'annualized_return': annualized_return,
                        'avg_hold_hours': avg_hold_hours,
                        'trades_per_day': len(trades) / days if days > 0 else 0,
                        'total_fees_paid': len(trades) * initial_capital * (trading_fee * 2) / 100  # 추정 수수료
                    }
                else:
                    results[strategy_name] = {
                        'total_trades': len(trades),
                        'avg_profit': 0,
                        'total_profit': 0,
                        'win_rate': 0,
                        'win_count': 0,
                        'loss_count': 0,
                        'max_profit': 0,
                        'max_loss': 0,
                        'sharpe_ratio': 0,
                        'profit_factor': 0
                    }
            
            # 최적 전략 선택 (샤프비율 + 승률 고려)
            valid_strategies = [(k, v) for k, v in results.items() if v['total_trades'] >= 5]
            
            if valid_strategies:
                # 샤프비율과 승률을 모두 고려
                best_strategy = max(valid_strategies, 
                                  key=lambda x: (x[1]['sharpe_ratio'] * 0.5 + 
                                               x[1]['win_rate'] / 100 * 0.3 + 
                                               x[1]['profit_factor'] / 10 * 0.2))
            else:
                # 유효한 전략이 없으면 기본값
                best_strategy = ('trend_following', results.get('trend_following', {}))
            
            # 현재 상황에 맞는 진입점 계산
            current_ratio = ratio_values[-1]
            current_sma_5 = df_long['sma_5'].iloc[-1]
            current_sma_10 = df_long['sma_10'].iloc[-1]
            current_rsi = df_long['rsi'].iloc[-1]
            current_bb_upper = df_long['bb_upper'].iloc[-1]
            current_bb_lower = df_long['bb_lower'].iloc[-1]
            
            return {
                'backtest_results': results,
                'backtest_trades': strategies,  # 거래 데이터 추가
                'best_strategy': best_strategy[0],
                'best_performance': best_strategy[1],
                'current_signals': {
                    'ratio': current_ratio,
                    'sma_5': current_sma_5,
                    'sma_10': current_sma_10,
                    'rsi': current_rsi,
                    'bb_position': (current_ratio - current_bb_lower) / (current_bb_upper - current_bb_lower) * 100
                },
                'recommended_entry': self.get_optimized_entry_point(
                    current_ratio, current_sma_5, current_sma_10, 
                    current_rsi, current_bb_upper, current_bb_lower,
                    best_strategy[0]
                )
            }
            
        except Exception as e:
            print(f"백테스팅 오류: {e}")
            return {}
    
    def calculate_optimized_exit(self, entry_price, current_price, hold_period, strategy, adx=None, timeframe='1h'):
        """추세에 맞는 최적화된 출구 전략 계산"""
        profit_pct = (current_price - entry_price) / entry_price * 100
        
        # 타임프레임별 기본 목표 수익률
        target_profits = {
            '5m': {'short': 0.5, 'medium': 1.0, 'long': 1.5},
            '15m': {'short': 0.8, 'medium': 1.5, 'long': 2.5},
            '1h': {'short': 1.5, 'medium': 3.0, 'long': 5.0},
            '4h': {'short': 3.0, 'medium': 5.0, 'long': 8.0}
        }
        
        # 전략별 출구 조건
        exit_signals = []
        
        if strategy == 'trend_following':
            # ADX가 약해지면 추세 종료 신호
            if adx and adx < 20:
                exit_signals.append({'reason': 'ADX 약화 (추세 종료)', 'urgency': 'HIGH'})
            
            # 목표 수익 도달
            if timeframe in target_profits:
                if hold_period <= 8:
                    target = target_profits[timeframe]['short']
                elif hold_period <= 16:
                    target = target_profits[timeframe]['medium']
                else:
                    target = target_profits[timeframe]['long']
                
                if profit_pct >= target:
                    exit_signals.append({'reason': f'목표 수익 {target}% 도달', 'urgency': 'MEDIUM'})
            
            # Trailing Stop: 최고점 대비 일정 % 하락시
            trailing_stop_pct = 2.0 if timeframe in ['1h', '4h'] else 1.0
            if profit_pct > 3.0 and profit_pct < (profit_pct * 0.7):  # 30% 이상 하락
                exit_signals.append({'reason': f'Trailing Stop 발동', 'urgency': 'HIGH'})
        
        elif strategy == 'mean_reversion':
            # 평균으로 회귀 완료
            if abs(profit_pct) < 0.5:  # 평균 근처 도달
                exit_signals.append({'reason': '평균 회귀 완료', 'urgency': 'HIGH'})
        
        elif strategy == 'bb_bounce':
            # 볼린저 밴드 반대편 도달
            if profit_pct >= 2.0:
                exit_signals.append({'reason': '볼린저 밴드 목표 도달', 'urgency': 'MEDIUM'})
        
        # 손절 조건 (모든 전략 공통)
        stop_loss_pct = -3.0 if timeframe in ['1h', '4h'] else -2.0
        if profit_pct <= stop_loss_pct:
            exit_signals.append({'reason': f'손절 {stop_loss_pct}%', 'urgency': 'IMMEDIATE'})
        
        return {
            'current_profit': profit_pct,
            'exit_signals': exit_signals,
            'should_exit': len(exit_signals) > 0,
            'urgency': max([s['urgency'] for s in exit_signals], default='LOW')
        }
    
    def get_optimized_entry_point(self, current_ratio, sma_5, sma_10, rsi, bb_upper, bb_lower, strategy):
        """최적화된 진입점 계산 (개선)"""
        entry_points = []
        
        if strategy == 'ratio_momentum':
            # 모멘텀 전략: SMA 돌파 대기
            if current_ratio < sma_5:
                entry_points.append({
                    'type': 'LONG',
                    'entry': sma_5 * 1.002,  # 살짝 위에서 진입
                    'reason': 'SMA5 돌파 대기',
                    'confidence': 'HIGH'
                })
            else:
                entry_points.append({
                    'type': 'LONG',
                    'entry': current_ratio * 1.001,  # 현재가 근처
                    'reason': '모멘텀 진행중',
                    'confidence': 'MEDIUM'
                })
        
        elif strategy == 'mean_reversion':
            # 평균회귀: SMA에서 벗어난 정도에 따라
            deviation = (current_ratio - sma_10) / sma_10 * 100
            if deviation < -1:  # 1% 아래
                entry_points.append({
                    'type': 'LONG',
                    'entry': current_ratio,
                    'reason': 'SMA 아래 매수 기회',
                    'confidence': 'HIGH'
                })
            elif deviation > 1:  # 1% 위
                entry_points.append({
                    'type': 'SHORT',
                    'entry': current_ratio,
                    'reason': 'SMA 위 매도 기회',
                    'confidence': 'MEDIUM'
                })
        
        elif strategy == 'bb_bounce':
            # 볼린저 밴드 전략
            bb_position = (current_ratio - bb_lower) / (bb_upper - bb_lower) * 100
            if bb_position < 20:
                entry_points.append({
                    'type': 'LONG',
                    'entry': current_ratio,
                    'reason': '볼린저 하단 근처',
                    'confidence': 'HIGH'
                })
            elif bb_position > 80:
                entry_points.append({
                    'type': 'SHORT',
                    'entry': current_ratio,
                    'reason': '볼린저 상단 근처',
                    'confidence': 'HIGH'
                })
        
        elif strategy == 'macd_cross':
            # MACD 크로스 전략
            entry_points.append({
                'type': 'LONG',
                'entry': current_ratio * 1.001,
                'reason': 'MACD 골든크로스 대기',
                'confidence': 'MEDIUM'
            })
        
        elif strategy == 'bb_squeeze':
            # 볼린저 스퀴즈 전략
            bb_position = (current_ratio - bb_lower) / (bb_upper - bb_lower) * 100
            if bb_position < 50:
                entry_points.append({
                    'type': 'LONG',
                    'entry': current_ratio,
                    'reason': 'BB 스퀴즈 하단',
                    'confidence': 'MEDIUM'
                })
            else:
                entry_points.append({
                    'type': 'SHORT',
                    'entry': current_ratio,
                    'reason': 'BB 스퀴즈 상단',
                    'confidence': 'MEDIUM'
                })
        
        else:  # trend_following
            # 추세추종: EMA/SMA 관계
            if current_ratio > sma_5 > sma_10:
                entry_points.append({
                    'type': 'LONG',
                    'entry': sma_5,  # SMA5에서 지지
                    'reason': '상승 추세 지지선',
                    'confidence': 'HIGH'
                })
            elif current_ratio < sma_5 < sma_10:
                entry_points.append({
                    'type': 'SHORT',
                    'entry': sma_5,  # SMA5에서 저항
                    'reason': '하락 추세 저항선',
                    'confidence': 'HIGH'
                })
        
        # 현재가 대비 너무 멀지 않은 진입점만 반환
        valid_entries = []
        for entry in entry_points:
            distance = abs(entry['entry'] - current_ratio) / current_ratio * 100
            if distance < 1.5:  # 1.5% 이내만 (더 엄격하게)
                entry['distance_pct'] = distance
                valid_entries.append(entry)
        
        return valid_entries
    
    def calculate_entry_points(self, long_symbol: str, short_symbol: str, timeframe: str = '1h') -> Dict:
        """진입 타점 계산"""
        try:
            # OHLCV 데이터 가져오기 (단기 분석용 50개)
            long_ohlcv = self.exchange.fetch_ohlcv(long_symbol, timeframe, limit=50)
            short_ohlcv = self.exchange.fetch_ohlcv(short_symbol, timeframe, limit=50)
            
            df_long = pd.DataFrame(long_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_short = pd.DataFrame(short_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 비율 계산
            ratio = df_long['close'] / df_short['close']
            ratio_values = ratio.values
            
            # 기술적 지표 계산 (단기용)
            sma_5 = talib.SMA(ratio_values, timeperiod=5)[-1]
            sma_10 = talib.SMA(ratio_values, timeperiod=10)[-1]
            sma_20 = talib.SMA(ratio_values, timeperiod=20)[-1]
            upper, middle, lower = talib.BBANDS(ratio_values, timeperiod=10)
            rsi = talib.RSI(ratio_values, timeperiod=9)[-1]
            
            # 백테스팅 수행
            backtest = self.backtest_entry_strategies(long_symbol, short_symbol, timeframe)
            best_strategy = backtest.get('best_strategy', 'trend_following')
            
            current_ratio = ratio_values[-1]
            
            # 지지/저항 레벨 계산
            recent_highs = []
            recent_lows = []
            for i in range(1, len(ratio_values)-1):
                if ratio_values[i] > ratio_values[i-1] and ratio_values[i] > ratio_values[i+1]:
                    recent_highs.append(ratio_values[i])
                if ratio_values[i] < ratio_values[i-1] and ratio_values[i] < ratio_values[i+1]:
                    recent_lows.append(ratio_values[i])
            
            resistance_levels = sorted(recent_highs[-3:])[::-1] if recent_highs else []
            support_levels = sorted(recent_lows[-3:]) if recent_lows else []
            
            # 백테스팅 기반 최적화된 진입점
            if backtest and 'recommended_entry' in backtest:
                optimized_entries = backtest['recommended_entry']
            else:
                optimized_entries = []
            
            # 진입 타점 계산 (백테스팅 결과 반영)
            entry_points = {
                'optimized': optimized_entries,  # 백테스팅 최적화
                'immediate': [],
                'conservative': [],
                'aggressive': []
            }
            
            # 추가 진입점 (폴백)
            deviation = (current_ratio - sma_10) / sma_10 * 100
            
            # 즉시 진입 가능 조건 (백테스팅 기반)
            if best_strategy == 'momentum_breakout' and current_ratio > sma_5:
                entry_points['immediate'].append({
                    'type': 'LONG',
                    'entry': current_ratio,
                    'reason': '모멘텀 진행중',
                    'strategy': best_strategy
                })
            elif best_strategy == 'mean_reversion' and abs(deviation) > 1.5:
                entry_type = 'LONG' if deviation < 0 else 'SHORT'
                entry_points['immediate'].append({
                    'type': entry_type,
                    'entry': current_ratio,
                    'reason': '평균회귀 기회',
                    'strategy': best_strategy
                })
            
            # 보수적 진입 (SMA 기반)
            if current_ratio > sma_5:
                entry_points['conservative'].append({
                    'type': 'LONG',
                    'entry': sma_5,
                    'reason': 'SMA5 지지',
                    'distance': abs(sma_5 - current_ratio) / current_ratio * 100
                })
            else:
                entry_points['conservative'].append({
                    'type': 'SHORT',
                    'entry': sma_5,
                    'reason': 'SMA5 저항',
                    'distance': abs(sma_5 - current_ratio) / current_ratio * 100
                })
            
            # 공격적 진입 (볼린저/RSI)
            bb_position = (current_ratio - lower[-1]) / (upper[-1] - lower[-1]) * 100
            if bb_position < 30 or rsi < 35:
                entry_points['aggressive'].append({
                    'type': 'LONG',
                    'entry': current_ratio * 0.998,  # 약간 아래
                    'reason': f'과매도 (RSI:{rsi:.1f}, BB:{bb_position:.1f}%)',
                    'confidence': 'HIGH' if bb_position < 20 else 'MEDIUM'
                })
            elif bb_position > 70 or rsi > 65:
                entry_points['aggressive'].append({
                    'type': 'SHORT',
                    'entry': current_ratio * 1.002,  # 약간 위
                    'reason': f'과매수 (RSI:{rsi:.1f}, BB:{bb_position:.1f}%)',
                    'confidence': 'HIGH' if bb_position > 80 else 'MEDIUM'
                })
            
            # 지지/저항 기반 진입점
            if support_levels:
                nearest_support = min(support_levels, key=lambda x: abs(x - current_ratio))
                if current_ratio > nearest_support:
                    entry_points['conservative'].append({
                        'type': 'LONG',
                        'entry': nearest_support,
                        'reason': f'지지선 {nearest_support:.4f}'
                    })
            
            if resistance_levels:
                nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_ratio))
                if current_ratio < nearest_resistance:
                    entry_points['conservative'].append({
                        'type': 'SHORT',
                        'entry': nearest_resistance,
                        'reason': f'저항선 {nearest_resistance:.4f}'
                    })
            
            return {
                'current_ratio': current_ratio,
                'sma_5': sma_5,
                'sma_10': sma_10,
                'sma_20': sma_20,
                'bb_upper': upper[-1],
                'bb_lower': lower[-1],
                'rsi': rsi,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'entry_points': entry_points,
                'backtest_summary': backtest.get('best_performance', {}) if backtest else {},
                'best_strategy': best_strategy
            }
            
        except Exception as e:
            print(f"진입점 계산 오류: {e}")
            return {}
    
    def calculate_pair_trend_strength(self, long_symbol: str, short_symbol: str, timeframe: str = '4h') -> Dict:
        """페어의 추세 강도 계산
        
        Returns:
            dict: {
                'strength': 0-100 추세 강도 점수,
                'direction': 'bullish' or 'bearish',
                'ratio_trend': 비율 추세 정보,
                'entry_points': 진입 타점 정보
            }
        """
        try:
            # OHLCV 데이터 가져오기 (단기 분석용 50개)
            long_ohlcv = self.exchange.fetch_ohlcv(long_symbol, timeframe, limit=50)
            short_ohlcv = self.exchange.fetch_ohlcv(short_symbol, timeframe, limit=50)
            
            df_long = pd.DataFrame(long_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_short = pd.DataFrame(short_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 비율 계산
            ratio = df_long['close'] / df_short['close']
            ratio_high = df_long['high'] / df_short['low']  # 비율의 high
            ratio_low = df_long['low'] / df_short['high']   # 비율의 low
            
            # 기술적 지표 계산 (비율 기준)
            ratio_values = ratio.values
            ratio_high_values = ratio_high.values
            ratio_low_values = ratio_low.values
            
            # 1. ADX (비율의 추세 강도)
            adx = talib.ADX(ratio_high_values, ratio_low_values, ratio_values, timeperiod=14)[-1]
            plus_di = talib.PLUS_DI(ratio_high_values, ratio_low_values, ratio_values, timeperiod=14)[-1]
            minus_di = talib.MINUS_DI(ratio_high_values, ratio_low_values, ratio_values, timeperiod=14)[-1]
            
            # 2. RSI (비율의 모멘텀)
            rsi = talib.RSI(ratio_values, timeperiod=14)[-1]
            
            # 3. 이동평균 (비율의 추세) - 단기용
            sma_10 = talib.SMA(ratio_values, timeperiod=10)[-1]
            sma_20 = talib.SMA(ratio_values, timeperiod=20)[-1]
            ema_9 = talib.EMA(ratio_values, timeperiod=9)[-1]
            ema_21 = talib.EMA(ratio_values, timeperiod=21)[-1]
            
            # 4. MACD (비율의 모멘텀)
            macd, macd_signal, macd_hist = talib.MACD(ratio_values)
            macd_current = macd[-1]
            macd_signal_current = macd_signal[-1]
            
            # 5. Bollinger Bands (비율의 변동성) - 단기용
            upper, middle, lower = talib.BBANDS(ratio_values, timeperiod=10)
            bb_position = (ratio_values[-1] - lower[-1]) / (upper[-1] - lower[-1]) * 100 if (upper[-1] - lower[-1]) > 0 else 50
            
            # 6. 비율 변화율 (단기: 10개 캔들)
            ratio_change = (ratio_values[-1] - ratio_values[-10]) / ratio_values[-10] * 100 if ratio_values[-10] != 0 else 0
            
            # 추세 방향 결정
            bullish_signals = 0
            bearish_signals = 0
            
            # ADX 방향성
            if plus_di > minus_di:
                bullish_signals += 2  # ADX는 가중치 높게
            else:
                bearish_signals += 2
            
            # RSI
            if rsi > 50:
                bullish_signals += 1
                if rsi > 70:
                    bullish_signals += 1
            else:
                bearish_signals += 1
                if rsi < 30:
                    bearish_signals += 1
            
            # 이동평균
            if ratio_values[-1] > sma_10:
                bullish_signals += 1
            else:
                bearish_signals += 1
                
            if sma_10 > sma_20:
                bullish_signals += 1
            else:
                bearish_signals += 1
                
            if ema_9 > ema_21:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # MACD
            if macd_current > macd_signal_current:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # Bollinger Bands
            if bb_position > 80:  # 상단 밴드 근처
                bullish_signals += 1
            elif bb_position < 20:  # 하단 밴드 근처
                bearish_signals += 1
            
            # 추세 방향 결정
            direction = 'bullish' if bullish_signals > bearish_signals else 'bearish'
            
            # 추세 강도 점수 계산 (0-100)
            strength_score = 0
            
            # ADX 기반 점수 (40점)
            if adx > 50:
                strength_score += 40
            elif adx > 35:
                strength_score += 30
            elif adx > 25:
                strength_score += 20
            elif adx > 20:
                strength_score += 10
            
            # RSI 극단값 (20점)
            rsi_extreme = abs(rsi - 50)
            strength_score += min(rsi_extreme * 0.4, 20)
            
            # 이동평균 정렬도 (20점)
            if direction == 'bullish':
                if ratio_values[-1] > sma_10 > sma_20:
                    strength_score += 20
                elif ratio_values[-1] > sma_10:
                    strength_score += 10
            else:
                if ratio_values[-1] < sma_10 < sma_20:
                    strength_score += 20
                elif ratio_values[-1] < sma_10:
                    strength_score += 10
            
            # MACD 강도 (10점)
            if ratio_values[-1] != 0:
                macd_strength = abs(macd_current - macd_signal_current) / ratio_values[-1] * 100
                strength_score += min(macd_strength * 5, 10)
            
            # 비율 변화 강도 (10점) - 단기 기준 조정
            if abs(ratio_change) > 5:  # 단기라 기준 낮춤
                strength_score += 10
            elif abs(ratio_change) > 2:
                strength_score += 5
            
            # 진입점 계산 추가
            entry_analysis = self.calculate_entry_points(long_symbol, short_symbol, timeframe)
            
            return {
                'strength': min(strength_score, 100),
                'direction': direction,
                'ratio_trend': {
                    'current_ratio': ratio_values[-1],
                    'ratio_change': ratio_change,
                    'adx': adx,
                    'rsi': rsi,
                    'macd': macd_current,
                    'bb_position': bb_position
                },
                'entry_points': entry_analysis.get('entry_points', {})
            }
            
        except Exception as e:
            print(f"페어 추세 계산 오류 ({long_symbol}/{short_symbol}): {e}")
            return {
                'strength': 0,
                'direction': 'neutral',
                'ratio_trend': {}
            }
    
    def create_sector_leader_pairs(self, top_coins):
        """섹터 리더 페어 생성 (ETH/SOL 같은 주요 페어)"""
        sector_coins = {}
        
        # 코인을 섹터별로 분류
        for idx, coin in enumerate(top_coins):
            coin['rank'] = idx + 1  # 순위 추가
            sector = self.identify_coin_sector(coin['base'])
            if sector not in sector_coins:
                sector_coins[sector] = []
            sector_coins[sector].append(coin)
        
        leader_pairs = []
        
        for sector, coins in sector_coins.items():
            if len(coins) < 2:
                continue
            
            # 시가총액 기준 정렬
            coins.sort(key=lambda x: x['market_cap_proxy'], reverse=True)
            
            # 섹터 내 상위 5개 코인
            top_5 = coins[:min(5, len(coins))]
            
            # 섯터 리더 페어 (1-2, 1-3, 2-3)
            if len(top_5) >= 2:
                # 1위 vs 2위 (최우선)
                leader_pairs.append({
                    'long': top_5[0],
                    'short': top_5[1],
                    'pair_name': f"{top_5[0]['base']}/{top_5[1]['base']}",
                    'pair_category': 'SECTOR_LEADER',
                    'sector': sector,
                    'rank_diff': abs(top_5[0]['rank'] - top_5[1]['rank'])
                })
                
                # 1위 vs 3위
                if len(top_5) >= 3:
                    leader_pairs.append({
                        'long': top_5[0],
                        'short': top_5[2],
                        'pair_name': f"{top_5[0]['base']}/{top_5[2]['base']}",
                        'pair_category': 'SECTOR_TOP3',
                        'sector': sector,
                        'rank_diff': abs(top_5[0]['rank'] - top_5[2]['rank'])
                    })
                
                # 2위 vs 3위
                if len(top_5) >= 3:
                    leader_pairs.append({
                        'long': top_5[1],
                        'short': top_5[2],
                        'pair_name': f"{top_5[1]['base']}/{top_5[2]['base']}",
                        'pair_category': 'SECTOR_ADJACENT',
                        'sector': sector,
                        'rank_diff': abs(top_5[1]['rank'] - top_5[2]['rank'])
                    })
        
        return leader_pairs
    
    def create_mid_tier_pairs(self, mid_coins):
        """중간 티어 코인들간의 페어 생성 (31-100위)"""
        mid_pairs = []
        
        # 섯터별로 분류
        sector_groups = {}
        for coin in mid_coins:
            sector = self.identify_coin_sector(coin['base'])
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(coin)
        
        # 각 섹터에서 페어 생성
        for sector, coins in sector_groups.items():
            if len(coins) >= 2:
                # 섯터 내 상위 페어
                for i in range(min(3, len(coins))):
                    for j in range(i+1, min(5, len(coins))):
                        mid_pairs.append({
                            'long': coins[i],
                            'short': coins[j],
                            'pair_name': f"{coins[i]['base']}/{coins[j]['base']}",
                            'pair_category': 'MID_TIER',
                            'sector': sector
                        })
        
        # 다른 섯터간 페어도 추가
        sectors = list(sector_groups.keys())
        for i in range(len(sectors)):
            for j in range(i+1, min(i+3, len(sectors))):
                if sector_groups[sectors[i]] and sector_groups[sectors[j]]:
                    coin1 = sector_groups[sectors[i]][0]
                    coin2 = sector_groups[sectors[j]][0]
                    mid_pairs.append({
                        'long': coin1,
                        'short': coin2,
                        'pair_name': f"{coin1['base']}/{coin2['base']}",
                        'pair_category': 'CROSS_SECTOR_MID',
                        'sector': 'Cross-Sector'
                    })
        
        return mid_pairs
    
    def create_cross_tier_pairs(self, top_coins, lower_coins):
        """상위와 하위 티어간 페어 생성"""
        cross_pairs = []
        
        # 상위 10개와 하위 코인들 페어링
        for top_coin in top_coins[:10]:
            # 각 상위 코인에 대해 2-3개의 하위 코인과 페어
            import random
            selected_lower = random.sample(lower_coins, min(3, len(lower_coins)))
            for lower_coin in selected_lower:
                cross_pairs.append({
                    'long': top_coin,
                    'short': lower_coin,
                    'pair_name': f"{top_coin['base']}/{lower_coin['base']}",
                    'pair_category': 'CROSS_TIER',
                    'sector': 'Cross-Tier'
                })
        
        return cross_pairs
    
    def create_lower_tier_pairs(self, lower_coins):
        """하위 티어 코인들간의 페어 생성 (101-200위)"""
        lower_pairs = []
        
        # 간단한 순차 페어링
        for i in range(0, min(30, len(lower_coins)), 3):
            for j in range(i+1, min(i+5, len(lower_coins))):
                lower_pairs.append({
                    'long': lower_coins[i],
                    'short': lower_coins[j],
                    'pair_name': f"{lower_coins[i]['base']}/{lower_coins[j]['base']}",
                    'pair_category': 'LOWER_TIER',
                    'sector': 'Lower-Tier'
                })
        
        return lower_pairs
    
    def create_market_cap_adjacent_pairs(self, top_coins):
        """시가총액 인접 순위 페어 생성"""
        adjacent_pairs = []
        max_rank_diff = self.config.get('max_rank_difference', 10)
        
        for i in range(len(top_coins)):
            for j in range(i + 1, min(i + max_rank_diff + 1, len(top_coins))):
                coin1 = top_coins[i]
                coin2 = top_coins[j]
                
                # 시가총액 유사성 체크
                cap_similarity = self.check_market_cap_similarity(coin1, coin2)
                
                if cap_similarity.get('suitable', False):
                    adjacent_pairs.append({
                        'long': coin1,
                        'short': coin2,
                        'pair_name': f"{coin1['base']}/{coin2['base']}",
                        'pair_category': 'MARKET_CAP_ADJACENT',
                        'rank_diff': j - i,
                        'cap_ratio': cap_similarity.get('cap_ratio', 1.0)
                    })
        
        return adjacent_pairs
    
    def create_pairs_by_sector(self, top_coins):
        """다양한 페어 생성 - 시총 범위별로 분산"""
        mode = self.config.get('pair_selection_mode', 'hybrid')
        use_correlation = self.config.get('use_correlation_filter', False)
        min_correlation = self.config.get('min_correlation', 0.0)
        
        sector_pairs = {}
        all_pairs = []
        
        # 코인을 시총 그룹별로 분류
        tier1_coins = top_coins[:30]    # Top 30
        tier2_coins = top_coins[30:100] # 31-100
        tier3_coins = top_coins[100:200] # 101-200
        
        # Track 1: 상위 코인 섹터 리더 페어 (Top 30)
        if self.config.get('include_sector_leaders', True):
            leader_pairs = self.create_sector_leader_pairs(tier1_coins)
            all_pairs.extend(leader_pairs)
        
        # Track 2: 중간 티어 크로스 페어 (31-100 vs 31-100)
        mid_tier_pairs = self.create_mid_tier_pairs(tier2_coins)
        all_pairs.extend(mid_tier_pairs[:30])  # 상위 30개
        
        # Track 3: 크로스 티어 페어 (Top 30 vs 101-200)
        cross_tier_pairs = self.create_cross_tier_pairs(tier1_coins, tier3_coins)
        all_pairs.extend(cross_tier_pairs[:20])  # 상위 20개
        
        # Track 4: 하위 티어 페어 (101-200 vs 101-200)
        lower_tier_pairs = self.create_lower_tier_pairs(tier3_coins)
        all_pairs.extend(lower_tier_pairs[:20])  # 상위 20개
        
        # 페어별로 상관관계 및 기술적 지표 계산
        enhanced_pairs = []
        
        for pair in all_pairs:
            try:
                # 상관관계 계산 (필터가 아닌 정보 제공용)
                correlation_data = self.calculate_correlation(
                    pair['long']['symbol'],
                    pair['short']['symbol']
                )
                
                if correlation_data.get('valid', False):
                    pair['correlation'] = correlation_data['correlation']
                    pair['cointegrated'] = correlation_data.get('cointegrated', False)
                    pair['beta'] = correlation_data.get('beta', 1.0)
                    pair['spread_stability'] = correlation_data.get('spread_stability', 0)
                    
                    # 페어 타입 분류
                    pair_classification = self.classify_pair_type(correlation_data)
                    pair['confidence'] = pair_classification['confidence']
                    pair['strategy'] = pair_classification['strategy']
                else:
                    # 상관관계 계산 실패시 기본값
                    pair['correlation'] = 0
                    pair['confidence'] = 'U'  # Unknown
                    pair['strategy'] = '기술적 분석 필요'
                
                # 상관관계 필터 적용 여부
                if use_correlation and abs(pair['correlation']) < min_correlation:
                    continue  # 필터링
                
                enhanced_pairs.append(pair)
                time.sleep(0.1)  # API 제한
                
            except Exception as e:
                print(f"페어 분석 오류 ({pair['pair_name']}): {e}")
                continue
        
        # 섹터별로 그룹핑
        for pair in enhanced_pairs:
            sector = pair.get('sector', 'Cross-Sector')
            if sector not in sector_pairs:
                sector_pairs[sector] = []
            sector_pairs[sector].append(pair)
        
        # 각 섹터별 정렬 (중요도 순)
        for sector in sector_pairs:
            sector_pairs[sector].sort(key=lambda x: (
                x.get('pair_category', '') == 'SECTOR_LEADER',  # 섹터 리더 우선
                abs(x.get('correlation', 0))  # 그 다음 상관관계
            ), reverse=True)
        
        return sector_pairs
    
    def find_sector_best_pairs(self):
        """섹터별 최강 페어 찾기"""
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 섹터별 페어 추세 분석 시작")
        print("="*60)
        
        # 1. 상위 코인 가져오기
        top_coins = self.get_top_coins_by_market_cap(self.config['top_n_coins'])
        
        if not top_coins:
            print("코인 데이터를 가져올 수 없습니다.")
            return {}
        
        # 2. 섹터별 페어 생성
        sector_pairs = self.create_pairs_by_sector(top_coins)
        
        # 3. 각 섹터별로 페어 분석
        sector_results = {}
        
        for sector, pairs in sector_pairs.items():
            print(f"\n[{sector}] 섹터 분석 중... ({len(pairs)}개 페어)")
            
            sector_trend_pairs = []
            
            for pair in pairs:
                # 여러 시간대 분석
                multi_tf_strength = []
                multi_tf_direction = []
                ratio_trends = []
                
                # 상관관계 정보가 이미 있는지 확인
                if 'correlation' not in pair:
                    pair['correlation'] = 0
                    pair['confidence'] = pair.get('confidence', 'U')
                
                # 상관관계 필터링 옵션 확인
                use_corr_filter = self.config.get('use_correlation_filter', False)
                min_corr = self.config.get('min_correlation', 0.0)
                
                # 섹터 리더는 항상 포함
                is_sector_leader = pair.get('pair_category') == 'SECTOR_LEADER'
                
                # 필터링 로직
                if use_corr_filter and not is_sector_leader:
                    if abs(pair.get('correlation', 0)) < min_corr:
                        print(f"  [{pair['pair_name']}] 상관관계 부족 (r={pair.get('correlation', 0):.2f}) - 스킵")
                        continue
                
                for tf in self.config['timeframes']:
                    result = self.calculate_pair_trend_strength(
                        pair['long']['symbol'], 
                        pair['short']['symbol'], 
                        tf
                    )
                    multi_tf_strength.append(result['strength'])
                    multi_tf_direction.append(result['direction'])
                    ratio_trends.append(result['ratio_trend'])
                    time.sleep(0.3)  # API 제한 방지
                
                # 평균 추세 강도
                avg_strength = np.mean(multi_tf_strength)
                
                # 방향 일치성 확인
                from collections import Counter
                direction_counts = Counter(multi_tf_direction)
                most_common_direction = direction_counts.most_common(1)[0]
                direction_consistency = most_common_direction[1] / len(multi_tf_direction)
                
                # 최종 점수
                final_score = avg_strength * direction_consistency
                
                if final_score >= self.config['trend_strength_threshold']:
                    sector_trend_pairs.append({
                        'pair_name': pair['pair_name'],
                        'long_symbol': pair['long']['symbol'],
                        'short_symbol': pair['short']['symbol'],
                        'long_base': pair['long']['base'],
                        'short_base': pair['short']['base'],
                        'trend_strength': avg_strength,
                        'direction': most_common_direction[0],
                        'direction_consistency': direction_consistency,
                        'final_score': final_score,
                        'timeframes': dict(zip(self.config['timeframes'], multi_tf_strength)),
                        'ratio_trend': ratio_trends[0] if ratio_trends else {},  # 첫 번째 시간대의 상세 정보
                        'entry_points': result.get('entry_points', {}),
                        'correlation': pair.get('correlation', 0),
                        'confidence': pair.get('confidence', 'C'),
                        'strategy': pair.get('strategy', ''),
                        'cointegrated': pair.get('cointegrated', False),
                        'beta': pair.get('beta', 1.0),
                        'pair_category': pair.get('pair_category', 'TREND')
                    })
            
            # 섹터별 상위 N개 선택
            sector_trend_pairs.sort(key=lambda x: x['final_score'], reverse=True)
            sector_results[sector] = sector_trend_pairs[:self.config['max_pairs_per_sector']]
        
        return sector_results
    
    async def send_telegram_message(self, message: str, sector_results: Dict = None):
        """텔레그램 메시지 전송"""
        if not self.bot or not self.config.get('telegram_chat_id'):
            print("텔레그램 설정이 없습니다.")
            return
        
        try:
            # httpx 연결 풀 설정
            import httpx
            from telegram import Bot
            
            # 새로운 봇 인스턴스 생성 (연결 풀 크기 증가)
            self.bot = Bot(
                token=self.config['telegram_bot_token']
            )
            
            # 메시지 길이 체크 및 분할 전송
            max_length = 3500  # 텔레그램 제한보다 여유있게
            
            if len(message) > max_length:
                # 메세지를 섹션별로 분할
                sections = []
                current_section = []
                current_length = 0
                
                for line in message.split('\n'):
                    if current_length + len(line) > max_length:
                        if current_section:
                            sections.append('\n'.join(current_section))
                            current_section = [line]
                            current_length = len(line)
                    else:
                        current_section.append(line)
                        current_length += len(line) + 1
                
                if current_section:
                    sections.append('\n'.join(current_section))
                
                # 각 섹션 전송
                for i, section in enumerate(sections):
                    if len(section.strip()) > 10:  # 빈 메시지 방지
                        await self.bot.send_message(
                            chat_id=self.config['telegram_chat_id'],
                            text=section
                            # parse_mode 제거 - 일반 텍스트로 전송
                        )
                        await asyncio.sleep(1)  # 각 메시지 사이 대기
            else:
                # 한 번에 전송
                # parse_mode 제거하여 일반 텍스트로 전송
                await self.bot.send_message(
                    chat_id=self.config['telegram_chat_id'],
                    text=message
                    # parse_mode 제거 - Markdown 파싱 오류 방지
                )
            
            print("텔레그램 메시지 전송 완료")
        except Exception as e:
            print(f"텔레그램 메시지 전송 오류: {e}")
            import traceback
            traceback.print_exc()
            
            # 오류 시 더 짧은 요약 메시지로 재시도
            try:
                await asyncio.sleep(2)
                
                # 섹터 결과가 있는 경우 주요 정보만 추출
                if sector_results:
                    # 상위 3개 페어만 추출
                    top_pairs = []
                    for sector, pairs in list(sector_results.items())[:3]:
                        if pairs and len(pairs) > 0:
                            top_pair = pairs[0]
                            top_pairs.append({
                                'sector': sector,
                                'pair': top_pair.get('pair_name', 'Unknown'),
                                'strength': top_pair.get('trend_strength', 0),
                                'direction': top_pair.get('direction', 'neutral')
                            })
                    
                    if top_pairs:
                        simple_msg = f"📊 **페어 분석 완료!**\n\n"
                        simple_msg += f"🏆 **TOP 3 추천:**\n"
                        for i, pair in enumerate(top_pairs, 1):
                            direction_emoji = "📈" if pair['direction'] == 'bullish' else "📉" if pair['direction'] == 'bearish' else "➡️"
                            simple_msg += f"{i}. {pair['pair']} {direction_emoji}\n"
                            simple_msg += f"   강도: {pair['strength']:.0f} | {pair['sector']}\n"
                        simple_msg += f"\n⏰ {datetime.now().strftime('%H:%M')}"
                    else:
                        simple_msg = "📊 페어 분석 완료. 강한 추세 페어 없음."
                else:
                    simple_msg = "📊 페어 분석 완료. 데이터 수집 중..."
                
                await self.bot.send_message(
                    chat_id=self.config['telegram_chat_id'],
                    text=simple_msg
                    # parse_mode 제거 - 일반 텍스트로 전송
                )
                print("간단한 요약 메시지 전송 완료")
            except Exception as e2:
                print(f"요약 메시지도 전송 실패: {e2}")
                print("텔레그램 전송 실패. 콘솔 출력만 진행합니다.")
    
    def format_sector_report(self, sector_results: Dict) -> str:
        """섹터별 리포트 포맷팅 - 더 구체적인 정보 제공"""
        if not sector_results or all(len(pairs) == 0 for pairs in sector_results.values()):
            return "📊 *페어 추세 분석 결과*\n\n오늘은 강한 추세를 보이는 페어가 없습니다."
        
        # 전체 TOP 페어 먼저 계산
        all_pairs = []
        for sector, pairs in sector_results.items():
            for pair in pairs:
                pair['sector'] = sector
                all_pairs.append(pair)
        
        all_pairs.sort(key=lambda x: x.get('final_score', x.get('trend_strength', 0)), reverse=True)
        top_pairs = all_pairs[:5]  # 상위 5개
        
        message = f"🔥 *즉시 매매 가능한 TOP 페어*\n"
        message += f"⏰ {datetime.now().strftime('%H:%M')} 기준\n"
        message += f"━━━━━━━━━━━━━━━━━━\n\n"
        
        # 섹터 이모지 매핑
        sector_emojis = {
            'L1': '⛓️',
            'L2': '🔗',
            'DeFi': '💰',
            'Meme': '🐕',
            'AI': '🤖',
            'Gaming': '🎮',
            'Web3': '🌐',
            'Oracle': '🔮',
            'Privacy': '🔒',
            'Exchange': '💱',
            'Payment': '💳',
            'RWA': '🏠',
            'Storage': '💾',
            'Metaverse': '🌌',
            'Other': '📦'
        }
        
        # TOP 5 페어 표시 (즉시 활용 가능한 정보)
        for i, pair in enumerate(top_pairs, 1):
            sector_emoji = sector_emojis.get(pair['sector'], '📦')
            
            # 방향 표시
            if pair['direction'] == 'bullish':
                dir_icon = "📈 LONG"
                action = "매수"
            elif pair['direction'] == 'bearish':
                dir_icon = "📉 SHORT"
                action = "매도"
            else:
                dir_icon = "➡️ NEUTRAL"
                action = "대기"
            
            # 메달 이모지
            medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
            medal = medals[i-1] if i <= 5 else ''
            
            message += f"{medal} *{pair['pair_name']}* {dir_icon}\n"
            message += f"   {sector_emoji} {pair['sector']} | 강도: {pair['trend_strength']:.0f}\n"
            
            # 핵심 정보만 표시
            if pair.get('ratio_trend'):
                ratio_info = pair['ratio_trend']
                current_ratio = ratio_info.get('current_ratio', 0)
                ratio_change = ratio_info.get('ratio_change', 0)
                rsi = ratio_info.get('rsi', 50)
                
                # RSI 기반 상태
                if rsi > 70:
                    rsi_status = "⚠️ 과매수"
                elif rsi < 30:
                    rsi_status = "⚠️ 과매도"
                else:
                    rsi_status = f"RSI {rsi:.0f}"
                
                message += f"   비율: {current_ratio:.4f} ({ratio_change:+.1f}%) | {rsi_status}\n"
            
            # 진입 추천 정보
            if pair.get('entry_points'):
                entry_pts = pair['entry_points']
                if entry_pts.get('optimized'):
                    ep = entry_pts['optimized'][0]
                    message += f"   💎 추천: {ep['type']} @ {ep['entry']:.4f}\n"
                elif entry_pts.get('immediate'):
                    ep = entry_pts['immediate'][0]
                    message += f"   ⚡ 즉시: {ep['type']} @ {ep['entry']:.4f}\n"
            
            # 백테스팅 성과 (있는 경우)
            if 'backtest_summary' in pair and pair['backtest_summary']:
                bt = pair['backtest_summary']
                if bt.get('total_trades', 0) > 0:
                    message += f"   📊 백테스트: 승률 {bt.get('win_rate', 0):.0f}% | 평균 {bt.get('avg_profit', 0):+.1f}%\n"
            
            message += "\n"
        
        # 섹터별 요약 (간단히)
        message += "━━━━━━━━━━━━━━━━━━\n"
        message += "📂 *섹터별 1위*\n\n"
        
        sector_summary = []
        for sector, pairs in sector_results.items():
            if pairs:
                top_pair = pairs[0]
                sector_emoji = sector_emojis.get(sector, '📦')
                if top_pair['direction'] == 'bullish':
                    dir_mark = "↗️"
                elif top_pair['direction'] == 'bearish':
                    dir_mark = "↘️"
                else:
                    dir_mark = "→"
                sector_summary.append(f"{sector_emoji} {sector}: {top_pair['pair_name']} {dir_mark}")
        
        # 3개씩 묶어서 표시
        for line in sector_summary[:6]:  # 최대 6개 섹터
            message += f"{line}\n"
        
        message += "\n━━━━━━━━━━━━━━━━━━\n"
        message += "⚡ *즉시 활용 가능*\n"
        message += "• 상위 5개 페어 중 선택\n"
        message += "• 강도 60 이상 = 강한 추세\n"
        message += "• RSI 30/70 = 과매도/과매수\n"
        message += "• 백테스트 승률 참고\n"
        message += "\n💰 *리스크 관리 필수!*"
        
        return message
    
    def run_analysis_and_notify(self):
        """분석 실행 및 알림 전송"""
        try:
            # 섹터별 페어 분석
            sector_results = self.find_sector_best_pairs()
            
            # 리포트 생성
            report = self.format_sector_report(sector_results)
            
            # 콘솔 출력
            print("\n" + "="*60)
            print(report.replace('*', '').replace('├', '|').replace('└', '|').replace('🏆', '[1위]'))
            print("="*60)
            
            # 텔레그램 전송 (sector_results도 함께 전달)
            if self.config.get('telegram_bot_token'):
                asyncio.run(self.send_telegram_message(report, sector_results))
            
            # CSV 저장 (기록용)
            all_pairs = []
            for sector, pairs in sector_results.items():
                for pair in pairs:
                    pair['sector'] = sector
                    all_pairs.append(pair)
            
            if all_pairs:
                df = pd.DataFrame(all_pairs)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"sector_pair_trend_{timestamp}.csv"
                df.to_csv(filename, index=False)
                print(f"\n결과 저장: {filename}")
                
        except Exception as e:
            print(f"분석 실행 오류: {e}")
            error_msg = f"⚠️ *오류 발생*\n{str(e)}"
            if self.config.get('telegram_bot_token'):
                asyncio.run(self.send_telegram_message(error_msg))

def setup_scheduler(analyzer):
    """스케줄러 설정"""
    # 매일 오전 9시에 실행
    schedule.every().day.at("09:00").do(analyzer.run_analysis_and_notify)
    
    # 추가 스케줄 옵션
    print("\n추가 스케줄 설정:")
    print("1. 오전 9시만")
    print("2. 오전 9시, 오후 3시")
    print("3. 오전 9시, 오후 3시, 오후 9시")
    print("4. 매 4시간마다")
    print("5. 매 2시간마다")
    
    schedule_choice = input("선택 (1-5): ").strip()
    
    if schedule_choice == '2':
        schedule.every().day.at("15:00").do(analyzer.run_analysis_and_notify)
        print("스케줄러 설정: 오전 9시, 오후 3시")
    elif schedule_choice == '3':
        schedule.every().day.at("15:00").do(analyzer.run_analysis_and_notify)
        schedule.every().day.at("21:00").do(analyzer.run_analysis_and_notify)
        print("스케줄러 설정: 오전 9시, 오후 3시, 오후 9시")
    elif schedule_choice == '4':
        schedule.every(4).hours.do(analyzer.run_analysis_and_notify)
        print("스케줄러 설정: 매 4시간마다")
    elif schedule_choice == '5':
        schedule.every(2).hours.do(analyzer.run_analysis_and_notify)
        print("스케줄러 설정: 매 2시간마다")
    else:
        print("스케줄러 설정: 매일 오전 9시")
    
    print("대기 중... (Ctrl+C로 종료)")
    
    # 다음 실행 예정 시간 표시
    next_run = schedule.next_run()
    if next_run:
        print(f"다음 실행 예정: {next_run}")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # 1분마다 체크

def create_trend_following_visualizations(df_results, timeframe, days):
    """Trend Following 백테스팅 결과 시각화"""
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import plotly.express as px
        
        print("\n📊 시각화 생성 중...")
        
        # 1. 종합 대시보드 생성
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                '🏆 TOP 15 수익률 페어',
                '📊 승률 분포',
                '💰 수익률 분포',
                '📈 승률 vs 수익률',
                '⚡ 샤프 비율 vs 수익률',
                '📉 거래 횟수 vs 수익률',
                '🎯 Profit Factor 분포',
                '⏱️ 평균 보유시간 분포',
                '🔥 종합 점수 TOP 10'
            ),
            specs=[[{'type': 'bar'}, {'type': 'histogram'}, {'type': 'histogram'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'box'}, {'type': 'histogram'}, {'type': 'bar'}]]
        )
        
        # 1. TOP 15 수익률 페어
        top15 = df_results.nlargest(15, 'total_profit')
        colors = ['green' if x > 0 else 'red' for x in top15['total_profit']]
        fig.add_trace(
            go.Bar(
                x=top15['pair'][:15], 
                y=top15['total_profit'],
                marker_color=colors,
                text=[f"{x:.1f}%" for x in top15['total_profit']],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 2. 승률 분포
        fig.add_trace(
            go.Histogram(
                x=df_results['win_rate'],
                nbinsx=20,
                marker_color='lightblue',
                name='Win Rate'
            ),
            row=1, col=2
        )
        
        # 3. 수익률 분포
        fig.add_trace(
            go.Histogram(
                x=df_results['total_profit'],
                nbinsx=30,
                marker_color='lightgreen',
                name='Total PnL'
            ),
            row=1, col=3
        )
        
        # 4. 승률 vs 수익률
        fig.add_trace(
            go.Scatter(
                x=df_results['win_rate'],
                y=df_results['total_profit'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df_results['sharpe_ratio'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe", x=1.15, y=0.5)
                ),
                text=df_results['pair'],
                hovertemplate='<b>%{text}</b><br>승률: %{x:.1f}%<br>수익: %{y:.2f}%'
            ),
            row=2, col=1
        )
        
        # 5. 샤프 비율 vs 수익률
        fig.add_trace(
            go.Scatter(
                x=df_results['sharpe_ratio'],
                y=df_results['total_profit'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df_results['win_rate'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Win%", x=1.35, y=0.5)
                ),
                text=df_results['pair'],
                hovertemplate='<b>%{text}</b><br>샤프: %{x:.2f}<br>수익: %{y:.2f}%'
            ),
            row=2, col=2
        )
        
        # 6. 거래 횟수 vs 수익률
        fig.add_trace(
            go.Scatter(
                x=df_results['total_trades'],
                y=df_results['total_profit'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df_results['avg_profit'],
                    colorscale='RdBu',
                    showscale=True,
                    colorbar=dict(title="Avg%", x=1.55, y=0.5)
                ),
                text=df_results['pair'],
                hovertemplate='<b>%{text}</b><br>거래: %{x}건<br>수익: %{y:.2f}%'
            ),
            row=2, col=3
        )
        
        # 7. Profit Factor 분포 (Box Plot)
        valid_pf = df_results[df_results['profit_factor'] != float('inf')]['profit_factor']
        if len(valid_pf) > 0:
            fig.add_trace(
                go.Box(
                    y=valid_pf,
                    name='PF',
                    marker_color='orange'
                ),
                row=3, col=1
            )
        
        # 8. 평균 보유시간 분포
        fig.add_trace(
            go.Histogram(
                x=df_results['avg_hold_hours'],
                nbinsx=20,
                marker_color='purple',
                name='Hold Hours'
            ),
            row=3, col=2
        )
        
        # 9. 종합 점수 TOP 10
        # 종합 점수 계산
        df_results['composite_score'] = (
            df_results['win_rate'] * 0.3 +
            df_results['total_profit'] * 0.4 +
            df_results['sharpe_ratio'] * 10 * 0.3
        )
        top10_composite = df_results.nlargest(10, 'composite_score')
        
        fig.add_trace(
            go.Bar(
                x=top10_composite['pair'][:10],
                y=top10_composite['composite_score'],
                marker_color='gold',
                text=[f"{x:.1f}" for x in top10_composite['composite_score']],
                textposition='outside'
            ),
            row=3, col=3
        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            height=1200,
            showlegend=False,
            title_text=f"📊 Trend Following 백테스팅 대시보드 ({timeframe}, {days}일)",
            title_font_size=20
        )
        
        # 축 레이블 업데이트
        fig.update_xaxes(title_text="페어", row=1, col=1, tickangle=45)
        fig.update_yaxes(title_text="총 수익률 (%)", row=1, col=1)
        
        fig.update_xaxes(title_text="승률 (%)", row=1, col=2)
        fig.update_yaxes(title_text="빈도", row=1, col=2)
        
        fig.update_xaxes(title_text="총 수익률 (%)", row=1, col=3)
        fig.update_yaxes(title_text="빈도", row=1, col=3)
        
        fig.update_xaxes(title_text="승률 (%)", row=2, col=1)
        fig.update_yaxes(title_text="총 수익률 (%)", row=2, col=1)
        
        fig.update_xaxes(title_text="샤프 비율", row=2, col=2)
        fig.update_yaxes(title_text="총 수익률 (%)", row=2, col=2)
        
        fig.update_xaxes(title_text="거래 횟수", row=2, col=3)
        fig.update_yaxes(title_text="총 수익률 (%)", row=2, col=3)
        
        fig.update_yaxes(title_text="Profit Factor", row=3, col=1)
        
        fig.update_xaxes(title_text="평균 보유시간 (시간)", row=3, col=2)
        fig.update_yaxes(title_text="빈도", row=3, col=2)
        
        fig.update_xaxes(title_text="페어", row=3, col=3, tickangle=45)
        fig.update_yaxes(title_text="종합 점수", row=3, col=3)
        
        # HTML 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_file = f"trend_following_dashboard_{timeframe}_{days}d_{timestamp}.html"
        fig.write_html(dashboard_file)
        print(f"📊 대시보드 저장: {dashboard_file}")
        
        # 2. 3D 산점도
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=df_results['win_rate'],
            y=df_results['sharpe_ratio'],
            z=df_results['total_profit'],
            mode='markers+text',
            marker=dict(
                size=6,
                color=df_results['total_trades'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="거래횟수")
            ),
            text=df_results['pair'],
            textposition='top center',
            textfont=dict(size=8),
            hovertemplate='<b>%{text}</b><br>' +
                         '승률: %{x:.1f}%<br>' +
                         '샤프: %{y:.2f}<br>' +
                         '수익: %{z:.2f}%<br>'
        )])
        
        fig_3d.update_layout(
            title=f"3D 분석: 승률 vs 샤프 비율 vs 총 수익률 ({timeframe}, {days}일)",
            scene=dict(
                xaxis_title='승률 (%)',
                yaxis_title='샤프 비율',
                zaxis_title='총 수익률 (%)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=700
        )
        
        analysis_3d_file = f"trend_following_3d_{timeframe}_{days}d_{timestamp}.html"
        fig_3d.write_html(analysis_3d_file)
        print(f"📊 3D 분석 저장: {analysis_3d_file}")
        
        # 3. 상관관계 히트맵
        numeric_cols = ['total_trades', 'win_rate', 'avg_profit', 'total_profit', 
                       'sharpe_ratio', 'avg_hold_hours']
        
        # 데이터 유효성 검사
        valid_cols = [col for col in numeric_cols if col in df_results.columns]
        if len(valid_cols) > 1:
            corr_matrix = df_results[valid_cols].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 12},
                colorbar=dict(title="상관계수")
            ))
            
            fig_corr.update_layout(
                title=f"상관관계 매트릭스 ({timeframe}, {days}일)",
                height=600,
                width=700
            )
            
            corr_file = f"trend_following_correlation_{timeframe}_{days}d_{timestamp}.html"
            fig_corr.write_html(corr_file)
            print(f"📊 상관관계 매트릭스 저장: {corr_file}")
        
        print("\n✅ 모든 시각화 파일이 생성되었습니다!")
        print("브라우저에서 HTML 파일을 열어 인터랙티브 차트를 확인하세요.")
        
    except ImportError:
        print("\n⚠️ Plotly가 설치되지 않았습니다. 시각화를 건너뜁니다.")
        print("설치하려면: pip install plotly")
    except Exception as e:
        print(f"\n⚠️ 시각화 생성 중 오류: {e}")

def simple_trend_following_backtest(analyzer, long_symbol, short_symbol, timeframe='1h', days=30):
    """간단한 trend following 백테스팅"""
    try:
        # 데이터 가져오기
        since = analyzer.exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
        long_ohlcv = analyzer.exchange.fetch_ohlcv(long_symbol, timeframe, since=since, limit=1000)
        short_ohlcv = analyzer.exchange.fetch_ohlcv(short_symbol, timeframe, since=since, limit=1000)
        
        if len(long_ohlcv) < 20 or len(short_ohlcv) < 20:
            return None
        
        # DataFrame 생성
        df_long = pd.DataFrame(long_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_short = pd.DataFrame(short_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 타임스탬프 매칭
        df_long['timestamp'] = pd.to_datetime(df_long['timestamp'], unit='ms')
        df_short['timestamp'] = pd.to_datetime(df_short['timestamp'], unit='ms')
        
        # 공통 타임스탬프만 사용
        merged = pd.merge(df_long, df_short, on='timestamp', suffixes=('_long', '_short'))
        
        if len(merged) < 20:
            return None
        
        # 비율 계산
        merged['ratio'] = merged['close_long'] / merged['close_short']
        
        # 볼린저 밴드 계산 (20일 이동평균)
        window = min(20, len(merged) // 3)
        merged['ma'] = merged['ratio'].rolling(window=window).mean()
        merged['std'] = merged['ratio'].rolling(window=window).std()
        merged['upper'] = merged['ma'] + (2 * merged['std'])
        merged['lower'] = merged['ma'] - (2 * merged['std'])
        
        # 시그널 생성
        merged['signal'] = 0
        merged.loc[merged['ratio'] > merged['upper'], 'signal'] = -1  # Short ratio
        merged.loc[merged['ratio'] < merged['lower'], 'signal'] = 1   # Long ratio
        
        # 백테스팅
        trades = []
        position = 0
        entry_price = 0
        entry_idx = 0
        
        for idx in range(window, len(merged)):
            current_signal = merged.iloc[idx]['signal']
            
            if position == 0 and current_signal != 0:
                # 진입
                position = current_signal
                entry_price = merged.iloc[idx]['ratio']
                entry_idx = idx
                
            elif position != 0 and (current_signal != position or idx == len(merged) - 1):
                # 청산
                exit_price = merged.iloc[idx]['ratio']
                
                if position == 1:  # Long ratio
                    pnl = (exit_price - entry_price) / entry_price * 100
                else:  # Short ratio
                    pnl = (entry_price - exit_price) / entry_price * 100
                
                trades.append({
                    'pnl': pnl,
                    'duration': idx - entry_idx
                })
                
                position = 0
        
        if len(trades) == 0:
            return None
        
        # 통계 계산
        pnls = [t['pnl'] for t in trades]
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        stats = {
            'total_trades': len(trades),
            'win_rate': len(winning_trades) / len(trades) * 100 if len(trades) > 0 else 0,
            'avg_profit': np.mean(pnls),
            'total_profit': np.sum(pnls),
            'max_profit': max(pnls),
            'max_loss': min(pnls),
            'sharpe_ratio': np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0,
            'profit_factor': abs(sum([t['pnl'] for t in winning_trades]) / sum([t['pnl'] for t in losing_trades])) if losing_trades and sum([t['pnl'] for t in losing_trades]) != 0 else float('inf') if winning_trades else 0,
            'avg_hold_hours': np.mean([t['duration'] for t in trades]) if trades else 0
        }
        
        return stats
        
    except Exception as e:
        return None

def run_realtime_monitoring(analyzer):
    """실시간 추천 페어 모니터링 및 진입 알림"""
    import schedule
    from datetime import datetime, timedelta
    import winsound  # Windows 알림음
    
    print("\n" + "="*70)
    print("🔔 실시간 추천 페어 모니터링 시스템")
    print("="*70)
    
    # 주요 코인 목록
    major_coins = [
        'BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'SOL/USDT:USDT',
        'XRP/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'DOGE/USDT:USDT',
        'TRX/USDT:USDT', 'LINK/USDT:USDT', 'MATIC/USDT:USDT', 'DOT/USDT:USDT',
        'SHIB/USDT:USDT', 'UNI/USDT:USDT', 'LTC/USDT:USDT', 'ATOM/USDT:USDT',
        'NEAR/USDT:USDT', 'ARB/USDT:USDT', 'OP/USDT:USDT', 'APT/USDT:USDT'
    ]
    
    # 마켓 로드
    if not analyzer.exchange.markets:
        analyzer.exchange.load_markets()
    
    # 유효한 심볼 필터링
    valid_symbols = [s for s in major_coins if s in analyzer.exchange.markets]
    
    # 현재 추천 페어 저장
    current_top_pair = None
    last_signal_time = {}
    
    def analyze_and_recommend():
        """페어 분석 및 추천"""
        nonlocal current_top_pair
        
        print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 페어 분석 시작")
        print("-"*60)
        
        # 모든 페어 조합에 대해 빠른 백테스팅
        from itertools import combinations
        pairs = list(combinations(valid_symbols, 2))
        
        results = []
        for long_symbol, short_symbol in pairs[:50]:  # 상위 50개만 빠르게 테스트
            try:
                # 7일 데이터로 빠른 백테스팅
                stats = simple_trend_following_backtest(analyzer, long_symbol, short_symbol, '5m', 7)
                
                if stats and stats['total_trades'] >= 5:  # 최소 5번 이상 거래
                    # 종합 점수 계산
                    composite_score = (
                        stats['win_rate'] * 0.3 +
                        stats['total_profit'] * 0.4 +
                        stats['sharpe_ratio'] * 10 * 0.3
                    )
                    
                    results.append({
                        'pair': f"{long_symbol.split('/')[0]}/{short_symbol.split('/')[0]}",
                        'long_symbol': long_symbol,
                        'short_symbol': short_symbol,
                        'win_rate': stats['win_rate'],
                        'total_profit': stats['total_profit'],
                        'sharpe_ratio': stats['sharpe_ratio'],
                        'total_trades': stats['total_trades'],
                        'composite_score': composite_score
                    })
            except:
                continue
            
            time.sleep(0.1)  # API 제한
        
        if not results:
            print("❌ 추천할 페어가 없습니다.")
            return
        
        # 종합 점수 순으로 정렬
        results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # TOP 5 출력
        print("\n🏆 추천 페어 TOP 5")
        print("-"*60)
        for i, pair in enumerate(results[:5], 1):
            print(f"{i}. {pair['pair']}")
            print(f"   • 종합점수: {pair['composite_score']:.1f}")
            print(f"   • 승률: {pair['win_rate']:.1f}%")
            print(f"   • 총수익: {pair['total_profit']:.2f}%")
            print(f"   • 샤프: {pair['sharpe_ratio']:.2f}")
            print(f"   • 거래수: {pair['total_trades']}건")
        
        # 1위 페어 저장
        current_top_pair = results[0]
        
        # 텔레그램 알림
        second_score = f"{results[1]['composite_score']:.1f}" if len(results) > 1 else "N/A"
        third_score = f"{results[2]['composite_score']:.1f}" if len(results) > 2 else "N/A"
        
        # 현재 가격 정보 가져오기
        try:
            # 1위 페어의 현재 가격 가져오기
            long_ticker = analyzer.exchange.fetch_ticker(results[0]['long_symbol'])
            short_ticker = analyzer.exchange.fetch_ticker(results[0]['short_symbol'])
            current_ratio = long_ticker['last'] / short_ticker['last']
            
            # 최근 5분봉 데이터로 볼린저 밴드 계산
            long_ohlcv = analyzer.exchange.fetch_ohlcv(results[0]['long_symbol'], '5m', limit=20)
            short_ohlcv = analyzer.exchange.fetch_ohlcv(results[0]['short_symbol'], '5m', limit=20)
            
            if len(long_ohlcv) >= 20 and len(short_ohlcv) >= 20:
                ratios = [long_ohlcv[i][4] / short_ohlcv[i][4] for i in range(len(long_ohlcv))]
                ma20 = np.mean(ratios)
                std20 = np.std(ratios)
                upper_band = ma20 + (2 * std20)
                lower_band = ma20 - (2 * std20)
                
                # 현재 위치 판단
                position_pct = ((current_ratio - lower_band) / (upper_band - lower_band)) * 100
                
                if current_ratio > upper_band:
                    signal_status = "🔴 **과매수 구간 - SHORT 대기**"
                elif current_ratio < lower_band:
                    signal_status = "🟢 **과매도 구간 - LONG 대기**"
                else:
                    signal_status = f"🟡 중립 구간 ({position_pct:.0f}%)"
            else:
                ma20 = current_ratio
                upper_band = current_ratio * 1.02
                lower_band = current_ratio * 0.98
                signal_status = "⚪ 데이터 수집 중"
                
        except:
            current_ratio = 0
            ma20 = 0
            upper_band = 0
            lower_band = 0
            signal_status = "⚪ 가격 정보 없음"
            
        # 상세한 메세지 생성
        message = f"""🔥 **즉시 매매 가능 페어 TOP 5**
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')} KST

━━━━━━━━━━━━━━━━━━━━━━━━━━━
🥇 **1위: {results[0]['pair']}**
━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 **백테스트 성과 (7일)**
• 종합점수: {results[0]['composite_score']:.0f}점
• 승률: {results[0]['win_rate']:.0f}%
• 누적수익: {results[0]['total_profit']:+.1f}%
• 샤프비율: {results[0]['sharpe_ratio']:.2f}
• 거래횟수: {results[0]['total_trades']}회

💹 **현재 상태**
• 현재 비율: {current_ratio:.6f}
• 이동평균(20): {ma20:.6f}
• 상단밴드: {upper_band:.6f}
• 하단밴드: {lower_band:.6f}
• {signal_status}

📍 **매매 전략**
【LONG 진입】
• 조건: 비율 < {lower_band:.6f}
• 방법: {results[0]['long_symbol'].split('/')[0]} 매수 + {results[0]['short_symbol'].split('/')[0]} 매도
• 목표: 중심선 {ma20:.6f} 회귀

【SHORT 진입】
• 조건: 비율 > {upper_band:.6f}
• 방법: {results[0]['long_symbol'].split('/')[0]} 매도 + {results[0]['short_symbol'].split('/')[0]} 매수
• 목표: 중심선 {ma20:.6f} 회귀

⚠️ **리스크 관리**
• 포지션: 자본의 5% 이내
• 손절선: -2%
• 익절선: 중심선 도달 or +3%

━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 **나머지 추천 페어**
━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        # 2-5위 표시
        for i in range(1, min(5, len(results))):
            medals = ['🥈', '🥉', '4️⃣', '5️⃣']
            medal = medals[i-1] if i <= 4 else ''
            
            # 간단한 진입 추천
            entry_hint = "LONG" if i % 2 == 0 else "SHORT"
            
            message += f"""
{medal} **{results[i]['pair']}**
• 점수: {results[i]['composite_score']:.0f} | 승률: {results[i]['win_rate']:.0f}% | 수익: {results[i]['total_profit']:+.1f}%
• 추천: {entry_hint} 포지션 검토
"""
        
        message += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ **실시간 모니터링 중**
• 5분마다 진입 신호 체크
• 신호 발생시 즉시 알림
• 다음 분석: 5시간 후
━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        print(f"\n📱 텔레그램 전송 중...")
        
        # 텔레그램 전송
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(analyzer.send_telegram_message(message))
            print("✅ 텔레그램 전송 완료!")
        except Exception as e:
            print(f"⚠️ 텔레그램 전송 실패: {e}")
    
    def check_entry_signal():
        """1위 페어의 진입 신호 체크"""
        nonlocal last_signal_time
        
        if not current_top_pair:
            return
        
        try:
            # 최신 5분봉 데이터 가져오기
            long_ohlcv = analyzer.exchange.fetch_ohlcv(
                current_top_pair['long_symbol'], '5m', limit=50
            )
            short_ohlcv = analyzer.exchange.fetch_ohlcv(
                current_top_pair['short_symbol'], '5m', limit=50
            )
            
            if len(long_ohlcv) < 20 or len(short_ohlcv) < 20:
                return
            
            # DataFrame 생성
            df_long = pd.DataFrame(long_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_short = pd.DataFrame(short_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 비율 계산
            ratio = df_long['close'].values / df_short['close'].values
            
            # 볼린저 밴드
            ma = pd.Series(ratio).rolling(window=20).mean()
            std = pd.Series(ratio).rolling(window=20).std()
            upper = ma + (2 * std)
            lower = ma - (2 * std)
            
            current_ratio = ratio[-1]
            current_ma = ma.iloc[-1]
            current_upper = upper.iloc[-1]
            current_lower = lower.iloc[-1]
            
            # 진입 신호 체크
            signal = None
            if current_ratio > current_upper:
                signal = "SHORT"
            elif current_ratio < current_lower:
                signal = "LONG"
            
            if signal:
                # 마지막 신호로부터 최소 30분 경과 체크
                pair_key = current_top_pair['pair']
                current_time = datetime.now()
                
                if pair_key in last_signal_time:
                    time_diff = (current_time - last_signal_time[pair_key]).total_seconds() / 60
                    if time_diff < 30:
                        return
                
                # 진입 신호 알림
                alert_message = f"""
🚨 진입 신호 발생! 🚨
━━━━━━━━━━━━━━━━━━
페어: {current_top_pair['pair']}
신호: {signal} (Ratio)
현재 비율: {current_ratio:.6f}
이동평균: {current_ma:.6f}
상단밴드: {current_upper:.6f}
하단밴드: {current_lower:.6f}
시간: {current_time.strftime('%H:%M:%S')}
━━━━━━━━━━━━━━━━━━
"""
                print(alert_message)
                
                # Windows 알림음 재생
                try:
                    winsound.Beep(1000, 500)  # 1000Hz, 500ms
                    winsound.Beep(1500, 500)  # 1500Hz, 500ms
                except:
                    pass
                
                # 간결하고 명확한 진입 신호 메세지
                telegram_entry_message = f"""🚨 **즉시 진입 신호!**

🎯 **{current_top_pair['pair']}**
📈 **{signal} 포지션**

**매매 방법:**
{"• " + current_top_pair['long_symbol'].split('/')[0] + " 매수\n• " + current_top_pair['short_symbol'].split('/')[0] + " 매도" if signal == "LONG" else "• " + current_top_pair['long_symbol'].split('/')[0] + " 매도\n• " + current_top_pair['short_symbol'].split('/')[0] + " 매수"}

**현재 상태:**
• 비율: {current_ratio:.4f}
• 밴드: {current_lower:.4f} ~ {current_upper:.4f}
• 중심: {current_ma:.4f}

**리스크 관리:**
• 손절: -2%
• 목표: 중심선 {current_ma:.4f}
• 포지션: 자본 5%

📊 백테스트: 승률 {current_top_pair['win_rate']:.0f}% | 샤프 {current_top_pair['sharpe_ratio']:.1f}
⏰ {current_time.strftime('%H:%M:%S')}

⚡ **즉시 확인 필요!**"""
                
                try:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(analyzer.send_telegram_message(telegram_entry_message))
                    print("✅ 텔레그램 진입 신호 전송 완료!")
                except Exception as e:
                    print(f"⚠️ 텔레그램 전송 실패: {e}")
                
                # 마지막 신호 시간 업데이트
                last_signal_time[pair_key] = current_time
                
        except Exception as e:
            print(f"신호 체크 오류: {e}")
    
    # 스케줄 설정
    print("\n📅 스케줄 설정:")
    print("  • 09:00 - 페어 분석 및 추천")
    print("  • 14:00 - 페어 분석 및 추천")
    print("  • 21:00 - 페어 분석 및 추천")
    print("  • 매 5분 - 1위 페어 진입 신호 체크")
    
    # 정기 분석 스케줄
    schedule.every().day.at("09:00").do(analyze_and_recommend)
    schedule.every().day.at("14:00").do(analyze_and_recommend)
    schedule.every().day.at("21:00").do(analyze_and_recommend)
    
    # 진입 신호 체크 (5분마다)
    schedule.every(5).minutes.do(check_entry_signal)
    
    # 즉시 한 번 실행
    print("\n초기 분석 실행...")
    analyze_and_recommend()
    
    print("\n🔄 실시간 모니터링 시작...")
    print("종료하려면 Ctrl+C를 누르세요.")
    print("-"*70)
    
    # 메인 루프
    try:
        while True:
            schedule.run_pending()
            
            # 현재 1위 페어 상태 표시
            if current_top_pair:
                print(f"\r📍 모니터링 중: {current_top_pair['pair']} | "
                      f"다음 분석: {schedule.next_run().strftime('%H:%M:%S') if schedule.next_run() else 'N/A'}", 
                      end='', flush=True)
            
            time.sleep(60)  # 1분마다 체크
            
    except KeyboardInterrupt:
        print("\n\n모니터링 종료")

def run_trend_following_all_pairs(analyzer):
    """모든 페어에 대한 Trend Following 전략 백테스팅"""
    print("\n" + "="*70)
    print("📊 모든 페어 Trend Following 전략 백테스팅")
    print("="*70)
    
    from itertools import combinations
    
    # 주요 코인 목록 (시가총액 상위)
    major_coins = [
        'BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'SOL/USDT:USDT',
        'XRP/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'DOGE/USDT:USDT',
        'TRX/USDT:USDT', 'LINK/USDT:USDT', 'MATIC/USDT:USDT', 'DOT/USDT:USDT',
        'SHIB/USDT:USDT', 'UNI/USDT:USDT', 'LTC/USDT:USDT', 'ATOM/USDT:USDT',
        'NEAR/USDT:USDT', 'ARB/USDT:USDT', 'OP/USDT:USDT', 'APT/USDT:USDT',
        'FIL/USDT:USDT', 'IMX/USDT:USDT', 'INJ/USDT:USDT', 'SUI/USDT:USDT',
        'SEI/USDT:USDT', 'TIA/USDT:USDT', 'ORDI/USDT:USDT', 'WLD/USDT:USDT',
        'FTM/USDT:USDT', 'RUNE/USDT:USDT'
    ]
    
    # 마켓 로드
    if not analyzer.exchange.markets:
        analyzer.exchange.load_markets()
    
    # 유효한 심볼만 필터링
    valid_symbols = []
    for symbol in major_coins:
        if symbol in analyzer.exchange.markets:
            valid_symbols.append(symbol)
    
    print(f"\n✅ 분석할 코인 수: {len(valid_symbols)}")
    print(f"📈 생성 가능한 페어 수: {len(list(combinations(valid_symbols, 2)))}")
    
    # 사용자 입력 받기
    max_pairs = input("\n테스트할 최대 페어 수를 입력하세요 (기본값: 50, 전체: all): ").strip()
    if max_pairs.lower() == 'all':
        max_pairs = len(list(combinations(valid_symbols, 2)))
    else:
        try:
            max_pairs = int(max_pairs) if max_pairs else 50
        except:
            max_pairs = 50
    
    timeframe = input("타임프레임을 선택하세요 (5m/15m/1h/4h, 기본값: 1h): ").strip() or '1h'
    days = input("백테스팅 기간을 입력하세요 (일 단위, 기본값: 30): ").strip()
    try:
        days = int(days) if days else 30
    except:
        days = 30
    
    print(f"\n설정: {max_pairs}개 페어, {timeframe} 타임프레임, {days}일 기간")
    print("-"*70)
    
    # 모든 페어 조합 생성
    all_pairs = list(combinations(valid_symbols, 2))[:max_pairs]
    
    results = []
    processed = 0
    successful = 0
    failed = 0
    
    print("\n백테스팅 시작...")
    print("-"*70)
    
    for long_symbol, short_symbol in all_pairs:
        processed += 1
        pair_name = f"{long_symbol.split('/')[0]}/{short_symbol.split('/')[0]}"
        
        if processed % 10 == 0:
            print(f"\n진행 상황: {processed}/{max_pairs} 페어 처리됨 (성공: {successful}, 실패: {failed})")
        
        try:
            # 간단한 trend_following 백테스팅 사용
            stats = simple_trend_following_backtest(analyzer, long_symbol, short_symbol, timeframe, days)
            
            if stats:
                results.append({
                    'pair': pair_name,
                    'long_symbol': long_symbol,
                    'short_symbol': short_symbol,
                    'total_trades': stats['total_trades'],
                    'win_rate': stats['win_rate'],
                    'avg_profit': stats['avg_profit'],
                    'total_profit': stats['total_profit'],
                    'max_profit': stats['max_profit'],
                    'max_loss': stats['max_loss'],
                    'sharpe_ratio': stats['sharpe_ratio'],
                    'profit_factor': stats['profit_factor'],
                    'avg_hold_hours': stats['avg_hold_hours']
                })
                successful += 1
                print(f"  ✅ {pair_name}: {stats['total_trades']}건, 승률 {stats['win_rate']:.1f}%, 총수익 {stats['total_profit']:.2f}%")
            else:
                failed += 1
                if processed <= 5:
                    print(f"  ⚠️ {pair_name}: 데이터 부족 또는 거래 신호 없음")
                
        except Exception as e:
            failed += 1
            if processed <= 5:  # 처음 5개만 에러 표시
                print(f"  ❌ {pair_name}: 오류 - {str(e)}")
            
        # API 제한 방지
        time.sleep(0.5)
    
    # 결과 분석 및 출력
    if results:
        df_results = pd.DataFrame(results)
        
        # 총 수익률 기준 정렬
        df_results = df_results.sort_values('total_profit', ascending=False)
        
        print("\n" + "="*70)
        print("📊 전체 백테스팅 통계 (Trend Following 전략)")
        print("="*70)
        
        print(f"\n✅ 테스트 완료: {len(results)}개 페어 (총 {processed}개 중)")
        print(f"📈 평균 승률: {df_results['win_rate'].mean():.1f}%")
        print(f"💰 평균 수익률: {df_results['avg_profit'].mean():.2f}%")
        print(f"📊 평균 거래 횟수: {df_results['total_trades'].mean():.1f}건")
        print(f"⚡ 평균 샤프 비율: {df_results['sharpe_ratio'].mean():.2f}")
        
        # Profit Factor 무한대 제외하고 계산
        valid_pf = df_results[df_results['profit_factor'] != float('inf')]['profit_factor']
        if len(valid_pf) > 0:
            print(f"💎 평균 Profit Factor: {valid_pf.mean():.2f}")
        
        print(f"⏱️ 평균 보유 시간: {df_results['avg_hold_hours'].mean():.1f}시간")
        
        print("\n" + "-"*70)
        print("🏆 TOP 10 수익률 페어")
        print("-"*70)
        
        top10 = df_results.head(10)
        for idx, row in top10.iterrows():
            print(f"\n{row['pair']}")
            print(f"  • 총 수익: {row['total_profit']:.2f}%")
            print(f"  • 승률: {row['win_rate']:.1f}%")
            print(f"  • 거래 횟수: {row['total_trades']}건")
            print(f"  • 평균 수익: {row['avg_profit']:.2f}%")
            print(f"  • 샤프 비율: {row['sharpe_ratio']:.2f}")
            if row['profit_factor'] != float('inf') and row['profit_factor'] > 0:
                print(f"  • Profit Factor: {row['profit_factor']:.2f}")
        
        print("\n" + "-"*70)
        print("📉 WORST 5 수익률 페어")
        print("-"*70)
        
        bottom5 = df_results.tail(5)
        for idx, row in bottom5.iterrows():
            print(f"\n{row['pair']}")
            print(f"  • 총 손실: {row['total_profit']:.2f}%")
            print(f"  • 승률: {row['win_rate']:.1f}%")
            print(f"  • 거래 횟수: {row['total_trades']}건")
            print(f"  • 평균 손실: {row['avg_profit']:.2f}%")
        
        # 승률별 분포
        print("\n" + "-"*70)
        print("📊 승률 분포")
        print("-"*70)
        
        win_rate_bins = [0, 30, 50, 70, 100]
        win_rate_labels = ['0-30%', '30-50%', '50-70%', '70-100%']
        df_results['win_rate_category'] = pd.cut(df_results['win_rate'], bins=win_rate_bins, labels=win_rate_labels)
        win_rate_dist = df_results['win_rate_category'].value_counts().sort_index()
        
        for category, count in win_rate_dist.items():
            percentage = count / len(df_results) * 100
            bar = '█' * int(percentage / 2)
            print(f"  {category}: {count}개 페어 ({percentage:.1f}%) {bar}")
        
        # 수익률별 분포
        print("\n" + "-"*70)
        print("💰 총 수익률 분포")
        print("-"*70)
        
        profitable = df_results[df_results['total_profit'] > 0]
        unprofitable = df_results[df_results['total_profit'] <= 0]
        
        print(f"  수익 페어: {len(profitable)}개 ({len(profitable)/len(df_results)*100:.1f}%)")
        print(f"  손실 페어: {len(unprofitable)}개 ({len(unprofitable)/len(df_results)*100:.1f}%)")
        
        if len(profitable) > 0:
            print(f"  평균 수익 (수익 페어): {profitable['total_profit'].mean():.2f}%")
            print(f"  최대 수익: {profitable['total_profit'].max():.2f}%")
        if len(unprofitable) > 0:
            print(f"  평균 손실 (손실 페어): {unprofitable['total_profit'].mean():.2f}%")
            print(f"  최대 손실: {unprofitable['total_profit'].min():.2f}%")
        
        # 거래 빈도별 분석
        print("\n" + "-"*70)
        print("📈 거래 빈도별 성과")
        print("-"*70)
        
        trade_bins = [0, 5, 10, 20, 50, 1000]
        trade_labels = ['1-5회', '6-10회', '11-20회', '21-50회', '50회+']
        df_results['trade_category'] = pd.cut(df_results['total_trades'], bins=trade_bins, labels=trade_labels)
        
        for category in trade_labels:
            category_data = df_results[df_results['trade_category'] == category]
            if len(category_data) > 0:
                print(f"  {category}: {len(category_data)}개 페어, 평균 수익 {category_data['avg_profit'].mean():.2f}%, 평균 승률 {category_data['win_rate'].mean():.1f}%")
        
        # CSV 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trend_following_all_pairs_{timeframe}_{days}d_{timestamp}.csv"
        df_results.to_csv(filename, index=False)
        print(f"\n💾 결과 저장됨: {filename}")
        
        # 시각화 생성
        create_trend_following_visualizations(df_results, timeframe, days)
        
        # 추천 페어 선정
        print("\n" + "="*70)
        print("🌟 추천 페어 (종합 점수 기준)")
        print("="*70)
        
        # 종합 점수 계산
        df_results['composite_score'] = (
            df_results['win_rate'] * 0.3 +  # 승률 30%
            df_results['total_profit'] * 0.4 +  # 총 수익 40%
            df_results['sharpe_ratio'] * 10 * 0.3  # 샤프 비율 30%
        )
        
        df_results = df_results.sort_values('composite_score', ascending=False)
        recommended = df_results.head(5)
        
        for idx, row in recommended.iterrows():
            print(f"\n⭐ {row['pair']}")
            print(f"   종합 점수: {row['composite_score']:.1f}")
            print(f"   총 수익: {row['total_profit']:.2f}%, 승률: {row['win_rate']:.1f}%, 샤프: {row['sharpe_ratio']:.2f}")
        
    else:
        print("\n❌ 백테스팅 결과가 없습니다.")
    
    return results

def run_comprehensive_backtest(analyzer):
    """대규모 통계 백테스팅"""
    print("\n📊 대규모 통계 백테스팅 시작...")
    print("="*60)
    
    # 1. 상위 200개 코인 가져오기
    print("\n1. 시총 상위 200개 코인 수집 중...")
    top_coins = analyzer.get_top_coins_by_market_cap(200)
    
    if len(top_coins) < 50:
        print(f"  ⚠️ 코인 데이터 부족: {len(top_coins)}개만 발견")
        return
    
    print(f"  ✅ {len(top_coins)}개 코인 수집 완료")
    
    # 2. 섹터별 페어 생성
    print("\n2. 섹터별 페어 생성 중...")
    sector_pairs = analyzer.create_pairs_by_sector(top_coins)
    
    total_pairs = sum(len(pairs) for pairs in sector_pairs.values())
    print(f"  ✅ 총 {total_pairs}개 페어 생성")
    
    # 3. 모든 페어 백테스팅
    print("\n3. 대규모 백테스팅 실행 중...")
    all_results = []
    timeframes = ['5m', '15m', '1h', '4h']  # 여러 타임프레임 테스트
    
    pair_count = 0
    max_pairs = 100  # 테스트할 최대 페어 수
    
    for sector, pairs in sector_pairs.items():
        for pair in pairs[:10]:  # 각 섹터에서 최대 10개
            if pair_count >= max_pairs:
                break
                
            pair_name = pair['pair_name']
            print(f"  [{pair_count+1}/{max_pairs}] {pair_name} ({sector})...")
            
            for tf in timeframes:
                try:
                    # 짧은 기간 백테스팅 (3일)
                    backtest = analyzer.backtest_entry_strategies(
                        pair['long']['symbol'], 
                        pair['short']['symbol'], 
                        tf, 
                        3
                    )
                    
                    if backtest and 'backtest_results' in backtest:
                        for strategy, perf in backtest['backtest_results'].items():
                            if perf['total_trades'] > 0:
                                all_results.append({
                                    'pair': pair_name,
                                    'sector': sector,
                                    'timeframe': tf,
                                    'strategy': strategy,
                                    'trades': perf['total_trades'],
                                    'win_rate': perf['win_rate'],
                                    'avg_profit': perf['avg_profit'],
                                    'sharpe': perf['sharpe_ratio'],
                                    'profit_factor': perf.get('profit_factor', 0),
                                    'correlation': pair.get('correlation', 0)
                                })
                    
                    time.sleep(0.5)  # API 제한
                    
                except Exception as e:
                    print(f"    ⚠️ {pair_name} {tf} 오류: {e}")
                    continue
            
            pair_count += 1
    
    # 4. 결과 분석 및 저장
    if not all_results:
        print("\n⚠️ 백테스팅 결과가 없습니다.")
        return
    
    df_results = pd.DataFrame(all_results)
    
    # CSV 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"comprehensive_backtest_{timestamp}.csv"
    df_results.to_csv(csv_filename, index=False)
    print(f"\n4. 결과 저장: {csv_filename}")
    
    # 5. 통계 분석 시각화
    print("\n5. 통계 분석 시각화 생성 중...")
    visualize_statistical_analysis(df_results)
    
    return df_results

def visualize_statistical_analysis(df_results):
    """통계 분석 시각화"""
    try:
        # Plotly 대시보드 생성
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '전략별 평균 승률',
                '타임프레임별 평균 수익',
                '전략 x 타임프레임 히트맵',
                '섹터별 성과',
                '샤프비율 분포',
                'Profit Factor 분포'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'heatmap'}, {'type': 'bar'}],
                [{'type': 'violin'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.15
        )
        
        # 1. 전략별 평균 승률
        strategy_winrate = df_results.groupby('strategy')['win_rate'].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=strategy_winrate.index, y=strategy_winrate.values,
                  name='승률', marker_color='green'),
            row=1, col=1
        )
        
        # 2. 타임프레임별 평균 수익
        tf_profit = df_results.groupby('timeframe')['avg_profit'].mean().sort_values(ascending=False)
        colors = ['red' if x < 0 else 'blue' for x in tf_profit.values]
        fig.add_trace(
            go.Bar(x=tf_profit.index, y=tf_profit.values,
                  name='수익', marker_color=colors),
            row=1, col=2
        )
        
        # 3. 전략 x 타임프레임 히트맵
        heatmap_data = df_results.pivot_table(
            values='avg_profit',
            index='strategy',
            columns='timeframe',
            aggfunc='mean'
        )
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='RdYlGn',
                text=np.round(heatmap_data.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ),
            row=2, col=1
        )
        
        # 4. 섹터별 성과
        sector_perf = df_results.groupby('sector')['avg_profit'].mean().sort_values(ascending=False)[:10]
        fig.add_trace(
            go.Bar(x=sector_perf.values, y=sector_perf.index,
                  orientation='h', name='섹터 수익',
                  marker_color='purple'),
            row=2, col=2
        )
        
        # 5. 샤프비율 분포 (전략별)
        for strategy in df_results['strategy'].unique():
            strategy_data = df_results[df_results['strategy'] == strategy]['sharpe']
            fig.add_trace(
                go.Violin(y=strategy_data, name=strategy,
                         box_visible=True, meanline_visible=True),
                row=3, col=1
            )
        
        # 6. Profit Factor 산점도
        fig.add_trace(
            go.Scatter(
                x=df_results['win_rate'],
                y=df_results['profit_factor'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=df_results['avg_profit'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Avg Profit")
                ),
                text=df_results['strategy'],
                hovertemplate='%{text}<br>Win Rate: %{x:.1f}%<br>PF: %{y:.2f}'
            ),
            row=3, col=2
        )
        
        # 레이아웃 설정
        fig.update_layout(
            title='📊 대규모 백테스팅 통계 분석',
            height=1200,
            showlegend=True
        )
        
        # 축 레이블
        fig.update_xaxes(title_text="전략", row=1, col=1)
        fig.update_xaxes(title_text="타임프레임", row=1, col=2)
        fig.update_xaxes(title_text="타임프레임", row=2, col=1)
        fig.update_xaxes(title_text="평균 수익 (%)", row=2, col=2)
        fig.update_xaxes(title_text="승률 (%)", row=3, col=2)
        
        fig.update_yaxes(title_text="승률 (%)", row=1, col=1)
        fig.update_yaxes(title_text="수익 (%)", row=1, col=2)
        fig.update_yaxes(title_text="전략", row=2, col=1)
        fig.update_yaxes(title_text="섹터", row=2, col=2)
        fig.update_yaxes(title_text="샤프비율", row=3, col=1)
        fig.update_yaxes(title_text="Profit Factor", row=3, col=2)
        
        # HTML 파일로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"statistical_analysis_{timestamp}.html"
        fig.write_html(filename)
        print(f"  ✅ 통계 분석 차트 저장: {filename}")
        
        # 핵심 통계 출력
        print("\n🏆 === 핵심 통계 === 🏆")
        
        # 최고 전략
        best_strategy = df_results.groupby('strategy').agg({
            'win_rate': 'mean',
            'avg_profit': 'mean',
            'sharpe': 'mean',
            'trades': 'sum'
        }).sort_values('avg_profit', ascending=False)
        
        print("\n[최고 전략 TOP 3] (수수료 0.08% 차감 후)")
        for idx, (strat, row) in enumerate(best_strategy.head(3).iterrows(), 1):
            print(f"{idx}. {strat}:")
            print(f"   승률: {row['win_rate']:.1f}% | 평균 순수익: {row['avg_profit']:.2f}% | 샤프: {row['sharpe']:.2f}")
            print(f"   거래 {row['trades']:.0f}건 | 예상 월 수익률: {row['avg_profit'] * row['trades'] / 30:.1f}%")
        
        # 최적 타임프레임
        best_tf = df_results.groupby('timeframe').agg({
            'win_rate': 'mean',
            'avg_profit': 'mean',
            'trades': 'sum'
        }).sort_values('avg_profit', ascending=False)
        
        print("\n[최적 타임프레임]")
        for idx, (tf, row) in enumerate(best_tf.iterrows(), 1):
            print(f"{idx}. {tf}: 승률 {row['win_rate']:.1f}% | 수익 {row['avg_profit']:.2f}% | 거래 {int(row['trades'])}건")
        
        # 최고 섹터
        best_sector = df_results.groupby('sector').agg({
            'avg_profit': 'mean',
            'trades': 'count'
        }).sort_values('avg_profit', ascending=False)
        
        print("\n[최고 성과 섹터 TOP 5]")
        for idx, (sector, row) in enumerate(best_sector.head(5).iterrows(), 1):
            print(f"{idx}. {sector}: 평균 {row['avg_profit']:.2f}% ({int(row['trades'])}건)")
        
        # 추천 조합
        print("\n🔥 === 추천 조합 === 🔥")
        
        # trend_following + 15m 조합 찾기
        optimal = df_results[
            (df_results['strategy'] == 'trend_following') & 
            (df_results['timeframe'].isin(['15m', '1h']))
        ]
        
        if not optimal.empty:
            avg_profit = optimal['avg_profit'].mean()
            win_rate = optimal['win_rate'].mean()
            print(f"\n✅ Trend Following + 15m/1h 조합:")
            print(f"   평균 수익: {avg_profit:.2f}%")
            print(f"   승률: {win_rate:.1f}%")
            print(f"   추천: 1H에서 추세 확인 → 15M에서 눌림목 진입")
        
        return fig
        
    except Exception as e:
        print(f"\n⚠️ 시각화 오류: {e}")
        return None

def run_visual_backtest(analyzer):
    """시각화된 백테스팅 실행"""
    print("\n시각화 백테스팅 분석 시작...")
    print("="*60)
    
    # 테스트할 페어
    test_pairs = [
        ('ETH/USDT:USDT', 'SOL/USDT:USDT'),
        ('BTC/USDT:USDT', 'ETH/USDT:USDT'),
        ('ARB/USDT:USDT', 'OP/USDT:USDT')
    ]
    
    for long_sym, short_sym in test_pairs:
        try:
            analyzer.visualize_backtest_results(long_sym, short_sym, '1h', 7)
            time.sleep(2)  # API 제한 방지
        except Exception as e:
            print(f"{long_sym}/{short_sym} 오류: {e}")
            continue
    
    print("\n모든 차트가 HTML 파일로 저장되었습니다.")
    print("브라우저에서 파일을 열어 상세 차트를 확인하세요.")

def run_backtest_analysis(analyzer):
    """백테스팅 분석 실행 (개선)"""
    print("\n백테스팅 분석 시작...")
    print("="*60)
    
    # 상위 코인 가져오기
    top_coins = analyzer.get_top_coins_by_market_cap(30)
    
    # 더 많은 테스트 페어 추가
    test_pairs = [
        ('ETH/USDT:USDT', 'SOL/USDT:USDT'),
        ('BTC/USDT:USDT', 'ETH/USDT:USDT'),
        ('AVAX/USDT:USDT', 'NEAR/USDT:USDT'),
        ('ARB/USDT:USDT', 'OP/USDT:USDT'),
        ('MATIC/USDT:USDT', 'AVAX/USDT:USDT')
    ]
    
    results = []
    for long_sym, short_sym in test_pairs:
        print(f"\n{long_sym}/{short_sym} 백테스팅...")
        try:
            # 더 긴 기간으로 테스트 (7일)
            backtest = analyzer.backtest_entry_strategies(long_sym, short_sym, '1h', 7)
        except Exception as e:
            print(f"  오류: {e}")
            continue
        
        if backtest and 'backtest_results' in backtest:
            print(f"최적 전략: {backtest['best_strategy']}")
            perf = backtest['best_performance']
            
            # 실제 투자 결과 출력
            print(f"\n💰 실제 투자 시뮬레이션 (초기자본: ${perf.get('initial_capital', 100000):,})")
            print(f"  - 최종 자본: ${perf.get('final_capital', 100000):,.2f}")
            print(f"  - 총 수익: ${perf.get('total_return_usd', 0):,.2f} ({perf.get('total_return_pct', 0):.2f}%)")
            print(f"  - 연환산 수익률: {perf.get('annualized_return', 0):.2f}%")
            print(f"  - 총 수수료: ${perf.get('total_fees_paid', 0):,.2f}")
            
            print(f"\n📊 거래 통계")
            print(f"  - 거래횟수: {perf['total_trades']}회 (일평균 {perf.get('trades_per_day', 0):.1f}회)")
            print(f"  - 평균 보유시간: {perf.get('avg_hold_hours', 0):.1f}시간")
            print(f"  - 승률: {perf['win_rate']:.1f}% (승: {perf['win_count']}회, 패: {perf['loss_count']}회)")
            print(f"  - 평균 수익(수수료 차감): {perf['avg_profit']:.2f}%")
            print(f"  - 최대 수익: {perf['max_profit']:.2f}%")
            print(f"  - 최대 손실: {perf['max_loss']:.2f}%")
            print(f"  - 샤프비율: {perf['sharpe_ratio']:.2f}")
            print(f"  - Profit Factor: {perf.get('profit_factor', 0):.2f}")
            
            # 모든 전략 간단 비교
            print("\n  모든 전략 성과:")
            for strat_name, strat_perf in backtest['backtest_results'].items():
                if strat_perf['total_trades'] > 0:
                    print(f"    {strat_name}: {strat_perf['total_trades']}건, "
                          f"승률 {strat_perf['win_rate']:.1f}%, "
                          f"평균 {strat_perf['avg_profit']:.2f}%")
            
            results.append({
                'pair': f"{long_sym}/{short_sym}",
                'strategy': backtest['best_strategy'],
                'performance': perf
            })
    
    # 결과 저장
    if results:
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_results_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\n백테스팅 결과 저장: {filename}")
    
    return results
