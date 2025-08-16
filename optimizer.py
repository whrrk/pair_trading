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
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from concurrent.futures import ThreadPoolExecutor

class ParameterOptimizer:
    """파라미터 최적화 클래스"""
    
    def __init__(self):
        self.best_params = {}
        
    def optimize_with_grid_search(self, historical_data, strategy='trend_following'):
        """Grid Search를 통한 파라미터 최적화"""
        
        # 파라미터 범위 정의
        param_grid = {
            'rsi_oversold': [15, 20, 25, 30, 35],
            'rsi_overbought': [65, 70, 75, 80, 85],
            'ema_short': [5, 8, 9, 12],
            'ema_long': [20, 21, 26, 34],
            'bb_period': [14, 20, 26],
            'bb_std': [1.5, 2.0, 2.5, 3.0],
            'adx_threshold': [20, 25, 30, 35],
            'zscore_threshold': [1.5, 2.0, 2.5, 3.0]
        }
        
        best_score = -np.inf
        best_params = {}
        
        # 간단한 Grid Search (실제로는 더 정교하게 구현 필요)
        for rsi_os in param_grid['rsi_oversold']:
            for rsi_ob in param_grid['rsi_overbought']:
                for bb_std in param_grid['bb_std']:
                    # 백테스팅으로 성과 측정
                    score = self.evaluate_parameters(
                        historical_data,
                        {'rsi_oversold': rsi_os, 'rsi_overbought': rsi_ob, 'bb_std': bb_std},
                        strategy
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'rsi_oversold': rsi_os,
                            'rsi_overbought': rsi_ob,
                            'bb_std': bb_std
                        }
        
        self.best_params[strategy] = best_params
        return best_params
    
    def evaluate_parameters(self, data, params, strategy):
        """파라미터 성과 평가"""
        # 간단한 샤프비율 계산 (실제 구현시 더 정교하게)
        returns = []
        
        for i in range(len(data) - 1):
            if strategy == 'trend_following':
                # RSI 기반 진입 신호
                if data['rsi'][i] < params['rsi_oversold']:
                    returns.append(data['returns'][i+1])
                elif data['rsi'][i] > params['rsi_overbought']:
                    returns.append(-data['returns'][i+1])
        
        if len(returns) > 0:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6)
            return sharpe
        return 0
    
    def get_dynamic_parameters(self, current_volatility, base_params=None):
        """현재 시장 상황에 따른 동적 파라미터 조정"""
        
        if base_params is None:
            base_params = {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'bb_period': 20,
                'bb_std': 2.0,
                'ema_short': 9,
                'ema_long': 21,
                'adx_threshold': 25
            }
        
        adjusted_params = base_params.copy()
        
        # 변동성에 따른 조정
        if current_volatility > 0.05:  # 높은 변동성
            adjusted_params['rsi_oversold'] = 20
            adjusted_params['rsi_overbought'] = 80
            adjusted_params['bb_std'] = 2.5
            adjusted_params['adx_threshold'] = 30
        elif current_volatility < 0.02:  # 낮은 변동성
            adjusted_params['rsi_oversold'] = 35
            adjusted_params['rsi_overbought'] = 65
            adjusted_params['bb_std'] = 1.5
            adjusted_params['adx_threshold'] = 20
        
        return adjusted_params
    
    def walk_forward_optimization(self, data, window_size=60, test_size=30):
        """Walk-Forward Optimization"""
        results = []
        
        for i in range(0, len(data) - window_size - test_size, test_size):
            # Training window
            train_data = data[i:i+window_size]
            
            # Optimize on training data
            best_params = self.optimize_with_grid_search(train_data)
            
            # Test on out-of-sample data
            test_data = data[i+window_size:i+window_size+test_size]
            score = self.evaluate_parameters(test_data, best_params, 'trend_following')
            
            results.append({
                'params': best_params,
                'score': score,
                'period': i
            })
        
        return results