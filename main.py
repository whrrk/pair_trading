"""
Bybit 페어 추세 강도 분석 시스템
- 시총 100위 이내 코인의 페어 분석
- 섹터별 최강 페어 선정
- 매일 오전 9시 텔레그램 알림
"""

import os
from telegram import Bot
from typing import List, Dict, Tuple
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
from analyzer import BybitPairTrendAnalyzer

def main():
    """메인 실행 함수"""
    print("Bybit 페어 추세 분석 시스템 (섹터별)")
    print("-" * 60)
    
    # 설정 파일 확인
    config_path = 'trend_config.json'
    
    # 환경변수 확인
    if not os.environ.get('TELEGRAM_BOT_TOKEN'):
        print("⚠️  TELEGRAM_BOT_TOKEN 환경변수가 설정되지 않았습니다.")
    if not os.environ.get('TELEGRAM_CHAT_ID'):
        print("⚠️  TELEGRAM_CHAT_ID 환경변수가 설정되지 않았습니다.")
    
    # 분석기 초기화
    analyzer = BybitPairTrendAnalyzer(config_path)
    
    # 실행 모드 선택
    print("\n실행 모드를 선택하세요:")
    print("1. 즉시 실행 (테스트)")
    print("2. 스케줄러 모드")
    print("3. 둘 다 (즉시 실행 후 스케줄러)")
    print("4. 백그라운드 실행 (Windows Service/Linux Daemon)")
    print("5. 백테스팅 분석")
    print("6. 시각화 백테스팅 (차트 생성)")
    print("7. 🔥 대규모 통계 백테스팅 (200개 코인)")
    print("8. 📊 모든 페어 Trend Following 전략 백테스팅")
    print("9. 🔔 실시간 추천 페어 모니터링 및 진입 알림")
    
    choice = input("선택 (1/2/3): ").strip()
    
    if choice == '1':
        analyzer.run_analysis_and_notify()
    elif choice == '2':
        analyzer.setup_scheduler(analyzer)
    elif choice == '3':
        analyzer.run_analysis_and_notify()
        analyzer.setup_scheduler(analyzer)
    elif choice == '4':
        print("\n백그라운드 실행 설정:")
        print("Windows: nssm 또는 Task Scheduler 사용")
        print("Linux: systemd service 또는 cron 사용")
        print("\n현재는 포그라운드로 실행합니다.")
        analyzer.setup_scheduler(analyzer)
    elif choice == '5':
        analyzer.run_backtest_analysis(analyzer)
    elif choice == '6':
        analyzer.run_visual_backtest(analyzer)
    elif choice == '7':
        analyzer.run_comprehensive_backtest(analyzer)
    elif choice == '8':
        analyzer.run_trend_following_all_pairs(analyzer)
    elif choice == '9':
        analyzer.run_realtime_monitoring(analyzer)
    else:
        print("잘못된 선택입니다.")

if __name__ == "__main__":
    # scipy, statsmodels 설치 확인
    try:
        import scipy
        import statsmodels
    except ImportError:
        print("필요한 패키지를 설치합니다...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "statsmodels"])
        print("설치 완료. 프로그램을 다시 시작해주세요.")
        sys.exit(0)
    
    main()