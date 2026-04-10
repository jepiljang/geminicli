import yfinance as yf
import pandas as pd
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# 프로젝트 루트 기준 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def fetch_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    yfinance로 OHLCV 데이터 수집.

    Args:
        ticker: 종목 티커 (예: 'AAPL')
        period: 데이터 기간 (예: '2y', '1y', '6mo')
        interval: 봉 간격 (예: '1d', '5m', '1h')
                  - 5m/1h 등 분봉은 최대 60일까지만 가능 (yfinance 제한)

    Returns:
        OHLCV DataFrame. 실패 시 빈 DataFrame.
    """
    try:
        data = yf.download(ticker, period=period, interval=interval, multi_level_index=False)
        if data.empty:
            print(f"Warning: {ticker} 데이터 없음 (period={period}, interval={interval})")
        return data
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()


def save_data(data: pd.DataFrame, ticker: str, period: str, interval: str = "1d"):
    """수집된 데이터를 CSV로 저장."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{ticker}_{period}_{interval}.csv"
    filepath = RAW_DATA_DIR / filename
    data.to_csv(filepath)
    print(f"Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="미국주식 OHLCV 데이터 수집 (yfinance)")
    parser.add_argument(
        "--ticker", type=str, required=True,
        help="종목 티커 (콤마 구분). 예: AAPL,MSFT,GOOGL"
    )
    parser.add_argument(
        "--period", type=str, default="2y",
        help="데이터 기간 (기본: 2y). 예: 1d, 5d, 1mo, 6mo, 1y, 2y, 5y, max"
    )
    parser.add_argument(
        "--interval", type=str, default="1d",
        help="봉 간격 (기본: 1d). 예: 5m, 15m, 1h, 1d, 1wk"
    )

    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.ticker.split(",")]
    if "SPY" not in tickers:
        tickers.append("SPY")

    # 분봉 데이터는 기간 제한 안내
    if args.interval != "1d" and args.period in ("2y", "5y", "10y", "max"):
        print(f"Warning: {args.interval} 간격은 최대 60일까지만 가능합니다. period를 '60d' 이하로 조정하세요.")
        return

    print(f"수집 대상: {', '.join(tickers)} | 기간: {args.period} | 간격: {args.interval}")

    for ticker in tqdm(tickers, desc="Fetching"):
        data = fetch_data(ticker, args.period, args.interval)
        if not data.empty:
            save_data(data, ticker, args.period, args.interval)
        else:
            print(f"Skip: {ticker} (데이터 없음)")


if __name__ == "__main__":
    main()
