"""
yfinance 데이터 로컬 캐싱.

첫 호출: yfinance 네트워크 호출 → pickle로 저장
재호출: 파일이 max_age_hours 이내면 로컬 로드

Optuna 튜닝 시 매 trial마다 데이터 재다운로드를 피하기 위함.
"""
import pickle
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from data.fetcher import fetch_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "cache"


def _cache_path(ticker: str, period: str, interval: str) -> Path:
    safe_ticker = ticker.replace("/", "_").replace("^", "IDX_")
    return CACHE_DIR / f"{safe_ticker}_{period}_{interval}.pkl"


def cached_fetch(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    max_age_hours: float = 24.0,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    캐시 우선 fetch. 캐시가 있고 max_age_hours 이내면 로컬 로드,
    아니면 yfinance 호출 후 저장.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker, period, interval)

    if not force_refresh and path.exists():
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours < max_age_hours:
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"[cache] 로드 실패 {path.name}: {e} → 재다운로드")

    # Fresh fetch
    df = fetch_data(ticker, period, interval)
    if not df.empty:
        try:
            with open(path, "wb") as f:
                pickle.dump(df, f)
        except Exception as e:
            print(f"[cache] 저장 실패 {path.name}: {e}")
    return df
