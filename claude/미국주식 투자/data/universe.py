"""
종목 유니버스 정의.

S&P500 상위 대형주 하드코딩 (survivorship bias 수용).
과거 편입/퇴출 데이터는 무시 — 현재 생존 종목만 사용.

주의: yfinance 1h 데이터는 ~730일 제한이므로
      50개 티커 × ~3500 bars ≈ 175K rows (수 분 소요).
"""
from typing import List


# 2026년 기준 S&P500 상위 50 (대략적 시총 순, 중복 방지)
SP500_TOP50: List[str] = [
    # Mega cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    # Tech/Semi
    "AVGO", "ORCL", "ADBE", "CRM", "AMD", "QCOM", "TXN", "INTU",
    # Financials
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS",
    # Healthcare
    "LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "ABT", "PFE",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "HD", "MCD", "NKE",
    # Industrial / Energy
    "XOM", "CVX", "CAT", "BA", "HON", "GE", "LMT",
    # Comm/Media
    "NFLX", "DIS",
    # Other
    "LIN",  # chemicals
    "ACN",  # consulting
]

assert len(SP500_TOP50) == 50, f"Expected 50 tickers, got {len(SP500_TOP50)}"


def get_sp500_top50() -> List[str]:
    """S&P500 상위 50개 티커 리스트 반환."""
    return list(SP500_TOP50)
