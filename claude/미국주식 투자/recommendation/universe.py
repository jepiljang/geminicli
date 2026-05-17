"""
미국 주식 유니버스 관리 모듈

전체 미국 상장 주식 티커 + 시가총액을 NASDAQ Stock Screener에서 수집.
- 출처: https://api.nasdaq.com/api/screener/stocks (무료/무인증)
- 거래소: NASDAQ + NYSE + AMEX 합산
- 시가총액 기준 상위 N개 필터링 지원
"""

import re
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_DIR = PROJECT_ROOT / "data" / "universe"
TICKERS_FILE = UNIVERSE_DIR / "tickers.parquet"

NASDAQ_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks"
NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
}

_TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}$")

# 이름에 다음 패턴이 있으면 비-보통주로 보고 제외:
#  - 우선주 마커: Preferred Stock, Cumulative, % Series
#  - 채권: Notes due, Senior Notes, Subordinated Notes, Debenture
#  - 파생: Warrant, Right(s), Unit(s)
#  - SPAC: Acquisition Corp, Merger Corp (합병 전 빈 껍데기)
#  - 폐쇄형펀드(CEF): Term Trust, Income Fund, Equity Fund, Opportunities Fund,
#                    Beneficial Interest
# 보존: Common Stock, American Depositary, Ordinary Shares (외국 ADR 살림)
_NAME_BLACKLIST = re.compile(
    r"Preferred Stock|Notes due|% Notes|Warrant|\bRights?\b|\bUnits?\b"
    r"|Convertible Debenture|Subordinated Notes|Senior Notes|Cumulative"
    r"|Acquisition Corp|Merger\s+\S*\s*Corp"
    r"|Term Trust|Income Fund|Equity Fund|Opportunities Fund"
    r"|Convertible Income|Beneficial Interest",
    re.IGNORECASE,
)
# Depositary Shares는 외국 ADR(American Depositary Shares)과 우선주 변형 두 가지가 섞임.
# American 접두어 없으면 우선주 변형으로 보고 제외.
_DEPOSITARY_PATTERN = re.compile(r"Depositary Shares", re.IGNORECASE)
_AMERICAN_DEPOSITARY_PATTERN = re.compile(r"American Depositary", re.IGNORECASE)


def _parse_market_cap(raw: str | None) -> float:
    if not raw:
        return 0.0
    try:
        return float(raw.replace(",", "").replace("$", "").strip())
    except (ValueError, AttributeError):
        return 0.0


def fetch_all_tickers() -> pd.DataFrame:
    """
    NASDAQ Screener에서 전체 미국 상장 종목을 가져온다.

    Returns:
        DataFrame columns: [ticker, name, market_cap, sector, industry, country, ipo_year]
    """
    params = {"tableonly": "true", "limit": 25000, "download": "true"}
    resp = requests.get(NASDAQ_SCREENER_URL, params=params, headers=NASDAQ_HEADERS, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    rows = payload.get("data", {}).get("rows", [])
    if not rows:
        raise RuntimeError("NASDAQ screener returned empty rows")

    records = []
    for r in rows:
        records.append({
            "ticker": (r.get("symbol") or "").upper().strip(),
            "name": r.get("name", ""),
            "market_cap": _parse_market_cap(r.get("marketCap")),
            "sector": r.get("sector", ""),
            "industry": r.get("industry", ""),
            "country": r.get("country", ""),
            "ipo_year": r.get("ipoyear", ""),
        })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["ticker"])
    return df


def filter_investable(df: pd.DataFrame) -> pd.DataFrame:
    """
    투자 불가능한 종목 제외:
    - 티커가 1~5자 영문 대문자가 아님 (특수문자 포함된 우선주 등)
    - 시가총액이 0 (정보 없거나 거래 중단)
    - 이름에 Preferred/Notes/Warrant/Right/Unit 등 비-보통주 패턴 포함
    """
    name = df["name"].fillna("")
    mask_ticker = df["ticker"].str.match(_TICKER_PATTERN)
    mask_mcap = df["market_cap"] > 0
    mask_name = ~name.str.contains(_NAME_BLACKLIST)
    # Depositary Shares 처리: American Depositary가 아니면 제외 (우선주 변형)
    has_depositary = name.str.contains(_DEPOSITARY_PATTERN)
    has_american = name.str.contains(_AMERICAN_DEPOSITARY_PATTERN)
    mask_depositary = ~(has_depositary & ~has_american)
    return df[mask_ticker & mask_mcap & mask_name & mask_depositary].reset_index(drop=True)


def get_universe(
    force_refresh: bool = False,
    max_age_days: int = 7,
    top_n_by_mcap: int | None = None,
    min_market_cap: float = 0.0,
) -> pd.DataFrame:
    """
    유니버스 티커 목록을 가져온다. 캐시가 있으면 재사용.

    Args:
        force_refresh: True이면 캐시 무시하고 새로 다운로드
        max_age_days: 캐시 유효 기간 (일)
        top_n_by_mcap: 시가총액 상위 N개로 제한 (None이면 전체)
        min_market_cap: 최소 시가총액 (달러). 0이면 미적용

    Returns:
        시가총액 내림차순으로 정렬된 투자 가능 종목 DataFrame
    """
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)

    df: pd.DataFrame | None = None
    if not force_refresh and TICKERS_FILE.exists():
        mod_time = datetime.fromtimestamp(TICKERS_FILE.stat().st_mtime)
        if datetime.now() - mod_time < timedelta(days=max_age_days):
            df = pd.read_parquet(TICKERS_FILE)

    if df is None:
        print("NASDAQ Screener에서 티커 목록 다운로드 중...")
        df = fetch_all_tickers()
        df.to_parquet(TICKERS_FILE, index=False)
        print(f"유니버스 캐시 저장: {len(df)}개 원본 → {TICKERS_FILE}")

    # 필터는 캐시 적중 시에도 매번 적용 (블랙리스트가 바뀌어도 즉시 반영되도록)
    df = filter_investable(df)
    df = df.sort_values("market_cap", ascending=False).reset_index(drop=True)

    if min_market_cap > 0:
        df = df[df["market_cap"] >= min_market_cap].reset_index(drop=True)
    if top_n_by_mcap is not None and top_n_by_mcap > 0:
        df = df.head(top_n_by_mcap).reset_index(drop=True)

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="유니버스 조회/갱신")
    parser.add_argument("--refresh", action="store_true", help="캐시 무시하고 재다운로드")
    parser.add_argument("--top-mcap", type=int, default=None, help="시가총액 상위 N개")
    parser.add_argument("--min-mcap", type=float, default=0.0, help="최소 시가총액 (USD)")
    args = parser.parse_args()

    universe = get_universe(
        force_refresh=args.refresh,
        top_n_by_mcap=args.top_mcap,
        min_market_cap=args.min_mcap,
    )
    print(f"\n결과: {len(universe)}개 종목")
    print(universe.head(20)[["ticker", "name", "market_cap", "sector"]].to_string())
    if len(universe) >= 1:
        print(f"\n시총 1위: {universe.iloc[0]['ticker']} ${universe.iloc[0]['market_cap']:,.0f}")
        print(f"시총 마지막: {universe.iloc[-1]['ticker']} ${universe.iloc[-1]['market_cap']:,.0f}")
