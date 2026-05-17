"""
뉴스 수집 모듈

Top N 종목에 대해 최신 뉴스를 수집하고 캐싱한다.
소스: yfinance news (기본) + 추후 확장 가능
"""

import json
import yfinance as yf
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEWS_DIR = PROJECT_ROOT / "data" / "news"


def fetch_news_yfinance(ticker: str, max_items: int = 5) -> list[dict]:
    """
    yfinance에서 종목 뉴스 헤드라인을 가져온다.

    Args:
        ticker: 종목 티커
        max_items: 최대 뉴스 수

    Returns:
        [{"title": str, "publisher": str, "link": str}]
    """
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        results = []
        for item in news[:max_items]:
            results.append({
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "link": item.get("link", ""),
                "published": item.get("providerPublishTime", ""),
            })
        return results
    except Exception:
        return []


def fetch_news_for_tickers(
    tickers: list[str],
    max_items_per_ticker: int = 5,
    use_cache: bool = True,
) -> dict[str, list[dict]]:
    """
    여러 종목의 뉴스를 수집한다.

    Args:
        tickers: 티커 리스트
        max_items_per_ticker: 종목당 최대 뉴스 수
        use_cache: 캐시 사용 여부

    Returns:
        {ticker: [news_items]}
    """
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    cache_file = NEWS_DIR / f"{today}_news.json"

    # 캐시 로드
    cached = {}
    if use_cache and cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)

    results = {}
    for ticker in tickers:
        if use_cache and ticker in cached:
            results[ticker] = cached[ticker]
        else:
            news = fetch_news_yfinance(ticker, max_items_per_ticker)
            results[ticker] = news

    # 캐시 저장
    all_news = {**cached, **results}
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(all_news, f, ensure_ascii=False, indent=2, default=str)

    return results


def format_news_for_analysis(ticker: str, news_items: list[dict]) -> str:
    """
    Claude 분석을 위한 뉴스 포맷팅.

    Returns:
        마크다운 형식의 뉴스 요약
    """
    if not news_items:
        return f"**{ticker}**: 최근 뉴스 없음"

    lines = [f"**{ticker} 최근 뉴스:**"]
    for i, item in enumerate(news_items, 1):
        title = item.get("title", "제목 없음")
        publisher = item.get("publisher", "")
        lines.append(f"  {i}. {title} ({publisher})")

    return "\n".join(lines)
