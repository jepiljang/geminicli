import yfinance as yf


def get_valuation(info: dict) -> dict:
    """Valuation: PER, PBR, PSR, PEG, EV/EBITDA"""
    return {
        "per_trailing": info.get("trailingPE"),
        "per_forward": info.get("forwardPE"),
        "pbr": info.get("priceToBook"),
        "psr": info.get("priceToSalesTrailing12Months"),
        "peg": info.get("trailingPegRatio") or info.get("pegRatio"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
    }


def get_profitability(info: dict) -> dict:
    """Profitability: ROE, ROA, Operating Margin, Profit Margin"""
    return {
        "roe": info.get("returnOnEquity"),
        "roa": info.get("returnOnAssets"),
        "operating_margin": info.get("operatingMargins"),
        "profit_margin": info.get("profitMargins"),
    }


def get_growth(info: dict) -> dict:
    """Growth: EPS, Revenue Growth, Earnings Growth"""
    return {
        "eps_trailing": info.get("trailingEps"),
        "eps_forward": info.get("forwardEps"),
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
    }


def get_financial_health(info: dict) -> dict:
    """Financial Health: Debt/Equity, Current/Quick Ratio"""
    return {
        "debt_to_equity": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "quick_ratio": info.get("quickRatio"),
    }


def get_dividend(info: dict) -> dict:
    """Dividend: Yield, Payout Ratio"""
    return {
        "dividend_yield": info.get("dividendYield"),
        "payout_ratio": info.get("payoutRatio"),
    }


def get_market_info(info: dict) -> dict:
    """Market: Market Cap, Beta, 52-week High/Low"""
    return {
        "market_cap": info.get("marketCap"),
        "beta": info.get("beta"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
    }


def get_fundamentals(ticker: str) -> dict:
    """
    미국 주식 펀더멘털 지표를 일괄 수집.

    Args:
        ticker: 종목 티커 (예: 'AAPL')

    Returns:
        카테고리별 펀더멘털 지표 dict. 실패 시 빈 dict.
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        print(f"Error fetching fundamentals for {ticker}: {e}")
        return {}

    if not info or len(info) < 5:
        print(f"Warning: {ticker} 펀더멘털 정보 없음")
        return {}

    return {
        "ticker": ticker,
        "valuation": get_valuation(info),
        "profitability": get_profitability(info),
        "growth": get_growth(info),
        "financial_health": get_financial_health(info),
        "dividend": get_dividend(info),
        "market": get_market_info(info),
    }
