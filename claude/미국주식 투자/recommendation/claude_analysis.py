"""
Claude 분석 결과 관리 모듈

Claude가 제공한 투자 의견을 저장/로드/표시한다.
"""

import json
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

# 분석 결과 스키마
ANALYSIS_SCHEMA = {
    "ticker": str,          # 종목 티커
    "rating": str,          # Strong Buy / Buy / Hold / Avoid
    "score": float,         # 정량 스코어 (0~100)
    "catalyst": str,        # 상승 동인
    "risk": str,            # 리스크 요인
    "holding_period": str,  # 추천 홀딩 기간
    "summary": str,         # 한줄 요약
    "sector": str,          # 섹터
}


def save_analysis(analyses: list[dict], analysis_date: str | None = None) -> Path:
    """
    Claude 분석 결과를 JSON으로 저장한다.

    Args:
        analyses: 분석 결과 리스트
        analysis_date: 날짜 (None이면 오늘)

    Returns:
        저장된 파일 경로
    """
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    target_date = analysis_date or date.today().isoformat()
    output_file = ANALYSIS_DIR / f"{target_date}_analysis.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(analyses, f, ensure_ascii=False, indent=2)

    return output_file


def load_analysis(analysis_date: str | None = None) -> list[dict]:
    """
    저장된 분석 결과를 로드한다.

    Args:
        analysis_date: 날짜 (None이면 오늘)

    Returns:
        분석 결과 리스트. 파일 없으면 빈 리스트.
    """
    target_date = analysis_date or date.today().isoformat()
    analysis_file = ANALYSIS_DIR / f"{target_date}_analysis.json"

    if not analysis_file.exists():
        return []

    with open(analysis_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_latest_analysis() -> tuple[list[dict], str]:
    """
    가장 최근 분석 결과를 찾아서 로드한다.

    Returns:
        (분석 결과 리스트, 날짜 문자열)
    """
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(ANALYSIS_DIR.glob("*_analysis.json"), reverse=True)

    if not files:
        return [], ""

    latest = files[0]
    date_str = latest.stem.replace("_analysis", "")

    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f), date_str


def format_analysis_prompt(scores_df, top_n: int = 30, news_data: dict | None = None) -> str:
    """
    Claude 분석을 요청하기 위한 프롬프트를 생성한다.

    Args:
        scores_df: 스코어 DataFrame (정렬된 상태)
        top_n: 분석할 상위 종목 수
        news_data: {ticker: [news_items]} 뉴스 데이터

    Returns:
        Claude에게 보낼 분석 요청 텍스트
    """
    top = scores_df.head(top_n)

    lines = [
        "# 종목 추천 분석 요청",
        "",
        f"아래 정량 스코어링 Top {top_n} 종목에 대해 투자 의견을 제공해주세요.",
        "투자 시계: 단기 스윙 (1~4주)",
        "",
        "## 정량 스코어 (0~100, 백분위 기반)",
        "",
        top[["rank", "ticker", "total_score", "momentum", "trend", "breakout", "valuation", "growth", "risk"]].to_string(index=False),
        "",
    ]

    if news_data:
        lines.append("## 최근 뉴스")
        lines.append("")
        for ticker in top["ticker"]:
            if ticker in news_data and news_data[ticker]:
                lines.append(f"### {ticker}")
                for item in news_data[ticker][:3]:
                    lines.append(f"- {item.get('title', '')}")
                lines.append("")

    lines.extend([
        "## 요청 형식",
        "",
        "각 종목에 대해 다음을 JSON 배열로 응답해주세요:",
        "```json",
        "[",
        "  {",
        '    "ticker": "AAPL",',
        '    "rating": "Strong Buy|Buy|Hold|Avoid",',
        '    "catalyst": "상승 동인 설명",',
        '    "risk": "리스크 요인",',
        '    "holding_period": "1~2주",',
        '    "summary": "한줄 투자 의견"',
        "  }",
        "]",
        "```",
    ])

    return "\n".join(lines)
