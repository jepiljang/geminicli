"""
Streamlit ↔ Claude 분석 요청 큐.

Streamlit에서 종목 선택 + 버튼 → create_request()로 .md 요청 파일 생성.
사용자가 Claude 세션에 "AAPL 분석해줘" 등으로 요청 → list_pending() 으로
요청 파일 읽어 분석 수행 → merge_analysis_json() + save_report() 로 결과
저장 → mark_processed()로 요청 큐에서 빼낸다.
"""

import json
import shutil
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
REQUEST_DIR = ANALYSIS_DIR / "requests"
PROCESSED_DIR = REQUEST_DIR / "processed"
REPORT_DIR = ANALYSIS_DIR / "reports"


def _ensure_dirs() -> None:
    for d in (ANALYSIS_DIR, REQUEST_DIR, PROCESSED_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def create_request(
    ticker: str,
    score_row: pd.Series,
    news_items: list[dict] | None = None,
) -> Path:
    """종목 분석 요청 마크다운 파일을 큐에 추가한다.

    Returns:
        생성된 요청 파일 경로.
    """
    _ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REQUEST_DIR / f"{ticker}_{ts}.md"

    lines = [
        f"# 분석 요청: {ticker}",
        f"_요청 시각: {datetime.now().isoformat(timespec='seconds')}_",
        "",
        "## 정량 스코어",
        f"- **종합 점수**: {score_row.get('total_score', 'N/A')}",
        f"- **랭킹**: #{int(score_row.get('rank', 0))}",
        "",
        "### 팩터 분해 (0~100, 백분위)",
        f"- Momentum: {score_row.get('momentum', 'N/A')}",
        f"- Trend: {score_row.get('trend', 'N/A')}",
        f"- Breakout: {score_row.get('breakout', 'N/A')}",
        f"- Valuation: {score_row.get('valuation', 'N/A')}",
        f"- Growth: {score_row.get('growth', 'N/A')}",
        f"- Risk: {score_row.get('risk', 'N/A')}",
    ]
    if "exemplar_similarity" in score_row.index and pd.notna(score_row.get("exemplar_similarity")):
        lines.append(f"- Exemplar Similarity: {score_row['exemplar_similarity']:.1f} "
                     f"(닮은 모범: {score_row.get('best_match_id', 'N/A')})")

    lines.extend(["", "## 최근 뉴스"])
    if news_items:
        for i, item in enumerate(news_items, 1):
            title = item.get("title", "제목 없음")
            publisher = item.get("publisher", "")
            link = item.get("link", "")
            lines.append(f"{i}. **{title}** ({publisher})")
            if link:
                lines.append(f"   {link}")
    else:
        lines.append("_뉴스 없음 또는 수집 실패_")

    lines.extend([
        "",
        "---",
        "",
        "## 요청 사항",
        "",
        "위 종목에 대해 다음을 작성해 주세요:",
        "",
        "1. **등급**: Strong Buy / Buy / Hold / Avoid",
        "2. **카탈리스트**: 상승 동인 (2~3줄)",
        "3. **리스크**: 주의할 리스크 요인 (2~3줄)",
        "4. **홀딩 기간**: 추천 보유 기간 (예: 1~2주, 1개월)",
        "5. **한줄 요약**: 핵심 투자 의견",
        "6. **상세 리포트**: 차트 패턴 해석, 섹터 모멘텀, 진입/손절가 제안 등 자유 분석",
        "",
        "처리 완료 시 `recommendation.analysis_request.merge_analysis_json()`과 "
        "`save_report()`를 호출하고 `mark_processed()`로 큐에서 빼주세요.",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def list_pending(ticker: str | None = None) -> list[Path]:
    """대기 중인 요청 파일 목록. ticker 지정 시 해당 종목만."""
    _ensure_dirs()
    pattern = f"{ticker}_*.md" if ticker else "*.md"
    return sorted(REQUEST_DIR.glob(pattern), reverse=True)


def read_request(path: Path) -> str:
    """요청 파일 내용을 텍스트로 반환."""
    return path.read_text(encoding="utf-8")


def mark_processed(path: Path) -> Path:
    """요청 파일을 processed/로 이동."""
    _ensure_dirs()
    target = PROCESSED_DIR / path.name
    shutil.move(str(path), str(target))
    return target


def merge_analysis_json(ticker: str, analysis: dict, analysis_date: str | None = None) -> Path:
    """`data/analysis/{date}_analysis.json`에 해당 종목 항목을 추가/교체.

    같은 ticker가 이미 있으면 덮어쓴다.
    """
    _ensure_dirs()
    target_date = analysis_date or date.today().isoformat()
    path = ANALYSIS_DIR / f"{target_date}_analysis.json"

    items: list[dict] = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)

    items = [it for it in items if it.get("ticker") != ticker]
    item = {"ticker": ticker, **analysis}
    items.append(item)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return path


def save_report(ticker: str, markdown: str, report_date: str | None = None) -> Path:
    """상세 분석 리포트를 마크다운으로 저장."""
    _ensure_dirs()
    target_date = report_date or date.today().isoformat()
    path = REPORT_DIR / f"{ticker}_{target_date}.md"
    path.write_text(markdown, encoding="utf-8")
    return path
