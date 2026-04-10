"""
Telegram 알림 모듈.

사용:
    from notification.telegram import send_message, notify_signal

    send_message("백테스트 완료")
    notify_signal("AAPL", "BUY", price=175.50, score=0.45)

CLI 테스트:
    python notification/telegram.py --test
"""
import argparse
import os
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

# 프로젝트 루트의 .env 로드
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _get_credentials() -> tuple[str, str]:
    """환경변수에서 BOT_TOKEN, CHAT_ID 읽기."""
    bot_token = os.getenv("BOT_TOKEN")
    chat_id = os.getenv("CHAT_ID")
    if not bot_token or not chat_id:
        raise RuntimeError(
            "BOT_TOKEN 또는 CHAT_ID가 .env에 설정되지 않았습니다."
        )
    return bot_token, chat_id


def _mask_sensitive(text: str) -> str:
    """에러 메시지에서 토큰 마스킹 (CLAUDE.md 보안규칙)."""
    try:
        bot_token, _ = _get_credentials()
        if bot_token and bot_token in text:
            text = text.replace(bot_token, "***MASKED***")
    except Exception:
        pass
    return text


def send_message(
    message: str,
    parse_mode: str = "Markdown",
    disable_web_page_preview: bool = True,
) -> bool:
    """
    Telegram으로 메시지 전송.

    Args:
        message: 전송할 텍스트 (Markdown 형식 권장)
        parse_mode: 'Markdown' or 'HTML'
        disable_web_page_preview: 링크 미리보기 비활성화

    Returns:
        전송 성공 여부
    """
    try:
        bot_token, chat_id = _get_credentials()
    except RuntimeError as e:
        print(f"[Telegram] 설정 오류: {e}")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": parse_mode,
        "disable_web_page_preview": disable_web_page_preview,
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        # 민감정보 마스킹 후 출력
        print(f"[Telegram] 전송 실패: {_mask_sensitive(str(e))}")
        return False


def notify_signal(
    ticker: str,
    action: str,
    price: Optional[float] = None,
    score: Optional[float] = None,
    reason: Optional[str] = None,
) -> bool:
    """
    매매 시그널 알림.

    Args:
        ticker: 종목 티커 (예: 'AAPL')
        action: 'BUY', 'SELL', 'HOLD'
        price: 현재가
        score: 전략 스코어
        reason: 시그널 발생 이유

    Returns:
        전송 성공 여부
    """
    action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⏸️"}.get(action.upper(), "📊")

    lines = [f"{action_emoji} *{action.upper()} Signal*", f"종목: `{ticker}`"]
    if price is not None:
        lines.append(f"가격: *${price:.2f}*")
    if score is not None:
        lines.append(f"스코어: *{score:+.3f}*")
    if reason:
        lines.append(f"사유: {reason}")

    return send_message("\n".join(lines))


def notify_backtest_result(summary: dict, ticker: str, strategy_name: str) -> bool:
    """
    백테스트 결과 요약 알림.

    Args:
        summary: backtest.metrics.summarize() 결과 dict
        ticker: 종목 티커
        strategy_name: 전략 이름

    Returns:
        전송 성공 여부
    """
    # KPI 통과 여부
    sharpe_icon = "✅" if summary.get("sharpe_ratio", 0) > 1.0 else "❌"
    win_icon = "✅" if summary.get("win_rate", 0) > 0.55 else "❌"
    mdd_icon = "✅" if summary.get("max_drawdown", -1) > -0.20 else "❌"
    alpha = summary.get("alpha_vs_benchmark")
    alpha_icon = "✅" if (alpha is not None and alpha > 0) else "❌"

    def pct(v):
        return f"{v:.2%}" if isinstance(v, (int, float)) else "N/A"

    def num(v, fmt=".2f"):
        return format(v, fmt) if isinstance(v, (int, float)) else "N/A"

    lines = [
        f"📊 *백테스트 결과*",
        f"종목: `{ticker}` | 전략: `{strategy_name}`",
        "",
        f"총수익률: *{pct(summary.get('total_return'))}*",
        f"연수익률: *{pct(summary.get('annualized_return'))}*",
        f"{sharpe_icon} Sharpe: *{num(summary.get('sharpe_ratio'))}*",
        f"{win_icon} Win Rate: *{pct(summary.get('win_rate'))}*",
        f"{mdd_icon} MDD: *{pct(summary.get('max_drawdown'))}*",
        f"거래: {summary.get('num_trades', 0)}회",
    ]

    if summary.get("benchmark_return") is not None:
        lines.extend([
            "",
            f"*vs SPY*",
            f"SPY 연수익률: {pct(summary.get('benchmark_return'))}",
            f"{alpha_icon} Alpha: *{pct(alpha)}*",
            f"초과수익: *{pct(summary.get('excess_return'))}*",
        ])

    return send_message("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Telegram 알림 테스트")
    parser.add_argument("--test", action="store_true", help="테스트 메시지 전송")
    parser.add_argument("--message", type=str, help="커스텀 메시지 전송")
    args = parser.parse_args()

    if args.test:
        ok = send_message("🤖 Telegram 연동 테스트 성공!")
        print("✅ 전송 성공" if ok else "❌ 전송 실패")
    elif args.message:
        ok = send_message(args.message)
        print("✅ 전송 성공" if ok else "❌ 전송 실패")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
