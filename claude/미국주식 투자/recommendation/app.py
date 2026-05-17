"""
미국주식 종목 추천 Streamlit 대시보드

페이지 구성:
1. 메인: Top 추천 카드 (점수 + AI 의견)
2. 전체 랭킹: 전체 종목 테이블 (검색, 필터, 정렬)
3. 종목 상세: 차트 + 팩터 레이더차트 + Claude 의견 + 뉴스
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import subprocess
import sys as _sys
from pathlib import Path
from datetime import date, datetime

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recommendation.claude_analysis import load_analysis, get_latest_analysis
from recommendation.news_fetcher import fetch_news_for_tickers
from recommendation.exemplar.features import FEATURE_KEYS
from recommendation.exemplar.library import Exemplar, ExemplarLibrary, make_exemplar_id
from recommendation.exemplar.profile import build_profile, load_profile, save_profile
from recommendation.analysis_request import create_request, list_pending
from recommendation.news_fetcher import fetch_news_yfinance

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORES_DIR = PROJECT_ROOT / "data" / "scores"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
UNIVERSE_DIR = PROJECT_ROOT / "data" / "universe"
BACKTEST_DIR = PROJECT_ROOT / "data" / "backtest"
EXEMPLAR_DIR = PROJECT_ROOT / "data" / "exemplars"
EXEMPLAR_JSON = EXEMPLAR_DIR / "exemplars.json"
PROFILES_DIR = EXEMPLAR_DIR / "profiles"


def reset_caches() -> dict:
    """유니버스/오늘 가격/오늘 스코어 캐시를 삭제하고 삭제된 파일 수 반환."""
    counts = {"universe": 0, "prices": 0, "scores": 0}
    universe_file = UNIVERSE_DIR / "tickers.parquet"
    if universe_file.exists():
        universe_file.unlink()
        counts["universe"] = 1
    today = date.today().isoformat()
    for p in CACHE_DIR.glob(f"*_{today}.parquet"):
        p.unlink()
        counts["prices"] += 1
    for p in SCORES_DIR.glob(f"{today}_scores.parquet"):
        p.unlink()
        counts["scores"] += 1
    return counts


def run_backtest_subprocess(top_mcap: int, top_n: int, rebalance: str):
    """recommendation.backtest 모듈을 subprocess로 실행하고 stdout을 한 줄씩 yield."""
    cmd = [
        _sys.executable, "-m", "recommendation.backtest",
        "--top-mcap", str(top_mcap),
        "--top-n", str(top_n),
        "--rebalance", rebalance,
        "--use-cache",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT), text=True, encoding="utf-8", errors="replace",
        bufsize=1,
    )
    try:
        for line in proc.stdout:
            yield line.rstrip()
    finally:
        proc.wait()
        if proc.returncode != 0:
            yield f"[ERROR] backtest exited with code {proc.returncode}"


def load_latest_backtest() -> tuple[dict | None, pd.DataFrame | None]:
    """가장 최근 백테스트 결과(summary.json + equity.parquet) 로드."""
    if not BACKTEST_DIR.exists():
        return None, None
    summaries = sorted(BACKTEST_DIR.glob("*_summary.json"), reverse=True)
    if not summaries:
        return None, None
    summary_path = summaries[0]
    equity_path = BACKTEST_DIR / summary_path.name.replace("_summary.json", "_equity.parquet")
    import json as _json
    with open(summary_path, encoding="utf-8") as f:
        summary = _json.load(f)
    equity_df = pd.read_parquet(equity_path) if equity_path.exists() else None
    return summary, equity_df


def run_refresh_pipeline(top_mcap: int, exemplar_weight: float = 0.0):
    """파이프라인을 subprocess로 실행하고 stdout을 한 줄씩 yield한다."""
    cmd = [
        _sys.executable, "-m", "recommendation.pipeline",
        "--top-mcap", str(top_mcap), "--top", "30",
    ]
    if exemplar_weight > 0:
        cmd.extend(["--exemplar-weight", str(exemplar_weight)])
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT), text=True, encoding="utf-8", errors="replace",
        bufsize=1,
    )
    try:
        for line in proc.stdout:
            yield line.rstrip()
    finally:
        proc.wait()
        if proc.returncode != 0:
            yield f"[ERROR] pipeline exited with code {proc.returncode}"

st.set_page_config(
    page_title="미국주식 종목 추천",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(ttl=3600)
def load_scores(scores_date: str | None = None) -> pd.DataFrame:
    """스코어 파일 로드"""
    if scores_date:
        path = SCORES_DIR / f"{scores_date}_scores.parquet"
    else:
        files = sorted(SCORES_DIR.glob("*_scores.parquet"), reverse=True)
        if not files:
            return pd.DataFrame()
        path = files[0]

    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


def render_rating_badge(rating: str) -> str:
    """등급별 색상 뱃지"""
    colors = {
        "Strong Buy": "🟢",
        "Buy": "🔵",
        "Hold": "🟡",
        "Avoid": "🔴",
    }
    return f"{colors.get(rating, '⚪')} {rating}"


def trigger_analysis_request(ticker: str, score_row: pd.Series, key_prefix: str = "") -> None:
    """분석 요청 버튼 + 결과 토스트. score_row는 Top 카드/상세 어디서든 호출 가능."""
    pending = list_pending(ticker)
    if pending:
        st.caption(f"⏳ 대기 중인 요청 {len(pending)}개 — Claude 세션에서 '{ticker} 분석해줘'")
    if st.button(f"🤖 {ticker} 분석 요청", key=f"{key_prefix}analyze_{ticker}"):
        with st.spinner(f"{ticker} 뉴스 수집 중..."):
            news = fetch_news_yfinance(ticker, max_items=5)
            path = create_request(ticker, score_row, news_items=news)
        st.success(
            f"요청 생성됨: `{path.name}` (뉴스 {len(news)}개 포함)\n\n"
            f"Claude Code 세션에서 **'{ticker} 분석해줘'** 라고 말하세요."
        )


def render_radar_chart(row: pd.Series) -> go.Figure:
    """6팩터 레이더 차트"""
    categories = ["Momentum", "Trend", "Breakout", "Valuation", "Growth", "Risk"]
    values = [
        row.get("momentum", 0),
        row.get("trend", 0),
        row.get("breakout", 0),
        row.get("valuation", 0),
        row.get("growth", 0),
        row.get("risk", 0),
    ]

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        line_color="rgb(67, 147, 195)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=20, b=20),
    )
    return fig


@st.cache_resource
def get_library() -> ExemplarLibrary:
    return ExemplarLibrary(EXEMPLAR_JSON)


def render_exemplar_form() -> None:
    """모범 추가 폼 + 저장 로직."""
    st.subheader("모범 추가")
    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        ticker = st.text_input("티커", value="").strip().upper()
    with col2:
        name = st.text_input("이름 (선택)", value="")
    with col3:
        c1, c2 = st.columns(2)
        with c1:
            start = st.date_input("시작일", value=date(2023, 1, 1))
        with c2:
            end = st.date_input("종료일", value=date(2023, 7, 1))

    preview_clicked = st.button("미리보기")
    save_clicked = st.button("저장", type="primary")

    if not (preview_clicked or save_clicked):
        return
    if not ticker:
        st.error("티커를 입력하세요.")
        return
    if end <= start:
        st.error("종료일은 시작일보다 뒤여야 합니다.")
        return

    try:
        profile = build_profile(ticker, start, end)
    except Exception as e:
        st.error(f"프로파일 빌드 실패: {e}")
        return

    st.markdown("**프로파일 미리보기**")
    preview_df = pd.DataFrame([
        {"피처": k, "평균": v["mean"], "표준편차": v["std"]}
        for k, v in profile.items()
    ])
    st.dataframe(preview_df, use_container_width=True, hide_index=True)

    if save_clicked:
        exemplar_id = make_exemplar_id(ticker, start, name or None)
        lib = get_library()
        if lib.get(exemplar_id) is not None:
            st.error(f"이미 존재하는 ID: {exemplar_id}")
            return
        profile_path = PROFILES_DIR / f"{exemplar_id}.parquet"
        save_profile(profile_path, profile)
        ex = Exemplar(
            id=exemplar_id,
            ticker=ticker,
            name=name or ticker,
            start_date=start,
            end_date=end,
            active=True,
            created_at=datetime.now().isoformat(timespec="seconds"),
            profile_path=str(profile_path.relative_to(PROJECT_ROOT)),
        )
        lib.add(ex)
        st.success(f"저장 완료: {exemplar_id}")
        st.cache_resource.clear()
        st.rerun()


def render_exemplar_list() -> None:
    """등록된 모범 목록 + 활성 토글 + 삭제."""
    st.subheader("등록된 모범")
    lib = get_library()
    items = lib.list_all()
    if not items:
        st.info("등록된 모범이 없습니다.")
        return

    for ex in items:
        cols = st.columns([2, 3, 2, 1, 1])
        with cols[0]:
            st.code(ex.id, language=None)
        with cols[1]:
            st.write(f"**{ex.name}** ({ex.ticker})")
            st.caption(f"{ex.start_date} ~ {ex.end_date}")
        with cols[2]:
            new_active = st.checkbox("활성", value=ex.active, key=f"active_{ex.id}")
            if new_active != ex.active:
                lib.toggle_active(ex.id, new_active)
                st.rerun()
        with cols[3]:
            if st.button("상세", key=f"detail_{ex.id}"):
                st.session_state[f"show_detail_{ex.id}"] = not st.session_state.get(f"show_detail_{ex.id}", False)
        with cols[4]:
            if st.button("🗑", key=f"del_{ex.id}"):
                profile_path = PROJECT_ROOT / ex.profile_path
                if profile_path.exists():
                    profile_path.unlink()
                lib.delete(ex.id)
                st.cache_resource.clear()
                st.rerun()

        if st.session_state.get(f"show_detail_{ex.id}", False):
            profile_path = PROJECT_ROOT / ex.profile_path
            if profile_path.exists():
                profile = load_profile(profile_path)
                detail_df = pd.DataFrame([
                    {"피처": k, "평균": v["mean"], "표준편차": v["std"]}
                    for k, v in profile.items()
                ])
                st.dataframe(detail_df, use_container_width=True, hide_index=True)
            else:
                st.warning(f"프로파일 파일 없음: {ex.profile_path}")


def render_price_chart(ticker: str) -> go.Figure | None:
    """종목 가격 차트"""
    today = date.today().isoformat()
    cache_file = CACHE_DIR / f"{ticker}_{today}.parquet"

    if not cache_file.exists():
        # 오늘 캐시 없으면 최근 캐시 검색
        files = sorted(CACHE_DIR.glob(f"{ticker}_*.parquet"), reverse=True)
        if not files:
            return None
        cache_file = files[0]

    df = pd.read_parquet(cache_file)
    if df.empty:
        return None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.05)

    # 캔들스틱
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
    ), row=1, col=1)

    # SMA
    if len(df) >= 20:
        sma20 = df["Close"].rolling(20).mean()
        fig.add_trace(go.Scatter(x=df.index, y=sma20, name="SMA20",
                                 line=dict(width=1, color="orange")), row=1, col=1)
    if len(df) >= 50:
        sma50 = df["Close"].rolling(50).mean()
        fig.add_trace(go.Scatter(x=df.index, y=sma50, name="SMA50",
                                 line=dict(width=1, color="blue")), row=1, col=1)

    # 거래량
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                         marker_color="rgba(100,100,200,0.3)"), row=2, col=1)

    fig.update_layout(height=400, showlegend=True, xaxis_rangeslider_visible=False,
                      margin=dict(l=40, r=40, t=20, b=20))
    return fig


# ─── 메인 대시보드 ───

st.title("📈 미국주식 종목 추천")

# 스코어 로드
scores_df = load_scores()
analyses, analysis_date = get_latest_analysis()

if scores_df.empty:
    st.warning("아직 스코어링 결과가 없습니다. `python -m recommendation.pipeline`을 먼저 실행하세요.")
    st.code("python -m recommendation.pipeline --subset AAPL,MSFT,GOOGL,AMZN,NVDA --top 10", language="bash")
    st.stop()

# 사이드바
st.sidebar.header("설정")
top_n = st.sidebar.slider("Top N 표시", 5, 50, 15)
score_date = st.sidebar.text_input("스코어 날짜 (YYYY-MM-DD)", value="")

if score_date:
    scores_df = load_scores(score_date)
    if scores_df.empty:
        st.sidebar.error(f"{score_date} 데이터 없음")

st.sidebar.divider()
st.sidebar.subheader("모범 유사도 팩터")
exemplar_weight = st.sidebar.slider("가중치", 0.0, 0.40, 0.20, 0.05,
                                     help="0이면 7번째 팩터 비활성. 슬라이더 변경은 재실행 후 반영됩니다.")
lib_preview = get_library()
n_active = sum(1 for e in lib_preview.list_all() if e.active)
n_total = len(lib_preview.list_all())
st.sidebar.caption(f"활성 모범: {n_active} / {n_total}")
if n_active == 0:
    st.sidebar.warning("활성 모범이 없습니다 — 7번째 팩터가 비활성화됩니다.")

st.sidebar.divider()
st.sidebar.subheader("데이터 새로고침")
refresh_mcap = st.sidebar.number_input(
    "시총 상위 N개", min_value=50, max_value=8000, value=5000, step=500,
    help="유니버스 + 가격 캐시 전체 재수집 후 재스코어링",
)
refresh_clicked = st.sidebar.button("🔄 전체 새로고침", type="primary",
                                     help="유니버스(NASDAQ) + 오늘 가격 + 스코어 캐시 삭제 후 재실행")

if refresh_clicked:
    with st.sidebar.status("데이터 새로고침 중...", expanded=True) as status:
        counts = reset_caches()
        status.write(f"캐시 삭제: 유니버스 {counts['universe']}, 가격 {counts['prices']}, 스코어 {counts['scores']}")
        log_box = st.empty()
        log_lines: list[str] = []
        for line in run_refresh_pipeline(int(refresh_mcap), exemplar_weight):
            log_lines.append(line)
            log_box.code("\n".join(log_lines[-15:]), language=None)
        status.update(label="✅ 새로고침 완료", state="complete", expanded=False)
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("백테스트")
bt_mcap = st.sidebar.number_input(
    "유니버스 시총 상위 N개", min_value=50, max_value=8000, value=5000, step=500,
    key="bt_mcap",
)
bt_topn = st.sidebar.number_input("Top N 픽", min_value=3, max_value=50, value=10, step=1)
bt_rebalance = st.sidebar.selectbox(
    "리밸런싱 빈도",
    options=["W-MON", "W-FRI", "2W-MON", "M"],
    index=0,
    help="W-MON=매주 월요일, M=매월 말일",
)
bt_clicked = st.sidebar.button("🧪 백테스트 실행", type="secondary",
                                help="현재 캐시된 가격 데이터로 워크포워드 백테스트. ~10-15분 소요")

if bt_clicked:
    with st.sidebar.status("백테스트 실행 중...", expanded=True) as status:
        log_box = st.empty()
        log_lines: list[str] = []
        for line in run_backtest_subprocess(int(bt_mcap), int(bt_topn), bt_rebalance):
            log_lines.append(line)
            log_box.code("\n".join(log_lines[-15:]), language=None)
        status.update(label="✅ 백테스트 완료", state="complete", expanded=False)
    st.cache_data.clear()
    st.rerun()

# 탭 구성
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Top 추천", "📊 전체 랭킹", "🔍 종목 상세", "📚 모범 라이브러리", "🧪 백테스트"])

# ─── Tab 1: Top 추천 ───
with tab1:
    st.subheader(f"Top {top_n} 추천 종목")

    if "exemplar_similarity" not in scores_df.columns:
        st.caption("💡 모범 유사도 팩터를 활성화하려면 사이드바에서 가중치를 조정 후 `python -m recommendation.pipeline --exemplar-weight 0.20`을 실행하세요.")

    # 분석 결과를 딕셔너리로 변환
    analysis_dict = {a["ticker"]: a for a in analyses} if analyses else {}

    top_df = scores_df.head(top_n)

    # 카드형 표시 (3열)
    cols_per_row = 3
    for i in range(0, len(top_df), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(top_df):
                break

            row = top_df.iloc[idx]
            ticker = row["ticker"]
            analysis = analysis_dict.get(ticker, {})

            with col:
                st.markdown(f"### #{row['rank']} {ticker}")
                st.metric("Total Score", f"{row['total_score']:.1f}")

                # 모범 유사도 표시 (있을 때만)
                if "exemplar_similarity" in row.index and pd.notna(row.get("exemplar_similarity")):
                    best_id = row.get("best_match_id", "")
                    best_name = ""
                    if best_id:
                        match_ex = lib_preview.get(best_id)
                        best_name = match_ex.name if match_ex else best_id
                    st.caption(f"모범 유사도: {row['exemplar_similarity']:.1f} ({best_name} 닮음)")

                if analysis:
                    st.markdown(render_rating_badge(analysis.get("rating", "N/A")))
                    st.caption(analysis.get("summary", ""))
                else:
                    st.caption("AI 분석 없음")
                    trigger_analysis_request(ticker, row, key_prefix="card_")

                # 미니 팩터 바
                factors = {
                    "Mom": row["momentum"],
                    "Trend": row["trend"],
                    "Break": row["breakout"],
                    "Val": row["valuation"],
                    "Growth": row["growth"],
                    "Risk": row["risk"],
                }
                factor_df = pd.DataFrame([factors])
                st.bar_chart(factor_df.T, height=100)

    # Claude 분석 요약
    if analyses:
        st.divider()
        st.subheader("Claude AI 분석 요약")
        st.caption(f"분석 날짜: {analysis_date}")

        for a in analyses[:top_n]:
            with st.expander(f"{render_rating_badge(a['rating'])} **{a['ticker']}** — {a.get('summary', '')}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**카탈리스트:** {a.get('catalyst', 'N/A')}")
                with c2:
                    st.markdown(f"**리스크:** {a.get('risk', 'N/A')}")
                st.markdown(f"**홀딩 기간:** {a.get('holding_period', 'N/A')}")

# ─── Tab 2: 전체 랭킹 ───
with tab2:
    st.subheader(f"전체 랭킹 ({len(scores_df)}개 종목)")

    # 검색/필터
    search = st.text_input("종목 검색 (티커)", "")
    if search:
        filtered = scores_df[scores_df["ticker"].str.contains(search.upper())]
    else:
        filtered = scores_df

    # 정렬 옵션
    sort_col = st.selectbox("정렬 기준", ["total_score", "momentum", "trend", "breakout", "valuation", "growth", "risk"])
    filtered = filtered.sort_values(sort_col, ascending=False)

    # 테이블 표시
    st.dataframe(
        filtered[["rank", "ticker", "total_score", "momentum", "trend", "breakout", "valuation", "growth", "risk"]],
        use_container_width=True,
        height=600,
    )

    # 분포 차트
    st.subheader("점수 분포")
    fig = px.histogram(scores_df, x="total_score", nbins=50, title="Total Score Distribution")
    st.plotly_chart(fig, use_container_width=True)

# ─── Tab 3: 종목 상세 ───
with tab3:
    st.subheader("종목 상세 분석")

    # 종목 선택
    selected_ticker = st.selectbox(
        "종목 선택",
        options=scores_df["ticker"].tolist(),
        index=0,
    )

    if selected_ticker:
        row = scores_df[scores_df["ticker"] == selected_ticker].iloc[0]

        # 헤더
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"## {selected_ticker}")
            st.markdown(f"**Rank:** #{int(row['rank'])} / {len(scores_df)}")
        with col2:
            st.metric("Total Score", f"{row['total_score']:.1f}")
        with col3:
            analysis = analysis_dict.get(selected_ticker, {})
            if analysis:
                st.markdown(render_rating_badge(analysis.get("rating", "N/A")))

        # 2열 레이아웃
        left, right = st.columns([2, 1])

        with left:
            # 가격 차트
            price_fig = render_price_chart(selected_ticker)
            if price_fig:
                st.plotly_chart(price_fig, use_container_width=True)
            else:
                st.info("가격 데이터 캐시 없음")

        with right:
            # 레이더 차트
            radar_fig = render_radar_chart(row)
            st.plotly_chart(radar_fig, use_container_width=True)

        # 모범 유사도 분해 (있을 때만)
        if "exemplar_similarity" in row.index and pd.notna(row.get("exemplar_similarity")):
            st.divider()
            st.subheader("모범 유사도 분해")
            best_id = row.get("best_match_id", "")
            match_ex = lib_preview.get(best_id) if best_id else None
            if match_ex:
                profile_path = PROJECT_ROOT / match_ex.profile_path
                if profile_path.exists():
                    profile = load_profile(profile_path)
                    cache_file = CACHE_DIR / f"{selected_ticker}_{date.today().isoformat()}.parquet"
                    if not cache_file.exists():
                        cache_files = sorted(CACHE_DIR.glob(f"{selected_ticker}_*.parquet"), reverse=True)
                        cache_file = cache_files[0] if cache_files else None
                    if cache_file and cache_file.exists():
                        candidate_df = pd.read_parquet(cache_file)
                        try:
                            from recommendation.exemplar.features import compute_features_snapshot
                            snap = compute_features_snapshot(candidate_df)
                            rows = []
                            for k in FEATURE_KEYS:
                                mu = profile[k]["mean"]
                                sigma = profile[k]["std"]
                                cv = snap[k]
                                z = (cv - mu) / sigma if sigma > 1e-6 else 0.0
                                rows.append({
                                    "피처": k,
                                    f"모범 (μ±σ)": f"{mu:.4f} ± {sigma:.4f}",
                                    "후보 오늘": f"{cv:.4f}",
                                    "z-거리": f"{z:+.2f}",
                                })
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                        except ValueError as e:
                            st.warning(f"스냅샷 계산 실패: {e}")

        # Claude 분석
        st.divider()
        if analysis:
            st.subheader("Claude AI 의견")
            st.markdown(f"**등급:** {render_rating_badge(analysis.get('rating', 'N/A'))}")
            st.markdown(f"**요약:** {analysis.get('summary', 'N/A')}")
            st.markdown(f"**카탈리스트:** {analysis.get('catalyst', 'N/A')}")
            st.markdown(f"**리스크:** {analysis.get('risk', 'N/A')}")
            st.markdown(f"**홀딩 기간:** {analysis.get('holding_period', 'N/A')}")
            # 상세 리포트 파일 표시
            from recommendation.analysis_request import REPORT_DIR
            from datetime import date as _date
            report_files = sorted(REPORT_DIR.glob(f"{selected_ticker}_*.md"), reverse=True)
            if report_files:
                with st.expander(f"📄 상세 리포트 ({report_files[0].stem})"):
                    st.markdown(report_files[0].read_text(encoding="utf-8"))
        else:
            st.subheader("AI 분석 없음")
            trigger_analysis_request(selected_ticker, row, key_prefix="detail_")

# ─── Tab 4: 모범 라이브러리 ───
with tab4:
    render_exemplar_form()
    st.divider()
    render_exemplar_list()

# ─── Tab 5: 백테스트 ───
with tab5:
    st.subheader("백테스트 결과")
    bt_summary, bt_equity = load_latest_backtest()

    if bt_summary is None:
        st.info("아직 백테스트 결과가 없습니다. 사이드바에서 **🧪 백테스트 실행** 버튼을 누르세요.")
    else:
        # KPI 카드
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 수익률",
                      f"{bt_summary['total_return']*100:+.2f}%",
                      delta=f"SPY {bt_summary['spy_total_return']*100:+.2f}%")
        with col2:
            st.metric("연환산 수익률", f"{bt_summary['annualized_return']*100:+.2f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{bt_summary['sharpe_ratio']:.2f}")
        with col4:
            st.metric("Max Drawdown", f"{bt_summary['max_drawdown']*100:.2f}%")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            alpha = bt_summary['alpha_vs_spy']
            st.metric("Alpha vs SPY (연환산)", f"{alpha*100:+.2f}%",
                      delta="목표 > 0", delta_color="off")
        with col6:
            st.metric("연환산 변동성", f"{bt_summary['annualized_volatility']*100:.2f}%")
        with col7:
            st.metric("리밸런싱", f"{bt_summary['n_rebalances']}회")
        with col8:
            st.metric("최종 자본",
                      f"${bt_summary['final_capital']:,.0f}",
                      delta=f"${bt_summary['final_capital'] - bt_summary['initial_capital']:+,.0f}")

        st.caption(f"기간: {bt_summary['start_date']} ~ {bt_summary['end_date']}")

        # KPI 통과 여부
        kpi_pass = {
            "Alpha > 0": alpha > 0,
            "Sharpe > 1.0": bt_summary['sharpe_ratio'] > 1.0,
            "Max DD > -20%": bt_summary['max_drawdown'] > -0.20,
            "Strategy > SPY": bt_summary['total_return'] > bt_summary['spy_total_return'],
        }
        passed = sum(kpi_pass.values())
        st.write(f"### KPI 통과: {passed}/{len(kpi_pass)}")
        kpi_cols = st.columns(len(kpi_pass))
        for i, (name, ok) in enumerate(kpi_pass.items()):
            with kpi_cols[i]:
                st.write(f"{'✅' if ok else '❌'} {name}")

        # 자본 곡선 차트
        if bt_equity is not None and not bt_equity.empty:
            st.divider()
            st.subheader("자본 곡선 (전략 vs SPY)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=bt_equity.index, y=bt_equity["strategy"],
                name="Strategy", line=dict(color="rgb(67, 147, 195)", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=bt_equity.index, y=bt_equity["spy"],
                name="SPY", line=dict(color="rgb(200, 100, 100)", width=2, dash="dash"),
            ))
            fig.update_layout(
                height=400, hovermode="x unified",
                xaxis_title="Date", yaxis_title="Portfolio Value ($)",
                margin=dict(l=40, r=40, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            # 일별 수익률 차트
            st.subheader("일별 수익률 (전략)")
            daily_returns = bt_equity["strategy"].pct_change().dropna() * 100
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=daily_returns.index, y=daily_returns,
                marker_color=["green" if r > 0 else "red" for r in daily_returns],
            ))
            fig2.update_layout(
                height=200, yaxis_title="Daily Return (%)",
                margin=dict(l=40, r=40, t=20, b=40),
            )
            st.plotly_chart(fig2, use_container_width=True)
