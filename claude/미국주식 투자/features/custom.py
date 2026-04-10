"""
커스텀 피처 모듈.

모든 피처는 과거 데이터만 사용해서 백테스팅 가능하게 설계됨.
실시간 전용 데이터(Fear/Greed, 내부자 거래 등) 사용 금지.

주의: 대부분의 함수는 features.technical.add_all_indicators()가
먼저 호출되어 SMA, RSI, ATR, BB, Volume_SMA_20 컬럼이 있어야 함.
"""
import pandas as pd
import numpy as np

EPS = 1e-9


def add_relative_strength(df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """
    종목 vs 벤치마크(SPY) 상대강도.
    - RS_vs_Benchmark: (종목 20일 변화율) / (벤치마크 20일 변화율)
    - RS_Momentum_60d: 60일간 상대강도 변화

    왜: SPY 대비 강한 종목을 선별하기 위함. 모멘텀 전략의 핵심 팩터.
    """
    df = df.copy()
    bench_close = benchmark_df['Close'].reindex(df.index, method='ffill')

    stock_ratio = df['Close'] / df['Close'].shift(20)
    bench_ratio = bench_close / bench_close.shift(20)

    df['RS_vs_Benchmark'] = stock_ratio / (bench_ratio + EPS)
    df['RS_Momentum_60d'] = df['RS_vs_Benchmark'] / df['RS_vs_Benchmark'].shift(60) - 1
    return df


def add_52week_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    52주 고점/저점 대비 현재 위치.

    왜: 52주 고점 근처 = 강세/과열, 저점 근처 = 약세/바닥. 추세 전략에 사용.
    """
    df = df.copy()
    df['High_52w'] = df['High'].rolling(window=252).max()
    df['Low_52w'] = df['Low'].rolling(window=252).min()

    df['Pct_From_52w_High'] = (df['Close'] - df['High_52w']) / (df['High_52w'] + EPS)
    df['Pct_From_52w_Low'] = (df['Close'] - df['Low_52w']) / (df['Low_52w'] + EPS)

    range_52w = df['High_52w'] - df['Low_52w']
    df['Position_in_52w_Range'] = (df['Close'] - df['Low_52w']) / (range_52w + EPS)
    df['Position_in_52w_Range'] = df['Position_in_52w_Range'].clip(0, 1)
    return df


def add_distance_from_ma(df: pd.DataFrame) -> pd.DataFrame:
    """
    주요 이동평균(SMA 20/50/200)과의 거리 (%).

    왜: 이동평균에서 너무 멀어진 종목은 평균회귀 대상. 추세 강도도 판단.
    """
    df = df.copy()
    for window in [20, 50, 200]:
        col = f'SMA_{window}'
        if col not in df.columns:
            print(f"Warning: {col} 없음 — technical.add_all_indicators 먼저 호출 필요")
            continue
        df[f'Dist_From_SMA{window}'] = (df['Close'] - df[col]) / (df[col] + EPS)
    return df


def add_volatility_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    ATR 기반 정규화 변동성 (ATR / Close).

    왜: 종목간 비교 가능한 변동성 지표. 포지션 사이징에 사용.
    """
    df = df.copy()
    if 'ATR_14' not in df.columns:
        print("Warning: ATR_14 없음 — technical.add_all_indicators 먼저 호출 필요")
        return df
    df['Volatility_Ratio'] = df['ATR_14'] / (df['Close'] + EPS)
    return df


def add_volume_spike(df: pd.DataFrame) -> pd.DataFrame:
    """
    거래량 스파이크 감지 (현재 거래량 / 20일 평균 거래량).

    왜: 거래량 급증은 기관/큰손의 관심 신호. 돌파/반전의 유효성 판단.
    """
    df = df.copy()
    if 'Volume_SMA_20' not in df.columns:
        print("Warning: Volume_SMA_20 없음 — technical.add_all_indicators 먼저 호출 필요")
        return df
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA_20'] + EPS)
    df['Volume_Spike'] = df['Volume_Ratio'] > 2.0
    return df


def add_overbought_oversold(df: pd.DataFrame) -> pd.DataFrame:
    """
    RSI + 볼린저밴드 결합 과매수/과매도 신호.

    왜: RSI 단독보다 BB 결합이 false signal 줄임. stock-analysis skill의 Overbought 로직 참고.
    """
    df = df.copy()
    required = ['RSI_14', 'BB_Upper', 'BB_Lower']
    for col in required:
        if col not in df.columns:
            print(f"Warning: {col} 없음 — technical.add_all_indicators 먼저 호출 필요")
            return df

    df['Overbought'] = (df['RSI_14'] > 70) & (df['Close'] > df['BB_Upper'])
    df['Oversold'] = (df['RSI_14'] < 30) & (df['Close'] < df['BB_Lower'])
    return df


def add_golden_death_cross(df: pd.DataFrame) -> pd.DataFrame:
    """
    골든크로스(50일선이 200일선 상향돌파) / 데드크로스(하향돌파).

    왜: 장기 추세 전환 신호. 추세 추종 전략의 주요 진입/청산 시그널.
    """
    df = df.copy()
    if 'SMA_50' not in df.columns or 'SMA_200' not in df.columns:
        print("Warning: SMA_50/SMA_200 없음 — technical.add_all_indicators 먼저 호출 필요")
        return df

    prev_50 = df['SMA_50'].shift(1)
    prev_200 = df['SMA_200'].shift(1)
    df['Golden_Cross'] = (prev_50 <= prev_200) & (df['SMA_50'] > df['SMA_200'])
    df['Death_Cross'] = (prev_50 >= prev_200) & (df['SMA_50'] < df['SMA_200'])
    return df


def add_vix_regime(df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """
    VIX 레벨에 따른 시장 변동성 레짐 분류.
    - low (<15): 안정
    - normal (15~25): 평상
    - high (25~35): 경계
    - extreme (>35): 공포

    왜: VIX가 높을 때는 기술적 돌파 신호의 신뢰도가 낮음. 시장 체제 필터로 사용.
    """
    df = df.copy()
    if 'Close' not in vix_df.columns:
        print("Warning: vix_df에 'Close' 컬럼 없음")
        return df

    df['VIX_Level'] = vix_df['Close'].reindex(df.index, method='ffill')

    # right=True (기본): [구간 시작, 구간 끝] → 15는 normal, 25는 high
    df['VIX_Regime'] = pd.cut(
        df['VIX_Level'],
        bins=[-np.inf, 15, 25, 35, np.inf],
        labels=['low', 'normal', 'high', 'extreme']
    )
    df['VIX_High_Flag'] = df['VIX_Level'] > 30
    return df


def add_safe_haven_signal(df: pd.DataFrame,
                           gld_df: pd.DataFrame,
                           tlt_df: pd.DataFrame,
                           uup_df: pd.DataFrame) -> pd.DataFrame:
    """
    안전자산 동반 상승 감지 (Risk-Off 신호).
    5일 수익률: GLD ≥ 2%, TLT ≥ 1%, UUP ≥ 1% 모두 만족 시 True.

    왜: 금/국채/달러가 동시에 오르면 투자자가 위험자산에서 안전자산으로 이동 중.
    백테스트 시 Risk_Off_Signal=True인 날은 신규 진입 자제 권장.
    (stock-analysis skill의 Risk-Off 로직 참고)
    """
    df = df.copy()
    for name, sdf in [('GLD', gld_df), ('TLT', tlt_df), ('UUP', uup_df)]:
        if 'Close' not in sdf.columns:
            print(f"Warning: {name} df에 'Close' 컬럼 없음")
            return df

    gld_5d = gld_df['Close'].reindex(df.index, method='ffill').pct_change(5)
    tlt_5d = tlt_df['Close'].reindex(df.index, method='ffill').pct_change(5)
    uup_5d = uup_df['Close'].reindex(df.index, method='ffill').pct_change(5)

    df['Risk_Off_Signal'] = (gld_5d >= 0.02) & (tlt_5d >= 0.01) & (uup_5d >= 0.01)
    return df


def add_all_custom_features(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame = None,
    vix_df: pd.DataFrame = None,
    gld_df: pd.DataFrame = None,
    tlt_df: pd.DataFrame = None,
    uup_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    모든 커스텀 피처를 한번에 추가.
    외부 df(benchmark, vix 등)가 None이면 해당 피처는 스킵.

    선행 조건: features.technical.add_all_indicators()가 먼저 호출되어야 함.
    """
    df = df.copy()

    if benchmark_df is not None and not benchmark_df.empty:
        df = add_relative_strength(df, benchmark_df)

    df = add_52week_position(df)
    df = add_distance_from_ma(df)
    df = add_volatility_ratio(df)
    df = add_volume_spike(df)
    df = add_overbought_oversold(df)
    df = add_golden_death_cross(df)

    if vix_df is not None and not vix_df.empty:
        df = add_vix_regime(df, vix_df)

    if (gld_df is not None and not gld_df.empty and
            tlt_df is not None and not tlt_df.empty and
            uup_df is not None and not uup_df.empty):
        df = add_safe_haven_signal(df, gld_df, tlt_df, uup_df)

    return df
