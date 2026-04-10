"""
전략 베이스 클래스 + MultiFactor 스코어링 시스템.

설계 원칙:
- 모든 스코어는 -1 ~ +1 범위로 정규화
- **Look-ahead bias 방지**: rolling 윈도우 또는 clip/tanh 기반 정규화만 사용.
  전체 기간의 min/max 정규화는 금지 (미래 데이터 누수).
- 필요한 컬럼이 없으면 해당 팩터는 0(중립) 반환

팩터 구성 (stock-analysis skill의 8차원 참고):
1. momentum (15%): RSI, 52주 위치, MACD
2. trend (20%): SMA 정렬, SMA200 거리
3. volatility (10%): 정규화 변동성 (역방향 — 낮을수록 가점)
4. volume (10%): 거래량 비율, 스파이크
5. relative_strength (15%): SPY 대비 RS
6. mean_reversion (10%): RSI/BB 결합
7. market_regime (10%): VIX 레짐, Risk-Off
8. breakout (10%): 골든크로스, 52주 고점 근접
"""
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


def _safe_tanh(series: pd.Series, scale: float = 1.0) -> pd.Series:
    """tanh로 -1~+1 스무스하게 매핑 (look-ahead bias 없음)."""
    return np.tanh(series * scale).fillna(0)


def _rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score 정규화 (look-ahead bias 없음)."""
    mean = series.rolling(window=window, min_periods=20).mean()
    std = series.rolling(window=window, min_periods=20).std()
    z = (series - mean) / (std + 1e-9)
    return z.fillna(0)


class BaseStrategy(ABC):
    """
    모든 전략의 공통 인터페이스.

    하위 클래스는 generate_signals()를 반드시 구현해야 함.
    """
    name: str = "BaseStrategy"
    description: str = "전략 베이스 클래스"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params: Dict[str, Any] = dict(self.default_params())
        if params:
            self.set_params(**params)

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        """기본 파라미터. 하위 클래스에서 override."""
        return {}

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        매매 시그널 생성.

        Args:
            df: 가격 + 지표 + 커스텀 피처 포함 DataFrame

        Returns:
            'signal' 컬럼 추가된 DataFrame (1=buy, 0=hold, -1=sell)
        """
        ...

    def get_params(self) -> Dict[str, Any]:
        return self.params

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict) and isinstance(self.params.get(k), dict):
                self.params[k].update(v)
            else:
                self.params[k] = v


class MultiFactorStrategy(BaseStrategy):
    """
    8개 팩터의 가중합 스코어링 전략.

    total_score > buy_threshold → buy (1)
    total_score < sell_threshold → sell (-1)
    그 외 → hold (0)

    Risk_Off_Signal이 True인 날은 buy 시그널 강도를 감쇠.
    """
    name = "MultiFactorStrategy"
    description = "8개 팩터 가중합 기반 다중 팩터 전략"

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {
            'weights': {
                'momentum': 0.15,
                'trend': 0.20,
                'volatility': 0.10,
                'volume': 0.10,
                'relative_strength': 0.15,
                'mean_reversion': 0.10,
                'market_regime': 0.10,
                'breakout': 0.10,
            },
            'buy_threshold': 0.30,
            'sell_threshold': -0.30,
            'risk_off_buy_multiplier': 0.5,  # Risk-Off시 buy 강도 절반
        }

    # ------------------------------------------------------------------
    # 팩터별 스코어링 함수 (-1 ~ +1)
    # ------------------------------------------------------------------

    def score_momentum(self, df: pd.DataFrame) -> pd.Series:
        """RSI + 52주 위치 + MACD 히스토그램 결합."""
        required = ['RSI_14', 'Position_in_52w_Range', 'MACD_Hist']
        if not all(c in df.columns for c in required):
            return pd.Series(0.0, index=df.index)

        # RSI: 50 기준 정규화, 70+이면 +1 근접
        rsi_s = _safe_tanh((df['RSI_14'] - 50) / 20)

        # 52주 위치: 0~1을 -1~+1로 매핑
        pos_s = (df['Position_in_52w_Range'] * 2 - 1).clip(-1, 1).fillna(0)

        # MACD 히스토그램: rolling zscore → tanh
        macd_s = _safe_tanh(_rolling_zscore(df['MACD_Hist'], 60) / 2)

        combined = 0.4 * rsi_s + 0.3 * pos_s + 0.3 * macd_s
        return combined.clip(-1, 1)

    def score_trend(self, df: pd.DataFrame) -> pd.Series:
        """SMA 정렬 + SMA200 거리."""
        required = ['SMA_20', 'SMA_50', 'SMA_200', 'Dist_From_SMA200']
        if not all(c in df.columns for c in required):
            return pd.Series(0.0, index=df.index)

        # SMA 정렬: 20>50>200 → +1, 20<50<200 → -1, 그 외 → 0
        bullish = (df['SMA_20'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_200'])
        bearish = (df['SMA_20'] < df['SMA_50']) & (df['SMA_50'] < df['SMA_200'])
        alignment = bullish.astype(float) - bearish.astype(float)

        # SMA200 거리: tanh로 정규화 (0.2 = 20% 괴리 시 ~1)
        dist_s = _safe_tanh(df['Dist_From_SMA200'] * 5)

        combined = 0.6 * alignment + 0.4 * dist_s
        return combined.clip(-1, 1)

    def score_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Volatility_Ratio: 낮을수록 안정 → 양수. rolling zscore 역방향."""
        if 'Volatility_Ratio' not in df.columns:
            return pd.Series(0.0, index=df.index)

        # 낮은 변동성이 +1 방향, 높은 변동성이 -1 방향
        return -_safe_tanh(_rolling_zscore(df['Volatility_Ratio'], 60) / 2)

    def score_volume(self, df: pd.DataFrame) -> pd.Series:
        """거래량 비율 + 스파이크."""
        if 'Volume_Ratio' not in df.columns:
            return pd.Series(0.0, index=df.index)

        # Volume_Ratio: 1 기준으로 (ratio - 1)을 tanh로
        vr_s = _safe_tanh((df['Volume_Ratio'] - 1) / 2)

        if 'Volume_Spike' in df.columns:
            spike_s = df['Volume_Spike'].astype(float) * 0.5
            return (0.6 * vr_s + 0.4 * spike_s).clip(-1, 1)
        return vr_s.clip(-1, 1)

    def score_relative_strength(self, df: pd.DataFrame) -> pd.Series:
        """SPY 대비 상대강도."""
        if 'RS_vs_Benchmark' not in df.columns:
            return pd.Series(0.0, index=df.index)

        # RS 1.0 기준, tanh 정규화
        rs_s = _safe_tanh((df['RS_vs_Benchmark'] - 1.0) * 10)

        if 'RS_Momentum_60d' in df.columns:
            mom_s = _safe_tanh(df['RS_Momentum_60d'].fillna(0) * 5)
            return (0.7 * rs_s + 0.3 * mom_s).clip(-1, 1)
        return rs_s.clip(-1, 1)

    def score_mean_reversion(self, df: pd.DataFrame) -> pd.Series:
        """
        과매도 → 양수(매수), 과매수 → 음수(매도).
        주의: 이 팩터는 momentum/trend와 반대 방향이라 가중치에 주의.
        """
        required = ['Overbought', 'Oversold']
        if not all(c in df.columns for c in required):
            return pd.Series(0.0, index=df.index)

        score = df['Oversold'].astype(float) - df['Overbought'].astype(float)

        # BB 위치로 보조 (Close vs BB mid)
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            bb_width = (df['BB_Upper'] - df['BB_Lower']).replace(0, np.nan)
            bb_pos = (df['Close'] - df['BB_Lower']) / bb_width
            bb_s = (1 - 2 * bb_pos).clip(-1, 1).fillna(0)  # 하단=+1, 상단=-1
            score = 0.6 * score + 0.4 * bb_s

        return score.clip(-1, 1)

    def score_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """VIX 레짐 + Risk_Off_Signal."""
        score = pd.Series(0.0, index=df.index)

        if 'VIX_Regime' in df.columns:
            vix_map = {'low': 0.5, 'normal': 0.2, 'high': -0.3, 'extreme': -0.8}
            vix_s = df['VIX_Regime'].astype(str).map(vix_map).fillna(0)
            score = score + vix_s

        if 'Risk_Off_Signal' in df.columns:
            risk_s = -df['Risk_Off_Signal'].astype(float) * 0.5
            score = score + risk_s

        return score.clip(-1, 1)

    def score_breakout(self, df: pd.DataFrame) -> pd.Series:
        """골든/데드 크로스 + 52주 고점 근접."""
        score = pd.Series(0.0, index=df.index)

        if 'Golden_Cross' in df.columns:
            # 골든크로스 발생 후 20일간 유효
            gc = df['Golden_Cross'].astype(float).rolling(20, min_periods=1).max()
            score = score + gc * 0.5

        if 'Death_Cross' in df.columns:
            dc = df['Death_Cross'].astype(float).rolling(20, min_periods=1).max()
            score = score - dc * 0.5

        # 52주 고점 근접 (Pct_From_52w_High: 0=고점, 음수=아래)
        if 'Pct_From_52w_High' in df.columns:
            # 고점의 5% 이내면 +0.5
            near_high = (df['Pct_From_52w_High'] > -0.05).astype(float) * 0.5
            score = score + near_high

        return score.clip(-1, 1)

    # ------------------------------------------------------------------
    # 시그널 생성
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        8개 팩터 가중합 → total_score → signal.

        Returns:
            원본 컬럼 + 각 팩터 스코어 + total_score + signal
        """
        df = df.copy()

        scores = {
            'momentum': self.score_momentum(df),
            'trend': self.score_trend(df),
            'volatility': self.score_volatility(df),
            'volume': self.score_volume(df),
            'relative_strength': self.score_relative_strength(df),
            'mean_reversion': self.score_mean_reversion(df),
            'market_regime': self.score_market_regime(df),
            'breakout': self.score_breakout(df),
        }

        # 각 팩터 스코어 컬럼으로 기록 (분석/디버깅용)
        for name, s in scores.items():
            df[f'score_{name}'] = s

        # 가중합
        weights = self.params['weights']
        total = pd.Series(0.0, index=df.index)
        for name, w in weights.items():
            total = total + scores[name] * w
        df['total_score'] = total

        # Risk-Off 감쇠: buy 시그널(양수)만 감쇠
        adjusted = total.copy()
        if 'Risk_Off_Signal' in df.columns:
            mult = self.params['risk_off_buy_multiplier']
            risk_off = df['Risk_Off_Signal'].astype(bool)
            adjusted = adjusted.where(
                ~(risk_off & (adjusted > 0)),
                adjusted * mult,
            )

        # 시그널 생성
        buy_th = self.params['buy_threshold']
        sell_th = self.params['sell_threshold']
        df['signal'] = 0
        df.loc[adjusted > buy_th, 'signal'] = 1
        df.loc[adjusted < sell_th, 'signal'] = -1

        return df
