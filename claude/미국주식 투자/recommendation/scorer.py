"""
추천 전용 6팩터 스코어링 엔진

팩터 구성 (단기 스윙 1~4주 최적화):
- 모멘텀 (20%): RSI, 1주/1개월 수익률, MACD 히스토그램
- 트렌드 강도 (20%): ADX, SMA 정렬, 이평선 위 거래일 비율
- 브레이크아웃 (15%): 52주 신고가 근접, BB 상단 돌파, 거래량 급증
- 밸류에이션 (15%): PER/PBR 섹터 상대값, PEG
- 수익성/성장 (15%): ROE, EPS 성장률, 매출 성장률
- 리스크 (15%): ATR 정규화, 부채비율, 베타 (낮을수록 가점)

정규화: 크로스섹션 백분위 → 0~100
"""

import pandas as pd
import numpy as np
import ta

from recommendation.exemplar.features import compute_features_snapshot
from recommendation.exemplar.similarity import score_candidate


FACTOR_WEIGHTS = {
    "momentum": 0.20,
    "trend": 0.20,
    "breakout": 0.15,
    "valuation": 0.15,
    "growth": 0.15,
    "risk": 0.15,
}


def build_weights(exemplar_weight: float = 0.0) -> dict[str, float]:
    """6팩터 + 옵션으로 7번째(exemplar) 팩터 가중치 구성.

    exemplar_weight=0이면 기존 6팩터 딕셔너리 그대로 반환.
    >0이면 6팩터에 (1 - exemplar_weight) 스케일을 곱하고 exemplar 키 추가.
    """
    if exemplar_weight <= 0:
        return dict(FACTOR_WEIGHTS)
    if exemplar_weight > 1:
        raise ValueError(f"exemplar_weight must be in [0, 1], got {exemplar_weight}")
    scale = 1.0 - exemplar_weight
    return {**{k: v * scale for k, v in FACTOR_WEIGHTS.items()}, "exemplar": exemplar_weight}


class RecommendationScorer:
    """추천 전용 스코어링 엔진"""

    def __init__(
        self,
        weights: dict | None = None,
        profiles: list[tuple[str, dict[str, dict[str, float]]]] | None = None,
    ):
        self.weights = weights or FACTOR_WEIGHTS
        self.profiles = profiles or []

    def _compute_momentum(self, df: pd.DataFrame) -> float:
        """모멘텀 팩터: RSI, 단기 수익률, MACD"""
        if len(df) < 26:
            return np.nan

        close = df["Close"]

        # RSI(14) - 50 중심 정규화 (30~70 → 0~100)
        rsi = ta.momentum.rsi(close, window=14).iloc[-1]
        rsi_score = np.clip((rsi - 30) / 40 * 100, 0, 100) if not np.isnan(rsi) else 50

        # 1주 수익률
        if len(close) >= 5:
            ret_1w = (close.iloc[-1] / close.iloc[-5] - 1) * 100
        else:
            ret_1w = 0

        # 1개월 수익률
        if len(close) >= 21:
            ret_1m = (close.iloc[-1] / close.iloc[-21] - 1) * 100
        else:
            ret_1m = 0

        # 수익률 점수 (−20% ~ +20% → 0~100)
        ret_1w_score = np.clip((ret_1w + 20) / 40 * 100, 0, 100)
        ret_1m_score = np.clip((ret_1m + 20) / 40 * 100, 0, 100)

        # MACD 히스토그램 방향
        macd_hist = ta.trend.macd_diff(close).iloc[-1]
        macd_score = 75 if macd_hist > 0 else 25
        if not np.isnan(macd_hist):
            # 히스토그램 증가 중이면 추가 가점
            macd_hist_prev = ta.trend.macd_diff(close).iloc[-2] if len(df) > 26 else 0
            if macd_hist > macd_hist_prev:
                macd_score = min(macd_score + 15, 100)

        return (rsi_score * 0.3 + ret_1w_score * 0.25 + ret_1m_score * 0.25 + macd_score * 0.2)

    def _compute_trend(self, df: pd.DataFrame) -> float:
        """트렌드 강도 팩터: ADX, SMA 정렬, 이평선 위 비율"""
        if len(df) < 200:
            # 데이터 부족 시 짧은 기간으로 대체
            return self._compute_trend_short(df)

        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # ADX(14) - 25 이상이면 강한 트렌드
        adx = ta.trend.adx(high, low, close, window=14).iloc[-1]
        adx_score = np.clip(adx / 50 * 100, 0, 100) if not np.isnan(adx) else 50

        # SMA 정렬: 20 > 50 > 200 (완벽한 상승 정렬)
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1]

        alignment_score = 0
        if close.iloc[-1] > sma20:
            alignment_score += 25
        if sma20 > sma50:
            alignment_score += 25
        if sma50 > sma200:
            alignment_score += 25
        if close.iloc[-1] > sma200:
            alignment_score += 25

        # 최근 20일 중 종가가 SMA20 위에 있는 날 비율
        sma20_series = close.rolling(20).mean()
        above_sma = (close[-20:] > sma20_series[-20:]).sum() / 20 * 100

        return (adx_score * 0.35 + alignment_score * 0.40 + above_sma * 0.25)

    def _compute_trend_short(self, df: pd.DataFrame) -> float:
        """데이터가 200일 미만일 때의 트렌드 계산"""
        if len(df) < 20:
            return np.nan

        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        adx = ta.trend.adx(high, low, close, window=14).iloc[-1]
        adx_score = np.clip(adx / 50 * 100, 0, 100) if not np.isnan(adx) else 50

        sma20 = close.rolling(20).mean().iloc[-1]
        price_above = 100 if close.iloc[-1] > sma20 else 0

        if len(df) >= 50:
            sma50 = close.rolling(50).mean().iloc[-1]
            alignment = 100 if sma20 > sma50 else 0
        else:
            alignment = 50

        return (adx_score * 0.4 + price_above * 0.3 + alignment * 0.3)

    def _compute_breakout(self, df: pd.DataFrame) -> float:
        """브레이크아웃 팩터: 52주 고가 근접, BB 돌파, 거래량 급증"""
        if len(df) < 20:
            return np.nan

        close = df["Close"]
        volume = df["Volume"]

        # 52주 고가 근접도 (현재가 / 52주고가)
        period = min(252, len(df))
        high_52w = close[-period:].max()
        proximity = (close.iloc[-1] / high_52w) * 100 if high_52w > 0 else 50
        proximity_score = np.clip(proximity, 0, 100)

        # 볼린저밴드 상단 돌파
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_upper = bb.bollinger_hband().iloc[-1]
        bb_score = 100 if close.iloc[-1] > bb_upper else (close.iloc[-1] / bb_upper * 100 if bb_upper > 0 else 50)
        bb_score = np.clip(bb_score, 0, 100)

        # 거래량 급증 (현재 거래량 / 20일 평균)
        vol_avg = volume.rolling(20).mean().iloc[-1]
        if vol_avg > 0 and not np.isnan(vol_avg):
            vol_ratio = volume.iloc[-1] / vol_avg
            vol_score = np.clip(vol_ratio / 3 * 100, 0, 100)  # 3배면 100점
        else:
            vol_score = 50

        return (proximity_score * 0.40 + bb_score * 0.30 + vol_score * 0.30)

    def _compute_valuation(self, info: dict) -> float:
        """밸류에이션 팩터: PER, PBR, PEG (낮을수록 가점)"""
        scores = []

        # Forward PER (0~50이 정상 범위, 낮을수록 좋음)
        per = info.get("forwardPE") or info.get("trailingPE")
        if per and 0 < per < 200:
            per_score = np.clip((50 - per) / 50 * 100, 0, 100)
            scores.append(per_score)

        # PBR (0~10이 정상 범위)
        pbr = info.get("priceToBook")
        if pbr and 0 < pbr < 50:
            pbr_score = np.clip((10 - pbr) / 10 * 100, 0, 100)
            scores.append(pbr_score)

        # PEG (1 미만이면 저평가)
        peg = info.get("trailingPegRatio") or info.get("pegRatio")
        if peg and 0 < peg < 10:
            peg_score = np.clip((2 - peg) / 2 * 100, 0, 100)
            scores.append(peg_score)

        return np.mean(scores) if scores else 50.0

    def _compute_growth(self, info: dict) -> float:
        """수익성/성장 팩터: ROE, EPS 성장, 매출 성장"""
        scores = []

        # ROE (0~50% 정상 범위, 높을수록 좋음)
        roe = info.get("returnOnEquity")
        if roe is not None and -1 < roe < 2:
            roe_score = np.clip(roe * 100 / 0.3 * 100, 0, 100)  # 30%면 100점
            scores.append(roe_score)

        # EPS 성장률
        eps_growth = info.get("earningsGrowth")
        if eps_growth is not None and -1 < eps_growth < 5:
            growth_score = np.clip((eps_growth + 0.2) / 0.7 * 100, 0, 100)  # -20%~+50% → 0~100
            scores.append(growth_score)

        # 매출 성장률
        rev_growth = info.get("revenueGrowth")
        if rev_growth is not None and -1 < rev_growth < 5:
            rev_score = np.clip((rev_growth + 0.1) / 0.4 * 100, 0, 100)  # -10%~+30% → 0~100
            scores.append(rev_score)

        return np.mean(scores) if scores else 50.0

    def _compute_risk(self, df: pd.DataFrame, info: dict) -> float:
        """리스크 팩터: 변동성, 부채비율, 베타 (낮을수록 가점)"""
        scores = []

        # ATR/Close 정규화 (낮을수록 안정적)
        if len(df) >= 14:
            atr = ta.volatility.average_true_range(
                df["High"], df["Low"], df["Close"], window=14
            ).iloc[-1]
            close = df["Close"].iloc[-1]
            if close > 0 and not np.isnan(atr):
                atr_pct = atr / close * 100  # 0~10% 범위
                atr_score = np.clip((5 - atr_pct) / 5 * 100, 0, 100)
                scores.append(atr_score)

        # 부채비율 (낮을수록 좋음, 0~200% 범위)
        dte = info.get("debtToEquity")
        if dte is not None and 0 <= dte < 500:
            dte_score = np.clip((200 - dte) / 200 * 100, 0, 100)
            scores.append(dte_score)

        # 베타 (1 근처가 시장 평균, 낮을수록 안정적)
        beta = info.get("beta")
        if beta is not None and 0 < beta < 5:
            beta_score = np.clip((2 - beta) / 2 * 100, 0, 100)
            scores.append(beta_score)

        return np.mean(scores) if scores else 50.0

    def score_single(self, ticker_data: dict) -> dict | None:
        """
        단일 종목 스코어링.

        Args:
            ticker_data: {"ticker": str, "price_df": DataFrame, "info": dict}

        Returns:
            {"ticker": str, "momentum": float, ..., "total_score": float}
        """
        ticker = ticker_data["ticker"]
        df = ticker_data["price_df"]
        info = ticker_data.get("info", {})

        if df is None or len(df) < 20:
            return None

        try:
            momentum = self._compute_momentum(df)
            trend = self._compute_trend(df)
            breakout = self._compute_breakout(df)
            valuation = self._compute_valuation(info)
            growth = self._compute_growth(info)
            risk = self._compute_risk(df, info)

            # NaN 처리: 계산 불가능한 팩터는 50 (중립)
            factors = {
                "momentum": momentum if not np.isnan(momentum) else 50.0,
                "trend": trend if not np.isnan(trend) else 50.0,
                "breakout": breakout if not np.isnan(breakout) else 50.0,
                "valuation": valuation,
                "growth": growth,
                "risk": risk,
            }

            exemplar_score: float | None = None
            best_match_id: str | None = None
            if self.profiles:
                try:
                    snapshot = compute_features_snapshot(df)
                    sim = score_candidate(snapshot, self.profiles)
                    exemplar_score = sim.score
                    best_match_id = sim.best_match_id
                except ValueError:
                    exemplar_score = None
                    best_match_id = None

            if exemplar_score is not None:
                factors["exemplar"] = exemplar_score

            total = sum(factors[k] * self.weights[k] for k in self.weights if k in factors)

            result = {
                "ticker": ticker,
                **factors,
                "total_score": round(total, 2),
            }
            if exemplar_score is not None:
                result["exemplar_similarity"] = exemplar_score
                result["best_match_id"] = best_match_id
            return result
        except Exception:
            return None

    def score_all(self, data_list: list[dict]) -> pd.DataFrame:
        """
        전체 종목 스코어링 후 크로스섹션 백분위 정규화.

        Args:
            data_list: fetch_ticker_data 결과 리스트

        Returns:
            정규화된 스코어 DataFrame
        """
        raw_scores = []
        for item in tqdm(data_list, desc="Scoring"):
            result = self.score_single(item)
            if result:
                raw_scores.append(result)

        if not raw_scores:
            return pd.DataFrame()

        df = pd.DataFrame(raw_scores)

        # 크로스섹션 백분위 정규화 (각 팩터를 전체 유니버스 내 상대 순위로)
        factor_cols = list(self.weights.keys())
        for col in factor_cols:
            if col in df.columns:
                df[col] = df[col].rank(pct=True) * 100

        # exemplar 정규화 결과를 UI 표시용 컬럼에도 반영
        if "exemplar" in df.columns:
            df["exemplar_similarity"] = df["exemplar"]

        # 정규화 후 total_score 재계산
        df["total_score"] = sum(
            df[factor] * self.weights[factor]
            for factor in factor_cols
            if factor in df.columns
        )
        df["total_score"] = df["total_score"].round(2)

        return df


# tqdm import for score_all
from tqdm import tqdm
