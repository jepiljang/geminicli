"""실 yfinance 데이터로 sanity check. 외부 API 의존 → slow 마커."""

from datetime import date

import pytest

import yfinance as yf

from recommendation.exemplar.features import compute_features_snapshot
from recommendation.exemplar.profile import build_profile_from_df
from recommendation.exemplar.similarity import score_candidate


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def nvda_2023_rally_profile():
    df = yf.Ticker("NVDA").history(start="2023-01-01", end="2023-07-01", auto_adjust=True)
    assert not df.empty
    return build_profile_from_df(df)


def _snapshot(ticker: str, end_date: date) -> dict:
    df = yf.Ticker(ticker).history(end=end_date.isoformat(), period="2y", auto_adjust=True)
    return compute_features_snapshot(df)


def test_self_match_high(nvda_2023_rally_profile):
    """NVDA 자신의 2023-04-01 스냅샷은 모범에 매우 가까워야 함."""
    snap = _snapshot("NVDA", date(2023, 4, 1))
    result = score_candidate(snap, [("nvda-rally", nvda_2023_rally_profile)])
    assert result.score is not None
    assert result.score > 70.0


def test_different_period_lower(nvda_2023_rally_profile):
    """NVDA의 2018-12 (하락 구간) 스냅샷은 점수가 자기 매칭보다 낮아야 함."""
    snap_self = _snapshot("NVDA", date(2023, 4, 1))
    snap_other = _snapshot("NVDA", date(2018, 12, 28))
    r_self = score_candidate(snap_self, [("p", nvda_2023_rally_profile)])
    r_other = score_candidate(snap_other, [("p", nvda_2023_rally_profile)])
    assert r_other.score < r_self.score


def test_unrelated_ticker_lower(nvda_2023_rally_profile):
    """KO(코카콜라)는 NVDA 랠리 패턴과 거리가 멀어야 함."""
    snap_nvda = _snapshot("NVDA", date(2023, 4, 1))
    snap_ko = _snapshot("KO", date(2023, 4, 1))
    r_nvda = score_candidate(snap_nvda, [("p", nvda_2023_rally_profile)])
    r_ko = score_candidate(snap_ko, [("p", nvda_2023_rally_profile)])
    assert r_ko.score < r_nvda.score
