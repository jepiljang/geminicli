"""모범 프로파일 빌드/저장/로드.

프로파일 = {feature_key: {"mean": μ, "std": σ}}.
파일 포맷: parquet (feature, mean, std 컬럼).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from recommendation.exemplar.features import FEATURE_KEYS, compute_features_series


def build_profile_from_df(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """OHLCV → 10피처 시계열 → {피처: {mean, std}}."""
    series = compute_features_series(df)
    profile: dict[str, dict[str, float]] = {}
    for key in FEATURE_KEYS:
        col = series[key]
        profile[key] = {"mean": float(col.mean()), "std": float(col.std(ddof=0))}
    return profile


def build_profile(ticker: str, start: date, end: date) -> dict[str, dict[str, float]]:
    """티커 + 구간으로 yfinance에서 OHLCV 수집 → 프로파일 빌드.

    Note: yfinance 호출이 들어가므로 단위 테스트는 build_profile_from_df를 사용한다.
    """
    import yfinance as yf

    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if df.empty:
        raise ValueError(f"no data for {ticker} {start}~{end}")
    return build_profile_from_df(df)


def save_profile(path: Path, profile: dict[str, dict[str, float]]) -> None:
    """프로파일을 parquet로 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"feature": k, "mean": v["mean"], "std": v["std"]} for k, v in profile.items()]
    pd.DataFrame(rows).to_parquet(path, index=False)


def load_profile(path: Path) -> dict[str, dict[str, float]]:
    """parquet에서 프로파일 로드.

    Raises:
        FileNotFoundError: 파일 없음.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_parquet(path)
    return {
        row["feature"]: {"mean": float(row["mean"]), "std": float(row["std"])}
        for _, row in df.iterrows()
    }
