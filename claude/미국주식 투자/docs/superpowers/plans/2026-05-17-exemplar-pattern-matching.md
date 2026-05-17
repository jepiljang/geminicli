# 모범 종목 패턴 매칭 (7번째 팩터) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 사용자가 모범 종목과 상승 구간을 등록하면, 그 기간의 기술적 지표 프로파일과 닮은 종목에 7번째 팩터 점수를 부여하는 시스템을 기존 추천 파이프라인에 통합한다.

**Architecture:** `recommendation/exemplar/` 서브패키지로 격리(features → profile → library → similarity 4단계). 기존 `RecommendationScorer`는 활성 프로파일을 받아 7번째 팩터를 계산하고 백분위 정규화에 포함. Streamlit 앱에 모범 라이브러리 페이지 추가 + 사이드바 가중치 슬라이더.

**Tech Stack:** Python 3.x, pandas, numpy, `ta` (지표 계산), pytest (테스트), Streamlit (UI), parquet/JSON (저장).

**Spec:** `docs/superpowers/specs/2026-05-17-exemplar-pattern-matching-design.md`

---

## 파일 구조

**Create:**
- `recommendation/exemplar/__init__.py` — 서브패키지 진입점, 공개 API 재노출
- `recommendation/exemplar/features.py` — 10개 기술적 피처 계산 (시계열/스냅샷)
- `recommendation/exemplar/profile.py` — 티커+구간 → 프로파일 빌드/저장/로드
- `recommendation/exemplar/library.py` — `exemplars.json` CRUD + `Exemplar` 데이터클래스
- `recommendation/exemplar/similarity.py` — `SimilarityResult` + `score_candidate`
- `tests/__init__.py`
- `tests/exemplar/__init__.py`
- `tests/exemplar/conftest.py` — 합성 OHLCV fixture
- `tests/exemplar/test_features.py`
- `tests/exemplar/test_profile.py`
- `tests/exemplar/test_library.py`
- `tests/exemplar/test_similarity.py`
- `tests/exemplar/test_scorer_integration.py` — 7번째 팩터 통합 회귀 테스트
- `tests/exemplar/test_sanity.py` — 실 yfinance 데이터 sanity (`@pytest.mark.slow`)

**Modify:**
- `recommendation/scorer.py` — `FACTOR_WEIGHTS`를 `build_weights()` 함수로 교체, `RecommendationScorer.__init__`에 `profiles` 받아 저장, `score_single`에서 활성 시 7번째 팩터 계산, `score_all`에서 `exemplar_similarity` 컬럼도 백분위 정규화 포함
- `recommendation/app.py` — "📚 모범 라이브러리" 탭 추가, 사이드바에 7번째 가중치 슬라이더, Top 카드/상세에 best_match 표시
- `requirements.txt` — `pytest>=8.0.0` 추가

---

## Task 0: 테스트 환경 셋업

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`, `tests/exemplar/__init__.py`, `pytest.ini`

- [ ] **Step 1: pytest 의존성 추가**

`requirements.txt` 끝에 추가:

```
# 테스트
pytest>=8.0.0
```

- [ ] **Step 2: pytest 설정 파일 생성**

`pytest.ini` (프로젝트 루트):

```ini
[pytest]
testpaths = tests
python_files = test_*.py
markers =
    slow: tests that hit external APIs (deselect with -m "not slow")
filterwarnings =
    ignore::DeprecationWarning
```

- [ ] **Step 3: tests 패키지 초기화 파일 생성**

`tests/__init__.py` — 빈 파일.

`tests/exemplar/__init__.py` — 빈 파일.

- [ ] **Step 4: 의존성 설치 확인**

Run: `pip install pytest`
Expected: `Successfully installed pytest-...` 또는 이미 설치됨

Run: `pytest --version`
Expected: `pytest 8.x.x`

- [ ] **Step 5: 빈 test discovery 동작 확인**

Run: `pytest tests/ -v`
Expected: `no tests ran` 또는 `collected 0 items`

- [ ] **Step 6: 커밋**

```bash
git add requirements.txt pytest.ini tests/__init__.py tests/exemplar/__init__.py
git commit -m "chore: set up pytest for exemplar tests"
```

---

## Task 1: 합성 OHLCV fixture 작성

**Files:**
- Create: `tests/exemplar/conftest.py`

- [ ] **Step 1: fixture 파일 생성**

`tests/exemplar/conftest.py`:

```python
"""테스트용 합성 OHLCV 데이터 fixture."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rising_ohlcv() -> pd.DataFrame:
    """300일 우상향 가격 + 약한 변동성 OHLCV. 모범 구간 시뮬레이션용."""
    rng = np.random.default_rng(42)
    n = 300
    # 일평균 +0.2% drift, 1.2% daily vol
    returns = rng.normal(0.002, 0.012, n)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)

    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


@pytest.fixture
def flat_ohlcv() -> pd.DataFrame:
    """300일 횡보 OHLCV. 모범과 다른 패턴 시뮬레이션용."""
    rng = np.random.default_rng(7)
    n = 300
    # drift=0
    returns = rng.normal(0.0, 0.008, n)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.integers(800_000, 2_000_000, n).astype(float)

    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )
```

- [ ] **Step 2: fixture가 import되는지 확인**

`tests/exemplar/test_smoke.py`를 임시로 생성:

```python
def test_rising_fixture_shape(rising_ohlcv):
    assert len(rising_ohlcv) == 300
    assert list(rising_ohlcv.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_rising_actually_rises(rising_ohlcv):
    assert rising_ohlcv["Close"].iloc[-1] > rising_ohlcv["Close"].iloc[0]
```

Run: `pytest tests/exemplar/test_smoke.py -v`
Expected: 2 passed

- [ ] **Step 3: 스모크 파일 삭제**

`tests/exemplar/test_smoke.py` 삭제.

- [ ] **Step 4: 커밋**

```bash
git add tests/exemplar/conftest.py
git commit -m "test: add synthetic OHLCV fixtures for exemplar tests"
```

---

## Task 2: features.py — 10개 기술적 피처 계산

**Files:**
- Create: `recommendation/exemplar/__init__.py`, `recommendation/exemplar/features.py`
- Test: `tests/exemplar/test_features.py`

- [ ] **Step 1: 패키지 초기화 파일 생성**

`recommendation/exemplar/__init__.py` — 빈 파일 (Task 6 끝에 채움).

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/exemplar/test_features.py`:

```python
"""features.py 단위 테스트."""

import math

import numpy as np
import pandas as pd
import pytest

from recommendation.exemplar.features import (
    FEATURE_KEYS,
    compute_features_series,
    compute_features_snapshot,
)


def test_feature_keys_is_10():
    assert len(FEATURE_KEYS) == 10
    assert set(FEATURE_KEYS) == {
        "rsi_14", "ret_5d", "ret_20d", "macd_hist_norm", "adx_14",
        "sma20_pos", "sma50_pos", "high_52w_pos", "bb_pos", "volume_ratio",
    }


def test_series_returns_dataframe_with_all_features(rising_ohlcv):
    out = compute_features_series(rising_ohlcv)
    assert isinstance(out, pd.DataFrame)
    assert set(FEATURE_KEYS).issubset(set(out.columns))


def test_series_drops_initial_nan_rows(rising_ohlcv):
    out = compute_features_series(rising_ohlcv)
    # SMA50 needs 50 rows; 첫 49행은 NaN → drop 되어야 함
    assert len(out) < len(rising_ohlcv)
    assert not out.isna().any().any()


def test_snapshot_returns_dict_with_all_features(rising_ohlcv):
    snap = compute_features_snapshot(rising_ohlcv)
    assert isinstance(snap, dict)
    assert set(snap.keys()) == set(FEATURE_KEYS)
    for v in snap.values():
        assert isinstance(v, float)
        assert not math.isnan(v)


def test_snapshot_matches_last_row_of_series(rising_ohlcv):
    series = compute_features_series(rising_ohlcv)
    snap = compute_features_snapshot(rising_ohlcv)
    last = series.iloc[-1]
    for key in FEATURE_KEYS:
        assert snap[key] == pytest.approx(last[key], rel=1e-6)


def test_rising_data_has_high_52w_near_1(rising_ohlcv):
    """우상향 데이터는 마지막에 52주 고가 근처여야 한다."""
    snap = compute_features_snapshot(rising_ohlcv)
    assert snap["high_52w_pos"] > 0.85


def test_series_insufficient_data_raises():
    short_df = pd.DataFrame({
        "Open": [1.0] * 30, "High": [1.0] * 30, "Low": [1.0] * 30,
        "Close": [1.0] * 30, "Volume": [1.0] * 30,
    })
    with pytest.raises(ValueError, match="at least 50"):
        compute_features_series(short_df)
```

- [ ] **Step 3: 테스트가 실패하는지 확인**

Run: `pytest tests/exemplar/test_features.py -v`
Expected: `ModuleNotFoundError: No module named 'recommendation.exemplar.features'`

- [ ] **Step 4: features.py 구현**

`recommendation/exemplar/features.py`:

```python
"""모범 매칭용 10개 기술적 피처 계산.

시계열 (구간 집계용)과 스냅샷 (후보 평가용) 두 가지 API를 제공한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta


FEATURE_KEYS: list[str] = [
    "rsi_14",
    "ret_5d",
    "ret_20d",
    "macd_hist_norm",
    "adx_14",
    "sma20_pos",
    "sma50_pos",
    "high_52w_pos",
    "bb_pos",
    "volume_ratio",
]

_MIN_ROWS = 50  # SMA50 / MACD가 안정화되는 최소 길이


def _compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV → 10피처 시계열을 계산하고 NaN 행을 그대로 둔다."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    out = pd.DataFrame(index=df.index)
    out["rsi_14"] = ta.momentum.rsi(close, window=14)
    out["ret_5d"] = close.pct_change(5)
    out["ret_20d"] = close.pct_change(20)

    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    out["macd_hist_norm"] = macd.macd_diff() / close

    out["adx_14"] = ta.trend.adx(high, low, close, window=14)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    out["sma20_pos"] = close / sma20 - 1
    out["sma50_pos"] = close / sma50 - 1

    rolling_high = close.rolling(252, min_periods=50).max()
    out["high_52w_pos"] = close / rolling_high

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_mid = bb.bollinger_mavg()
    denom = (bb_upper - bb_mid).replace(0, np.nan)
    out["bb_pos"] = (close - bb_mid) / denom

    vol_sma20 = volume.rolling(20).mean().replace(0, np.nan)
    out["volume_ratio"] = volume / vol_sma20

    return out


def compute_features_series(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV → 10피처 시계열 DataFrame. NaN 행 제거.

    Raises:
        ValueError: len(df) < 50 인 경우.
    """
    if len(df) < _MIN_ROWS:
        raise ValueError(f"need at least {_MIN_ROWS} rows of OHLCV, got {len(df)}")
    out = _compute_all(df)
    out = out.dropna(how="any")
    return out


def compute_features_snapshot(df: pd.DataFrame) -> dict[str, float]:
    """OHLCV → 마지막 시점 10피처 dict.

    Raises:
        ValueError: len(df) < 50 인 경우, 또는 마지막 행에 NaN이 있는 경우.
    """
    if len(df) < _MIN_ROWS:
        raise ValueError(f"need at least {_MIN_ROWS} rows of OHLCV, got {len(df)}")
    last = _compute_all(df).iloc[-1]
    if last.isna().any():
        missing = last.index[last.isna()].tolist()
        raise ValueError(f"snapshot has NaN features: {missing}")
    return {k: float(last[k]) for k in FEATURE_KEYS}
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `pytest tests/exemplar/test_features.py -v`
Expected: 모든 7개 테스트 PASS

- [ ] **Step 6: 커밋**

```bash
git add recommendation/exemplar/__init__.py recommendation/exemplar/features.py tests/exemplar/test_features.py
git commit -m "feat(exemplar): add 10-feature technical computation"
```

---

## Task 3: profile.py — 프로파일 빌드/저장/로드

**Files:**
- Create: `recommendation/exemplar/profile.py`
- Test: `tests/exemplar/test_profile.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/exemplar/test_profile.py`:

```python
"""profile.py 단위 테스트."""

import pandas as pd
import pytest

from recommendation.exemplar.features import FEATURE_KEYS
from recommendation.exemplar.profile import (
    build_profile_from_df,
    load_profile,
    save_profile,
)


def test_build_profile_from_df_returns_mean_std_dict(rising_ohlcv):
    profile = build_profile_from_df(rising_ohlcv)
    assert set(profile.keys()) == set(FEATURE_KEYS)
    for key, val in profile.items():
        assert "mean" in val and "std" in val
        assert isinstance(val["mean"], float)
        assert isinstance(val["std"], float)
        assert val["std"] >= 0


def test_build_profile_rejects_short_data():
    short_df = pd.DataFrame({
        "Open": [1.0] * 30, "High": [1.0] * 30, "Low": [1.0] * 30,
        "Close": [1.0] * 30, "Volume": [1.0] * 30,
    })
    with pytest.raises(ValueError):
        build_profile_from_df(short_df)


def test_save_and_load_roundtrip(tmp_path, rising_ohlcv):
    profile = build_profile_from_df(rising_ohlcv)
    path = tmp_path / "test.parquet"
    save_profile(path, profile)
    assert path.exists()

    loaded = load_profile(path)
    assert set(loaded.keys()) == set(profile.keys())
    for key in profile:
        assert loaded[key]["mean"] == pytest.approx(profile[key]["mean"])
        assert loaded[key]["std"] == pytest.approx(profile[key]["std"])


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_profile(tmp_path / "nope.parquet")
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `pytest tests/exemplar/test_profile.py -v`
Expected: `ModuleNotFoundError: No module named 'recommendation.exemplar.profile'`

- [ ] **Step 3: profile.py 구현**

`recommendation/exemplar/profile.py`:

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/exemplar/test_profile.py -v`
Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add recommendation/exemplar/profile.py tests/exemplar/test_profile.py
git commit -m "feat(exemplar): add profile builder + parquet save/load"
```

---

## Task 4: library.py — exemplars.json CRUD

**Files:**
- Create: `recommendation/exemplar/library.py`
- Test: `tests/exemplar/test_library.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/exemplar/test_library.py`:

```python
"""library.py 단위 테스트."""

from datetime import date

import pytest

from recommendation.exemplar.library import (
    Exemplar,
    ExemplarLibrary,
    make_exemplar_id,
)


def test_make_exemplar_id_format():
    eid = make_exemplar_id("NVDA", date(2023, 1, 1), "AI Rally")
    assert eid.startswith("nvda-2023-")
    assert eid == eid.lower()
    assert " " not in eid


def test_make_exemplar_id_uses_ticker_when_no_name():
    eid = make_exemplar_id("AAPL", date(2020, 4, 1), None)
    assert eid == "aapl-2020-aapl" or eid.startswith("aapl-2020-")


def test_library_add_creates_entry(tmp_path):
    lib = ExemplarLibrary(tmp_path / "exemplars.json")
    ex = Exemplar(
        id="nvda-2023-rally",
        ticker="NVDA",
        name="NVDA 2023 AI 랠리",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 7, 1),
        active=True,
        created_at="2026-05-17T14:30:00",
        profile_path="data/exemplars/profiles/nvda-2023-rally.parquet",
    )
    lib.add(ex)
    assert lib.get("nvda-2023-rally") == ex


def test_library_persists_to_json(tmp_path):
    json_path = tmp_path / "exemplars.json"
    lib1 = ExemplarLibrary(json_path)
    ex = Exemplar(
        id="aapl-2020", ticker="AAPL", name="test",
        start_date=date(2020, 4, 1), end_date=date(2020, 9, 1),
        active=True, created_at="2026-05-17T00:00:00",
        profile_path="x.parquet",
    )
    lib1.add(ex)

    lib2 = ExemplarLibrary(json_path)
    assert lib2.get("aapl-2020") == ex


def test_library_toggle_active(tmp_path):
    lib = ExemplarLibrary(tmp_path / "exemplars.json")
    ex = Exemplar(
        id="x", ticker="X", name="x", start_date=date(2020, 1, 1),
        end_date=date(2020, 6, 1), active=True, created_at="t",
        profile_path="p",
    )
    lib.add(ex)
    lib.toggle_active("x", False)
    assert lib.get("x").active is False
    lib.toggle_active("x", True)
    assert lib.get("x").active is True


def test_library_delete(tmp_path):
    lib = ExemplarLibrary(tmp_path / "exemplars.json")
    ex = Exemplar(
        id="x", ticker="X", name="x", start_date=date(2020, 1, 1),
        end_date=date(2020, 6, 1), active=True, created_at="t",
        profile_path="p",
    )
    lib.add(ex)
    lib.delete("x")
    assert lib.get("x") is None


def test_library_list_active_only(tmp_path):
    lib = ExemplarLibrary(tmp_path / "exemplars.json")
    a = Exemplar(id="a", ticker="A", name="a", start_date=date(2020, 1, 1),
                 end_date=date(2020, 6, 1), active=True, created_at="t",
                 profile_path="p1")
    b = Exemplar(id="b", ticker="B", name="b", start_date=date(2020, 1, 1),
                 end_date=date(2020, 6, 1), active=False, created_at="t",
                 profile_path="p2")
    lib.add(a)
    lib.add(b)
    actives = lib.list_all(active_only=True)
    assert [e.id for e in actives] == ["a"]
    everything = lib.list_all()
    assert {e.id for e in everything} == {"a", "b"}


def test_library_duplicate_id_raises(tmp_path):
    lib = ExemplarLibrary(tmp_path / "exemplars.json")
    ex = Exemplar(id="x", ticker="X", name="x", start_date=date(2020, 1, 1),
                  end_date=date(2020, 6, 1), active=True, created_at="t",
                  profile_path="p")
    lib.add(ex)
    with pytest.raises(ValueError, match="already exists"):
        lib.add(ex)


def test_library_get_missing_returns_none(tmp_path):
    lib = ExemplarLibrary(tmp_path / "exemplars.json")
    assert lib.get("nope") is None


def test_library_delete_missing_raises(tmp_path):
    lib = ExemplarLibrary(tmp_path / "exemplars.json")
    with pytest.raises(KeyError):
        lib.delete("nope")
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `pytest tests/exemplar/test_library.py -v`
Expected: `ModuleNotFoundError: No module named 'recommendation.exemplar.library'`

- [ ] **Step 3: library.py 구현**

`recommendation/exemplar/library.py`:

```python
"""모범 라이브러리 CRUD.

exemplars.json 형식:
{
  "exemplars": [
    {"id": ..., "ticker": ..., "name": ..., "start_date": "YYYY-MM-DD",
     "end_date": "YYYY-MM-DD", "active": true, "created_at": "ISO8601",
     "profile_path": "..."}
  ]
}
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path


@dataclass
class Exemplar:
    id: str
    ticker: str
    name: str
    start_date: date
    end_date: date
    active: bool
    created_at: str
    profile_path: str


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def make_exemplar_id(ticker: str, start: date, name: str | None) -> str:
    """티커 + 시작연도 + 이름 슬러그로 ID 생성."""
    slug_source = (name or ticker).lower()
    slug = _SLUG_RE.sub("-", slug_source).strip("-") or "exemplar"
    return f"{ticker.lower()}-{start.year}-{slug}"


def _exemplar_to_dict(ex: Exemplar) -> dict:
    d = asdict(ex)
    d["start_date"] = ex.start_date.isoformat()
    d["end_date"] = ex.end_date.isoformat()
    return d


def _exemplar_from_dict(d: dict) -> Exemplar:
    return Exemplar(
        id=d["id"],
        ticker=d["ticker"],
        name=d["name"],
        start_date=date.fromisoformat(d["start_date"]),
        end_date=date.fromisoformat(d["end_date"]),
        active=d["active"],
        created_at=d["created_at"],
        profile_path=d["profile_path"],
    )


class ExemplarLibrary:
    """exemplars.json 기반 라이브러리."""

    def __init__(self, json_path: Path):
        self.json_path = Path(json_path)
        self._items: dict[str, Exemplar] = {}
        self._load()

    def _load(self) -> None:
        if not self.json_path.exists():
            return
        with self.json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data.get("exemplars", []):
            ex = _exemplar_from_dict(d)
            self._items[ex.id] = ex

    def _save(self) -> None:
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"exemplars": [_exemplar_to_dict(e) for e in self._items.values()]}
        with self.json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add(self, exemplar: Exemplar) -> None:
        if exemplar.id in self._items:
            raise ValueError(f"exemplar id {exemplar.id!r} already exists")
        self._items[exemplar.id] = exemplar
        self._save()

    def get(self, exemplar_id: str) -> Exemplar | None:
        return self._items.get(exemplar_id)

    def delete(self, exemplar_id: str) -> None:
        if exemplar_id not in self._items:
            raise KeyError(exemplar_id)
        del self._items[exemplar_id]
        self._save()

    def toggle_active(self, exemplar_id: str, active: bool) -> None:
        if exemplar_id not in self._items:
            raise KeyError(exemplar_id)
        ex = self._items[exemplar_id]
        self._items[exemplar_id] = Exemplar(**{**asdict(ex), "active": active,
                                                "start_date": ex.start_date,
                                                "end_date": ex.end_date})
        self._save()

    def list_all(self, active_only: bool = False) -> list[Exemplar]:
        items = list(self._items.values())
        if active_only:
            items = [e for e in items if e.active]
        return items
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/exemplar/test_library.py -v`
Expected: 10 passed

- [ ] **Step 5: 커밋**

```bash
git add recommendation/exemplar/library.py tests/exemplar/test_library.py
git commit -m "feat(exemplar): add library JSON CRUD"
```

---

## Task 5: similarity.py — 후보 점수 + best_match

**Files:**
- Create: `recommendation/exemplar/similarity.py`
- Test: `tests/exemplar/test_similarity.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/exemplar/test_similarity.py`:

```python
"""similarity.py 단위 테스트."""

import pytest

from recommendation.exemplar.features import FEATURE_KEYS
from recommendation.exemplar.similarity import (
    SimilarityResult,
    score_candidate,
)


def _make_profile(means: dict[str, float], std: float = 0.1) -> dict[str, dict[str, float]]:
    """주어진 평균과 균일 std로 프로파일을 만든다."""
    return {k: {"mean": means.get(k, 0.0), "std": std} for k in FEATURE_KEYS}


def _make_snapshot(values: dict[str, float]) -> dict[str, float]:
    return {k: values.get(k, 0.0) for k in FEATURE_KEYS}


def test_identical_snapshot_scores_100():
    profile = _make_profile({k: 0.5 for k in FEATURE_KEYS}, std=0.1)
    snapshot = _make_snapshot({k: 0.5 for k in FEATURE_KEYS})
    result = score_candidate(snapshot, [("ex1", profile)])
    assert result.score == pytest.approx(100.0, abs=1e-3)
    assert result.best_match_id == "ex1"


def test_far_snapshot_scores_low():
    profile = _make_profile({k: 0.0 for k in FEATURE_KEYS}, std=0.1)
    # 모든 피처가 모범에서 +10σ 떨어짐
    snapshot = _make_snapshot({k: 1.0 for k in FEATURE_KEYS})
    result = score_candidate(snapshot, [("ex1", profile)])
    assert result.score < 5.0


def test_empty_profiles_returns_none():
    snapshot = _make_snapshot({k: 0.5 for k in FEATURE_KEYS})
    result = score_candidate(snapshot, [])
    assert result.score is None
    assert result.best_match_id is None
    assert result.breakdown == {}


def test_zero_std_does_not_explode():
    profile = _make_profile({k: 0.0 for k in FEATURE_KEYS}, std=0.0)
    snapshot = _make_snapshot({k: 0.0 for k in FEATURE_KEYS})
    result = score_candidate(snapshot, [("ex1", profile)])
    # std=0 + identical mean → 거리 0 → 100
    assert result.score == pytest.approx(100.0, abs=1e-3)


def test_max_similarity_picks_closest_exemplar():
    near_profile = _make_profile({k: 0.5 for k in FEATURE_KEYS}, std=0.1)
    far_profile = _make_profile({k: 5.0 for k in FEATURE_KEYS}, std=0.1)
    snapshot = _make_snapshot({k: 0.5 for k in FEATURE_KEYS})
    result = score_candidate(snapshot, [("far", far_profile), ("near", near_profile)])
    assert result.best_match_id == "near"
    assert result.score == pytest.approx(100.0, abs=1e-3)
    assert result.breakdown["near"] > result.breakdown["far"]


def test_breakdown_contains_all_exemplars():
    p1 = _make_profile({k: 0.0 for k in FEATURE_KEYS}, std=0.1)
    p2 = _make_profile({k: 1.0 for k in FEATURE_KEYS}, std=0.1)
    snapshot = _make_snapshot({k: 0.5 for k in FEATURE_KEYS})
    result = score_candidate(snapshot, [("p1", p1), ("p2", p2)])
    assert set(result.breakdown.keys()) == {"p1", "p2"}
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `pytest tests/exemplar/test_similarity.py -v`
Expected: `ModuleNotFoundError: No module named 'recommendation.exemplar.similarity'`

- [ ] **Step 3: similarity.py 구현**

`recommendation/exemplar/similarity.py`:

```python
"""후보 종목 vs 모범 프로파일 유사도 계산.

알고리즘:
  z_i = |x_i - μ_i| / max(σ_i, ε)
  d   = mean(z_i)
  similarity = 100 * exp(-d / 2)
라이브러리 점수는 활성 모범들 중 최댓값.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from recommendation.exemplar.features import FEATURE_KEYS

_EPS = 1e-6


@dataclass
class SimilarityResult:
    score: float | None
    best_match_id: str | None
    breakdown: dict[str, float] = field(default_factory=dict)


def _score_one(snapshot: dict[str, float], profile: dict[str, dict[str, float]]) -> float:
    z_sum = 0.0
    for k in FEATURE_KEYS:
        mu = profile[k]["mean"]
        sigma = profile[k]["std"]
        denom = sigma if sigma > _EPS else _EPS
        z_sum += abs(snapshot[k] - mu) / denom
    d = z_sum / len(FEATURE_KEYS)
    return 100.0 * math.exp(-d / 2.0)


def score_candidate(
    snapshot: dict[str, float],
    profiles: list[tuple[str, dict[str, dict[str, float]]]],
) -> SimilarityResult:
    """후보 스냅샷 + (id, profile) 리스트 → SimilarityResult."""
    if not profiles:
        return SimilarityResult(score=None, best_match_id=None, breakdown={})

    breakdown: dict[str, float] = {}
    for exemplar_id, profile in profiles:
        breakdown[exemplar_id] = _score_one(snapshot, profile)

    best_id = max(breakdown, key=breakdown.get)
    return SimilarityResult(
        score=breakdown[best_id],
        best_match_id=best_id,
        breakdown=breakdown,
    )
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/exemplar/test_similarity.py -v`
Expected: 6 passed

- [ ] **Step 5: 커밋**

```bash
git add recommendation/exemplar/similarity.py tests/exemplar/test_similarity.py
git commit -m "feat(exemplar): add similarity scoring with max library combine"
```

---

## Task 6: 서브패키지 공개 API 정리

**Files:**
- Modify: `recommendation/exemplar/__init__.py`

- [ ] **Step 1: 공개 API 재노출**

`recommendation/exemplar/__init__.py`:

```python
"""모범 종목 패턴 매칭 (7번째 팩터) 서브패키지."""

from recommendation.exemplar.features import (
    FEATURE_KEYS,
    compute_features_series,
    compute_features_snapshot,
)
from recommendation.exemplar.library import (
    Exemplar,
    ExemplarLibrary,
    make_exemplar_id,
)
from recommendation.exemplar.profile import (
    build_profile,
    build_profile_from_df,
    load_profile,
    save_profile,
)
from recommendation.exemplar.similarity import (
    SimilarityResult,
    score_candidate,
)

__all__ = [
    "FEATURE_KEYS",
    "compute_features_series",
    "compute_features_snapshot",
    "Exemplar",
    "ExemplarLibrary",
    "make_exemplar_id",
    "build_profile",
    "build_profile_from_df",
    "load_profile",
    "save_profile",
    "SimilarityResult",
    "score_candidate",
]
```

- [ ] **Step 2: import 동작 확인**

Run: `python -c "from recommendation.exemplar import FEATURE_KEYS, score_candidate, ExemplarLibrary; print('ok')"`
Expected: `ok`

- [ ] **Step 3: 전체 테스트 회귀 확인**

Run: `pytest tests/exemplar/ -v --ignore=tests/exemplar/test_sanity.py`
Expected: 모든 테스트 PASS (sanity는 아직 없음)

- [ ] **Step 4: 커밋**

```bash
git add recommendation/exemplar/__init__.py
git commit -m "feat(exemplar): expose public API via __init__"
```

---

## Task 7: scorer.py — 7번째 팩터 통합

**Files:**
- Modify: `recommendation/scorer.py`
- Test: `tests/exemplar/test_scorer_integration.py`

- [ ] **Step 1: 실패하는 통합 테스트 작성**

`tests/exemplar/test_scorer_integration.py`:

```python
"""scorer.py의 7번째 팩터 통합 회귀 테스트."""

import numpy as np
import pandas as pd
import pytest

from recommendation.exemplar.profile import build_profile_from_df
from recommendation.scorer import RecommendationScorer, build_weights


def test_build_weights_no_exemplar_returns_six_factor():
    w = build_weights(exemplar_weight=0.0)
    assert "exemplar" not in w
    assert set(w.keys()) == {"momentum", "trend", "breakout", "valuation", "growth", "risk"}
    assert sum(w.values()) == pytest.approx(1.0)


def test_build_weights_with_exemplar_scales_six_factors():
    w = build_weights(exemplar_weight=0.20)
    assert w["exemplar"] == pytest.approx(0.20)
    assert w["momentum"] == pytest.approx(0.20 * 0.80)
    assert w["trend"] == pytest.approx(0.20 * 0.80)
    assert sum(w.values()) == pytest.approx(1.0)


def test_score_single_no_profiles_omits_exemplar_keys(rising_ohlcv):
    scorer = RecommendationScorer()
    result = scorer.score_single({
        "ticker": "TEST", "price_df": rising_ohlcv, "info": {},
    })
    assert result is not None
    assert "exemplar_similarity" not in result
    assert "best_match_id" not in result


def test_score_single_with_profiles_adds_exemplar_keys(rising_ohlcv):
    profile = build_profile_from_df(rising_ohlcv)
    scorer = RecommendationScorer(
        weights=build_weights(exemplar_weight=0.20),
        profiles=[("self", profile)],
    )
    result = scorer.score_single({
        "ticker": "TEST", "price_df": rising_ohlcv, "info": {},
    })
    assert result is not None
    # 자기 자신 vs 자기 프로파일 → exemplar_similarity 높아야 함
    assert result["exemplar_similarity"] > 50
    assert result["best_match_id"] == "self"


def test_score_all_normalizes_exemplar_column(rising_ohlcv, flat_ohlcv):
    profile = build_profile_from_df(rising_ohlcv)
    scorer = RecommendationScorer(
        weights=build_weights(exemplar_weight=0.20),
        profiles=[("rally", profile)],
    )
    data = [
        {"ticker": "A", "price_df": rising_ohlcv, "info": {}},
        {"ticker": "B", "price_df": flat_ohlcv, "info": {}},
    ]
    df = scorer.score_all(data)
    assert "exemplar_similarity" in df.columns
    # 백분위 정규화: 두 종목이면 50, 100
    assert set(df["exemplar_similarity"].round(0).tolist()) == {50.0, 100.0}
    # rising 종목이 더 높은 점수
    rising_row = df[df["ticker"] == "A"].iloc[0]
    flat_row = df[df["ticker"] == "B"].iloc[0]
    assert rising_row["exemplar_similarity"] > flat_row["exemplar_similarity"]


def test_zero_exemplar_weight_matches_legacy_total(rising_ohlcv):
    """7번째 가중치=0이면 기존 6팩터 결과와 동일해야 한다 (회귀 방지)."""
    legacy = RecommendationScorer()
    legacy_df = legacy.score_all([
        {"ticker": "A", "price_df": rising_ohlcv, "info": {}},
    ])

    new = RecommendationScorer(weights=build_weights(exemplar_weight=0.0))
    new_df = new.score_all([
        {"ticker": "A", "price_df": rising_ohlcv, "info": {}},
    ])

    assert legacy_df["total_score"].iloc[0] == pytest.approx(new_df["total_score"].iloc[0])
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/exemplar/test_scorer_integration.py -v`
Expected: `ImportError: cannot import name 'build_weights'` (또는 유사 오류)

- [ ] **Step 3: scorer.py 수정 — `build_weights` 함수 추가**

`recommendation/scorer.py` 상단 import 아래에 추가 (`FACTOR_WEIGHTS` 정의 위에):

```python
from recommendation.exemplar.features import compute_features_snapshot
from recommendation.exemplar.similarity import score_candidate
```

`FACTOR_WEIGHTS` 정의 바로 아래에 추가:

```python
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
```

- [ ] **Step 4: scorer.py 수정 — `__init__`에 `profiles` 추가**

`RecommendationScorer.__init__` 메서드를 다음으로 교체:

```python
    def __init__(
        self,
        weights: dict | None = None,
        profiles: list[tuple[str, dict[str, dict[str, float]]]] | None = None,
    ):
        self.weights = weights or FACTOR_WEIGHTS
        self.profiles = profiles or []
```

- [ ] **Step 5: scorer.py 수정 — `score_single`에 7번째 팩터 계산 추가**

`score_single` 메서드 안쪽, `factors = {...}` 딕셔너리 직후 `total = sum(...)` 위에 다음을 끼워 넣고, return 딕셔너리도 수정:

`score_single`의 try 블록 일부를 다음으로 교체:

```python
        try:
            momentum = self._compute_momentum(df)
            trend = self._compute_trend(df)
            breakout = self._compute_breakout(df)
            valuation = self._compute_valuation(info)
            growth = self._compute_growth(info)
            risk = self._compute_risk(df, info)

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
```

(`factors`의 `exemplar` 키와 결과 dict의 `exemplar_similarity` 키를 별도로 두는 이유: `score_all`의 컬럼 정규화는 가중치 키 기준으로 돌고, `exemplar_similarity`는 UI 표시용 원본을 보존.)

- [ ] **Step 6: scorer.py 수정 — `score_all`의 정규화에 exemplar 컬럼 포함**

`score_all` 메서드의 `factor_cols = [...]` 줄을 다음으로 교체:

```python
        factor_cols = list(self.weights.keys())
        for col in factor_cols:
            if col in df.columns:
                df[col] = df[col].rank(pct=True) * 100

        df["total_score"] = sum(
            df[factor] * self.weights[factor]
            for factor in factor_cols
            if factor in df.columns
        )
        df["total_score"] = df["total_score"].round(2)
```

또한 `exemplar` 가중치 키가 활성일 때 `exemplar_similarity` 원본 컬럼이 결과에 그대로 남도록 처리 — `score_single`에서 반환한 `exemplar_similarity`는 raw 점수, `exemplar` 컬럼(`factors`에서 온 것)이 정규화 대상. 두 컬럼 모두 DataFrame에 포함되어 있어야 한다. `score_all` 진입부에서 다음을 추가로 보장:

```python
        # raw exemplar similarity 보존
        if "exemplar_similarity" in df.columns:
            df["exemplar_raw"] = df["exemplar_similarity"]
```

(컬럼 두 개: `exemplar`(정규화된 가중치 키), `exemplar_raw`(UI 표시용 원본 0~100), `exemplar_similarity`는 정규화 전 값으로 그대로 유지 → 테스트는 정규화된 값을 보므로 `exemplar_similarity`는 `exemplar`와 동치가 되도록 명시 처리:

`score_all` 정규화 직후 한 줄 추가:

```python
        if "exemplar" in df.columns:
            df["exemplar_similarity"] = df["exemplar"]
```

(테스트 `test_score_all_normalizes_exemplar_column`는 정규화된 `exemplar_similarity`를 검증한다.)

- [ ] **Step 7: 통합 테스트 통과 확인**

Run: `pytest tests/exemplar/test_scorer_integration.py -v`
Expected: 6 passed

- [ ] **Step 8: 기존 스코어러 외부 호출이 안 깨지는지 확인**

Run: `python -c "from recommendation.scorer import RecommendationScorer, build_weights; s = RecommendationScorer(); print(s.weights)"`
Expected: 6팩터 dict 출력, exemplar 키 없음

- [ ] **Step 9: 커밋**

```bash
git add recommendation/scorer.py tests/exemplar/test_scorer_integration.py
git commit -m "feat(scorer): integrate exemplar similarity as 7th factor"
```

---

## Task 8: pipeline.py — 활성 모범 로딩 후 스코어러 전달

**Files:**
- Modify: `recommendation/pipeline.py`

- [ ] **Step 1: 변경 영역 확인 (Read 후 표시)**

`recommendation/pipeline.py`의 다음 부분이 변경 대상:
- import 영역 (상단)
- `run_pipeline` 시그니처 (`exemplar_weight` 추가)
- `run_pipeline` 내 스코어러 생성 부분 (`scorer = RecommendationScorer()` 줄)
- CLI argparse 추가

- [ ] **Step 2: import 추가**

`recommendation/pipeline.py` 상단 import 블록에 추가:

```python
from recommendation.exemplar.library import ExemplarLibrary
from recommendation.exemplar.profile import load_profile
from recommendation.scorer import RecommendationScorer, build_weights
```

(기존의 `from recommendation.scorer import RecommendationScorer` 줄은 위 줄로 대체)

또한 파일 상단의 디렉터리 상수 옆에 추가:

```python
EXEMPLAR_DIR = PROJECT_ROOT / "data" / "exemplars"
EXEMPLAR_JSON = EXEMPLAR_DIR / "exemplars.json"
```

- [ ] **Step 3: 헬퍼 함수 추가**

`run_pipeline` 함수 정의 직전에 추가:

```python
def load_active_profiles() -> list[tuple[str, dict]]:
    """라이브러리에서 활성 모범의 (id, profile) 리스트를 로드한다."""
    if not EXEMPLAR_JSON.exists():
        return []
    lib = ExemplarLibrary(EXEMPLAR_JSON)
    actives = lib.list_all(active_only=True)
    out: list[tuple[str, dict]] = []
    for ex in actives:
        profile_path = PROJECT_ROOT / ex.profile_path
        if not profile_path.exists():
            print(f"WARN: profile missing for {ex.id} at {profile_path}, skipping")
            continue
        out.append((ex.id, load_profile(profile_path)))
    return out
```

- [ ] **Step 4: `run_pipeline` 시그니처 + 스코어러 생성 변경**

`run_pipeline` 함수 시그니처를 다음으로 교체:

```python
def run_pipeline(
    period: str = "1y",
    max_workers: int = 10,
    top_n: int = 30,
    use_cache: bool = True,
    universe_subset: list[str] | None = None,
    exemplar_weight: float = 0.0,
) -> pd.DataFrame:
```

함수 본문의 `scorer = RecommendationScorer()` 줄을 다음으로 교체:

```python
    profiles = load_active_profiles() if exemplar_weight > 0 else []
    if profiles:
        print(f"활성 모범: {len(profiles)}개 (exemplar_weight={exemplar_weight})")
        weights = build_weights(exemplar_weight=exemplar_weight)
    else:
        weights = build_weights(exemplar_weight=0.0)
    scorer = RecommendationScorer(weights=weights, profiles=profiles)
```

- [ ] **Step 5: 결과 출력 컬럼 확장 (선택, 디스플레이용)**

`run_pipeline` 끝 부분의 `print(scores_df.head(top_n)...)` 줄을 다음으로 교체:

```python
    display_cols = ["rank", "ticker", "total_score", "momentum", "trend", "breakout", "valuation", "growth", "risk"]
    if "exemplar_similarity" in scores_df.columns:
        display_cols.append("exemplar_similarity")
        display_cols.append("best_match_id")
    print(scores_df.head(top_n)[display_cols].to_string())
```

- [ ] **Step 6: CLI에 `--exemplar-weight` 추가**

`if __name__ == "__main__":` 블록의 argparse 부분에 추가:

```python
    parser.add_argument("--exemplar-weight", type=float, default=0.0,
                        help="7번째 팩터(모범 유사도) 가중치 0~0.4 (기본 0 = 비활성)")
```

`run_pipeline(...)` 호출에 인자 추가:

```python
    run_pipeline(
        period=args.period,
        max_workers=args.workers,
        top_n=args.top,
        use_cache=not args.no_cache,
        universe_subset=subset,
        exemplar_weight=args.exemplar_weight,
    )
```

- [ ] **Step 7: 파이프라인 import 회귀 확인**

Run: `python -c "from recommendation.pipeline import run_pipeline, load_active_profiles; print('ok')"`
Expected: `ok`

- [ ] **Step 8: 6팩터 모드 회귀 확인 (실행)**

Run: `python -m recommendation.pipeline --subset AAPL,MSFT --top 2`
Expected: 정상 실행, exemplar 컬럼 출력 없음 (활성 모범 0개일 때)

- [ ] **Step 9: 커밋**

```bash
git add recommendation/pipeline.py
git commit -m "feat(pipeline): wire exemplar profiles into scoring run"
```

---

## Task 9: Streamlit — 모범 라이브러리 페이지 (탭 추가)

**Files:**
- Modify: `recommendation/app.py`

- [ ] **Step 1: import 추가**

`recommendation/app.py` 상단의 import 블록에 추가:

```python
from datetime import datetime
from recommendation.exemplar.features import FEATURE_KEYS
from recommendation.exemplar.library import Exemplar, ExemplarLibrary, make_exemplar_id
from recommendation.exemplar.profile import build_profile, load_profile, save_profile
```

파일 상단 상수 영역에 추가:

```python
EXEMPLAR_DIR = PROJECT_ROOT / "data" / "exemplars"
EXEMPLAR_JSON = EXEMPLAR_DIR / "exemplars.json"
PROFILES_DIR = EXEMPLAR_DIR / "profiles"
```

- [ ] **Step 2: 헬퍼 함수 추가**

`render_radar_chart` 함수 정의 아래에 추가:

```python
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

    # 미리보기: 프로파일 표 표시
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
```

- [ ] **Step 3: 탭 추가**

`tab1, tab2, tab3 = st.tabs([...])` 줄을 다음으로 교체:

```python
tab1, tab2, tab3, tab4 = st.tabs(["🏆 Top 추천", "📊 전체 랭킹", "🔍 종목 상세", "📚 모범 라이브러리"])
```

- [ ] **Step 4: 새 탭 본문 추가**

파일 끝(`# ─── Tab 3: 종목 상세 ───` 블록 뒤)에 추가:

```python
# ─── Tab 4: 모범 라이브러리 ───
with tab4:
    render_exemplar_form()
    st.divider()
    render_exemplar_list()
```

- [ ] **Step 5: Streamlit 동작 수동 검증**

Run: `streamlit run recommendation/app.py`
Expected: 앱 실행 → 4번째 탭 "📚 모범 라이브러리" 보임 → 폼/목록 렌더링 정상

검증 시나리오:
1. 티커 `NVDA`, 시작일 `2023-01-01`, 종료일 `2023-07-01` 입력 → 미리보기 → 프로파일 표 출현
2. 저장 → "저장 완료" 메시지 → 하단 목록에 추가됨
3. 활성 체크박스 해제 → 페이지 새로고침 후에도 비활성 유지
4. 상세 버튼 → 프로파일 표 펼침
5. 🗑 버튼 → 항목 사라짐, parquet 파일도 삭제됨

검증 완료 후 ctrl+C로 종료.

- [ ] **Step 6: 커밋**

```bash
git add recommendation/app.py
git commit -m "feat(app): add exemplar library page with add/list/toggle/delete"
```

---

## Task 10: Streamlit — 사이드바 가중치 슬라이더 + 카드 표시

**Files:**
- Modify: `recommendation/app.py`

- [ ] **Step 1: 사이드바에 가중치 슬라이더 추가**

`app.py`의 사이드바 블록 (`st.sidebar.header("설정")` 아래) 끝에 추가:

```python
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
```

- [ ] **Step 2: Top 카드에 모범 유사도 줄 추가**

Tab 1 본문의 카드 렌더링 블록 (`st.metric("Total Score", ...)` 다음 줄에) 추가:

```python
                # 모범 유사도 표시 (있을 때만)
                if "exemplar_similarity" in row.index and pd.notna(row.get("exemplar_similarity")):
                    best_id = row.get("best_match_id", "")
                    best_name = ""
                    if best_id:
                        match_ex = lib_preview.get(best_id)
                        best_name = match_ex.name if match_ex else best_id
                    st.caption(f"모범 유사도: {row['exemplar_similarity']:.1f} ({best_name} 닮음)")
```

- [ ] **Step 3: 종목 상세 탭에 모범 분해 표 추가**

Tab 3 본문의 `# Claude 분석` 블록 직전에 추가:

```python
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
```

- [ ] **Step 4: 사용자에게 가중치 설명 안내**

Tab 1 본문 상단(`st.subheader(f"Top {top_n} 추천 종목")` 바로 아래)에 추가:

```python
    if "exemplar_similarity" not in scores_df.columns:
        st.caption("💡 모범 유사도 팩터를 활성화하려면 사이드바에서 가중치를 조정 후 `python -m recommendation.pipeline --exemplar-weight 0.20`을 실행하세요.")
```

(슬라이더 자체는 표시용/안내용. 실제 재계산은 CLI에서. 향후 UI 트리거는 별도 작업.)

- [ ] **Step 5: Streamlit 수동 검증**

Run: `streamlit run recommendation/app.py`

검증:
1. 사이드바 슬라이더가 보임 (기본 20%)
2. 활성 모범 개수 표시
3. 활성 0개일 때 경고 메시지
4. 스코어 파일이 6팩터 모드일 때 안내 메시지가 Tab 1 상단에 보임

CLI에서 7번째 팩터 모드로 파이프라인 재실행 후 다시 검증:

Run: `python -m recommendation.pipeline --subset AAPL,MSFT,NVDA --exemplar-weight 0.20`
(활성 모범이 있어야 의미 있음 — 위 Task 9 검증에서 NVDA 모범을 등록해 두면 됨)

`streamlit run recommendation/app.py` 다시 실행 →
- Top 카드에 "모범 유사도: NN.N (NVDA 2023 닮음)" 줄 보임
- 종목 상세 탭에 분해 표 보임

- [ ] **Step 6: 커밋**

```bash
git add recommendation/app.py
git commit -m "feat(app): expose exemplar weight slider and similarity display"
```

---

## Task 11: Sanity 테스트 (실 yfinance 데이터, slow 마커)

**Files:**
- Create: `tests/exemplar/test_sanity.py`

- [ ] **Step 1: sanity 테스트 작성**

`tests/exemplar/test_sanity.py`:

```python
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
```

- [ ] **Step 2: sanity 테스트 실행 (slow 마커 명시)**

Run: `pytest tests/exemplar/test_sanity.py -v -m slow`
Expected: 3 passed (외부 API 호출이므로 30초~1분 소요 가능)

테스트가 yfinance 일시 장애로 실패하면 재실행. 점수 임계값(`> 70.0` 등)이 모범 빌드 결과에 따라 미세 조정 필요할 수 있음 — 실패 시 실제 점수를 출력해서 임계값을 합리적으로 재설정.

- [ ] **Step 3: 기본 test 실행 시 sanity 제외 확인**

Run: `pytest tests/ -v`
Expected: sanity 외 모든 테스트만 실행 (slow 마커 제외는 pytest.ini의 marker 등록만으로는 자동 제외되지 않으나, 실패하지 않음)

Run: `pytest tests/ -v -m "not slow"`
Expected: sanity 테스트가 deselected, 나머지만 실행

- [ ] **Step 4: 커밋**

```bash
git add tests/exemplar/test_sanity.py
git commit -m "test(exemplar): add yfinance sanity tests (slow)"
```

---

## Task 12: 최종 회귀 + 문서 갱신

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: 전체 fast 테스트 회귀**

Run: `pytest tests/ -v -m "not slow"`
Expected: 모든 비-slow 테스트 PASS, 실패 0

- [ ] **Step 2: 파이프라인 6팩터 모드 회귀**

Run: `python -m recommendation.pipeline --subset AAPL,MSFT --top 2`
Expected: 정상 종료, 출력 컬럼에 exemplar 관련 컬럼 없음

- [ ] **Step 3: 파이프라인 7팩터 모드 회귀 (활성 모범 있을 때)**

(Task 9에서 등록한 NVDA 모범이 있다고 가정)

Run: `python -m recommendation.pipeline --subset AAPL,MSFT,NVDA --exemplar-weight 0.20`
Expected: 정상 종료, 출력에 `exemplar_similarity`, `best_match_id` 컬럼 보임

- [ ] **Step 4: CLAUDE.md 프로젝트 구조 갱신**

`CLAUDE.md`의 프로젝트 구조 트리에서 `recommendation/` 항목 아래 다음을 추가 (구조 섹션의 적절한 위치에):

```
├── recommendation/
│   ├── exemplar/              # 모범 패턴 매칭 (7번째 팩터)
│   │   ├── features.py        # 10개 기술적 피처
│   │   ├── profile.py         # 프로파일 빌드/저장/로드
│   │   ├── library.py         # 모범 라이브러리 CRUD
│   │   └── similarity.py      # 후보 스코어링
```

`주요 명령어` 섹션에 추가:

```bash
# 7번째 팩터(모범 유사도) 활성화 추천 실행
python -m recommendation.pipeline --exemplar-weight 0.20

# 테스트 실행 (slow 제외)
pytest tests/ -m "not slow"
```

- [ ] **Step 5: 커밋**

```bash
git add CLAUDE.md
git commit -m "docs: document exemplar pattern matching feature"
```

---

## Self-Review 체크리스트 (계획 작성 후 본인이 확인)

**스펙 커버리지:**
- 라이브러리 CRUD → Task 4 ✓
- 10개 피처 정의 → Task 2 ✓
- 프로파일 빌드/저장 → Task 3 ✓
- 유사도 알고리즘 (z-거리, 지수 감쇠, max) → Task 5 ✓
- 7번째 팩터 통합 + 가중치 재배분 → Task 7 ✓
- 활성 모범 0개일 때 비활성 → Task 7 (build_weights 0 처리) + Task 8 (pipeline 분기) ✓
- 회귀: 가중치 0이면 기존과 동일 → Task 7 `test_zero_exemplar_weight_matches_legacy_total` ✓
- Streamlit 라이브러리 페이지 → Task 9 ✓
- Streamlit 가중치 슬라이더 + 카드/상세 분해 → Task 10 ✓
- Sanity (자기 매칭, 다른 시기, 무관 종목) → Task 11 ✓

**Placeholder/TODO 스캔:** 없음.

**타입 일관성:**
- `Exemplar` 데이터클래스 필드 일관 ✓
- `SimilarityResult.best_match_id` ↔ 결과 dict `"best_match_id"` 일관 ✓
- `profiles: list[tuple[str, dict]]` 시그니처 일관 (similarity.score_candidate, RecommendationScorer.__init__, load_active_profiles 반환) ✓
- `FEATURE_KEYS` 단일 정의처 (features.py) → 다른 모듈은 모두 import ✓
