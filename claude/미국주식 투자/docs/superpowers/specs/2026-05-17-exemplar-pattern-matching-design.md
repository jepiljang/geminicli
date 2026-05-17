# 모범 종목 패턴 매칭 (7번째 팩터) 설계

## Context

기존 추천 시스템(`docs/superpowers/specs/2026-05-17-stock-recommendation-design.md`)은 정량 6팩터 스코어링을 사용한다. 사용자는 여기에 더해 "본인이 정한 모범 종목의 상승 구간 패턴"과 닮은 종목을 가점하는 기능을 원한다.

핵심 아이디어: 사용자가 티커 + 상승 시작일/종료일을 입력하면, 그 구간의 기술적 지표 평균/표준편차를 "모범 프로파일"로 저장. 추천 시 각 후보 종목의 현재 시점 지표가 이 프로파일에 얼마나 가까운지 평가 → 추천 점수에 반영.

## 요구사항

- 사용자가 모범 종목을 **여러 개 라이브러리로 등록**하고 개별 활성/비활성 토글 가능
- 추천 시 활성화된 모범들 중 **최대 유사도**를 7번째 팩터 점수로 사용
- 모범 라이브러리가 비었거나 전부 비활성이면 **기존 6팩터 동작 유지** (회귀 없음)
- 기술적 지표 기반 (펀더멘털 제외) — 상승 구간의 "움직임 패턴"을 잡는 데 집중
- Streamlit 대시보드에서 모범 추가/삭제/토글 및 가중치 조정

## 아키텍처

### 점수 계산 흐름

```
사용자: 티커 + 시작일 + 종료일 입력
    ↓
[A] 구간 OHLCV 수집 → 10개 기술적 지표 시계열 계산
    ↓
[B] 각 지표의 (평균, 표준편차) 추출 → 모범 프로파일 저장
    ↓ (저장 후 반복 사용)
─────────────────────────────────────────────
추천 파이프라인 실행 시:
    ↓
[C] 활성 모범 프로파일들 일괄 로드
    ↓
[D] 각 후보 종목의 오늘 시점 10개 지표 스냅샷 계산
    ↓
[E] 활성 모범 각각에 대해 z-거리 → 지수 감쇠 → 0~100 유사도
    ↓
[F] 모든 활성 모범 중 최대값 = 7번째 팩터 점수
    ↓
[G] 기존 6팩터(× 0.8) + 7번째(× 0.2) 가중합 → 최종 점수
```

### 피처 (10개, 모두 일봉 기반)

| # | 피처 키 | 계산 |
|---|---------|------|
| 1 | `rsi_14` | RSI(14) |
| 2 | `ret_5d` | `pct_change(5)` |
| 3 | `ret_20d` | `pct_change(20)` |
| 4 | `macd_hist_norm` | `(MACD - signal) / Close` |
| 5 | `adx_14` | ADX(14) |
| 6 | `sma20_pos` | `Close / SMA20 - 1` |
| 7 | `sma50_pos` | `Close / SMA50 - 1` |
| 8 | `high_52w_pos` | `Close / rolling_max(252)` |
| 9 | `bb_pos` | `(Close - BB_mid) / (BB_upper - BB_mid)` |
| 10 | `volume_ratio` | `Volume / SMA(Volume, 20)` |

### 유사도 점수 계산

후보 스냅샷 `x` vs 모범 프로파일 `{μ_i, σ_i}` (i = 10 피처):

```
ε = 1e-6
z_i = |x_i - μ_i| / max(σ_i, ε)
d = mean(z_i for i in 10 features)
similarity = 100 * exp(-d / 2)
```

지수 감쇠 특성: `d=0 → 100`, `d=2 → ~37`, `d=4 → ~14`. 작은 차이는 큰 점수 변화 없고 멀어질수록 빠르게 0에 수렴.

라이브러리 통합:

```
exemplar_score = max(similarity(candidate, ex) for ex in active_exemplars)
```

활성 모범이 0개면 7번째 팩터를 비활성화 (가중치 0, 기존 6팩터가 100% 차지).

### 6팩터 시스템과의 통합

**가중치 (7번째 활성 시 기본값):**

| 팩터 | 기존 | 신규 |
|------|------|------|
| 모멘텀 | 20% | 16% |
| 트렌드 강도 | 20% | 16% |
| 브레이크아웃 | 15% | 12% |
| 밸류에이션 | 15% | 12% |
| 수익성/성장 | 15% | 12% |
| 리스크 | 15% | 12% |
| 모범 유사도 (신규) | — | 20% |

기존 6팩터에 일괄 `× 0.8` 적용 후 신규 20% 추가. 비례 유지.

**가중치 조정:** Streamlit 사이드바에서 7번째 팩터 가중치를 0~40% 슬라이더로 변경 가능. 다른 6팩터는 (1 - new_weight) 범위에서 원래 비율대로 자동 재정규화.

**비활성 케이스 (활성 모범 0개):** 7번째 팩터 가중치를 0으로 강제 → 기존 6팩터가 100%.

### 후보 종목 점수 출력 구조

```python
{
    "AMD": {
        "total_score": 78.5,
        "factor_scores": {...},                  # 기존 6팩터
        "exemplar_similarity": 82.3,             # 7번째 팩터 점수
        "best_match_id": "nvda-2023-rally",      # SimilarityResult.best_match_id
        "exemplar_breakdown": {                  # 디버깅/UI용
            "nvda-2023-rally": 82.3,
            "tsla-2020-rally": 64.1
        }
    }
}
```

## 데이터 구조

### 라이브러리 메타 (`data/exemplars/exemplars.json`)

```json
{
  "exemplars": [
    {
      "id": "nvda-2023-rally",
      "ticker": "NVDA",
      "name": "NVDA 2023 AI 랠리",
      "start_date": "2023-01-01",
      "end_date": "2023-07-01",
      "active": true,
      "created_at": "2026-05-17T14:30:00",
      "profile_path": "data/exemplars/profiles/nvda-2023-rally.parquet"
    }
  ]
}
```

`id`는 `{ticker_lower}-{start_year}-{slug}` 형태로 자동 생성. 동일 id 충돌 시 suffix `-2`, `-3` 등 부여.

### 프로파일 파일 (`data/exemplars/profiles/{id}.parquet`)

10행 × 3열 데이터프레임:

| feature | mean | std |
|---------|------|-----|
| rsi_14 | 64.2 | 8.1 |
| ret_5d | 0.024 | 0.018 |
| ... | ... | ... |

## 프로젝트 구조

```
recommendation/
├── exemplar/                       # 신규
│   ├── __init__.py
│   ├── features.py                 # 10개 기술적 피처 (시계열 + 스냅샷)
│   ├── profile.py                  # 티커+구간 → 프로파일 빌드/저장/로드
│   ├── library.py                  # exemplars.json CRUD
│   └── similarity.py               # 후보 점수 + best_match 계산
├── scorer.py                       # 기존 — 7번째 팩터 호출 + 가중합 추가
└── app.py                          # 기존 — 모범 라이브러리 페이지 + 사이드바 추가

data/exemplars/                     # 신규
├── exemplars.json
└── profiles/
    └── {id}.parquet

tests/exemplar/                     # 신규
├── test_features.py
├── test_profile.py
├── test_library.py
├── test_similarity.py
└── test_sanity.py                  # 실제 데이터 sanity (slow)
```

### 모듈 책임 분리

| 모듈 | 책임 | 의존 |
|------|------|------|
| `features.py` | OHLCV → 10 피처 DataFrame 또는 단일 행 | `features/technical.py` 재사용 |
| `profile.py` | 티커+구간 → 프로파일 dict, 저장/로드 | `features.py`, `data/fetcher.py` |
| `library.py` | `exemplars.json` CRUD, active 모범 조회 | `profile.py` |
| `similarity.py` | 후보 단일 행 + 활성 프로파일들 → 점수 + best_match | `library.py`, `features.py` |
| `scorer.py` (기존) | 6팩터 + `similarity.py` 호출 통합 | `similarity.py` 호출 추가 |
| `app.py` (기존) | 새 페이지 + 사이드바 슬라이더 + 카드 표시 | `library.py`, `similarity.py` |

### 핵심 인터페이스 시그니처

```python
# exemplar/features.py
def compute_features_series(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV → 10피처 시계열 (구간 집계용). NaN 행은 결과에서 제외."""

def compute_features_snapshot(df: pd.DataFrame) -> dict[str, float]:
    """OHLCV (최근 252+ 거래일) → 마지막 행의 10피처 dict."""

# exemplar/profile.py
def build_profile(ticker: str, start: date, end: date) -> dict[str, dict]:
    """구간 데이터 수집 → 피처 계산 → {피처: {mean, std}}."""

def save_profile(exemplar_id: str, profile: dict) -> Path: ...
def load_profile(exemplar_id: str) -> dict: ...

# exemplar/library.py
def list_exemplars(active_only: bool = False) -> list[Exemplar]: ...
def add_exemplar(ticker: str, start: date, end: date, name: str | None) -> Exemplar: ...
def delete_exemplar(exemplar_id: str) -> None: ...
def toggle_active(exemplar_id: str, active: bool) -> None: ...

# exemplar/similarity.py
@dataclass
class SimilarityResult:
    score: float | None              # None if no active exemplars
    best_match_id: str | None
    breakdown: dict[str, float]      # {exemplar_id: similarity}

def score_candidate(snapshot: dict[str, float],
                    profiles: list[tuple[str, dict]]) -> SimilarityResult: ...
```

`scorer.py`는 후보 처리 루프 시작 전 활성 프로파일 일괄 로드, 루프 안에서 `score_candidate` 호출, 결과를 7번째 팩터로 합산.

## Streamlit UI

### 신규 페이지: "📚 모범 라이브러리"

**모범 추가 폼:**
- 입력: 티커, 이름(선택), 시작일, 종료일
- 미리보기 버튼: 해당 구간 가격 차트 + 피처 10개 `(mean, std)` 표 + 구간 누적 수익률
- 저장 검증:
  - 티커 데이터 존재 여부 (yfinance 가져오기 성공)
  - `end_date > start_date`
  - 구간 길이 ≥ 30 거래일 (프로파일 신뢰도)
  - 구간 누적 수익률 > 0 (경고만, 저장은 허용)

**등록된 모범 목록:**
- 테이블: ID, 이름, 구간, 활성 체크박스, 삭제 버튼
- 행 클릭 시 가격 차트 + 프로파일 표 펼쳐 보기

### 메인 추천 페이지 수정

**사이드바 추가:**
- 7번째 팩터 가중치 슬라이더 (0~40%, 기본 20%)
- 활성 모범 개수 표시 (`2 / 3 활성`)

**Top N 카드 추가 줄:**
- `모범 유사도: 82.3 (NVDA 2023 닮음)` — best_match 이름을 함께 표시

**종목 상세 페이지 추가:**
- 기존 6팩터 레이더차트 옆에 "모범 유사도 분해" 표
  - 컬럼: 피처, 모범(μ±σ), 후보 오늘, z-거리(부호 포함)

## 검증

### 단위 테스트 (`tests/exemplar/`)

| 파일 | 검증 |
|------|------|
| `test_features.py` | 알려진 OHLCV → 10 피처가 예상값과 일치 (스냅샷/시계열 양쪽) |
| `test_profile.py` | NVDA 2023-01~07 → `build_profile` 호출 → mean/std 합리적 범위 |
| `test_library.py` | add → list → toggle → delete CRUD 라운드트립, 잘못된 ID 처리 |
| `test_similarity.py` | (a) 모범과 동일 스냅샷 → 100점, (b) 매우 다른 스냅샷 → 0 근접, (c) 빈 라이브러리 → `score=None` |

### Sanity 테스트 (`test_sanity.py`, slow)

실제 데이터를 fetch하여 자동 검증:

1. **자기 매칭**: NVDA 2023 모범 + NVDA 2023-04-01 스냅샷 → 90+ 점
2. **다른 시기**: NVDA 2023 모범 + NVDA 2018-12 스냅샷 → 낮은 점수
3. **무관 종목**: NVDA 2023 모범 + KO 스냅샷 → 낮은 점수

### Streamlit 수동 검증

- 라이브러리 페이지에서 NVDA 2023 추가 → 미리보기 정상 → 저장 → 목록 출현
- 사이드바 슬라이더 조정 시 Top N 순위가 변하는지 확인
- 7번째 팩터 가중치=0일 때 기존 6팩터 결과와 동일
- 활성 모범 0개일 때 자동 비활성화

### 회귀 방지

기존 6팩터 스코어러 로직은 손대지 않음. `scorer.py`에는 호출 + 합산만 추가. 7번째 팩터 가중치=0이면 기존 결과와 동일해야 함 (회귀 테스트 1건).

### KPI 연동 (본 spec 범위 외)

모범 유사도 Top 10 종목의 향후 4주 실제 수익률을 SPY와 비교 → 모범 매칭이 alpha를 만드는지 검증. 데이터 축적 후 별도 작업.
