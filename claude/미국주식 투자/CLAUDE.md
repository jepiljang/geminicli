# 미국주식 투자 자동화 시스템

## 프로젝트 목표

**핵심 목표**: SPY(S&P 500) 대비 높은 수익률 + 승률(Win Rate) > 55% 달성

### 성과 지표 (KPI)
| 지표 | 목표 |
|------|------|
| Alpha vs SPY | > 0 (초과 수익) |
| Win Rate | > 55% |
| Sharpe Ratio | > 1.0 |
| Max Drawdown | < -20% |
| 연간 수익률 | > SPY 연평균 |

---

## 기술 스택

- **데이터 수집**: KIS OpenAPI (한국투자증권), yfinance (보조/무료)
- **분석**: pandas, numpy, `ta` 라이브러리 (기술적 지표)
- **백테스팅**: Streamlit + vectorbt
- **알림**: Telegram Bot API
- **AI 보조**: Gemini MCP (분석/리서치 보조)
- **스킬 확장**: Claude Code skills (`/strategy/models/` 기반)

---

## 프로젝트 구조

```
미국주식 투자/
├── CLAUDE.md                  # 이 파일 — 프로젝트 가이드
├── requirements.txt           # 의존성 목록
├── data/
│   ├── fetcher.py             # KIS API + yfinance 데이터 수집
│   └── raw/                   # 원본 데이터 저장소 (gitignored)
├── features/
│   ├── technical.py           # 기술적 지표: MA, RSI, MACD, 볼린저밴드 등
│   ├── fundamental.py         # 펀더멘털 지표: PER, PBR, EPS 등
│   └── custom.py              # 커스텀 피처 — 새로운 아이디어 실험 공간
├── strategy/
│   ├── base.py                # 전략 베이스 클래스 (공통 인터페이스)
│   └── models/                # 개별 전략 파일들
│       └── .gitkeep
├── backtest/
│   ├── engine.py              # 백테스팅 엔진 (신호 → 수익률 계산)
│   ├── metrics.py             # 성과 지표: Sharpe, Alpha, Win Rate, MDD
│   └── app.py                 # Streamlit 대시보드 (SPY 비교 차트)
├── notification/
│   └── telegram.py            # Telegram 알림 모듈
└── .env                       # API 키 (gitignored)
```

---

## 개발 원칙

1. **백테스팅 먼저**: 새 전략/피처는 반드시 `backtest/app.py`에서 SPY 대비 성과 검증 후 채택
2. **SPY 벤치마크 항상 비교**: 모든 성과 측정은 SPY와 나란히 표시
3. **피처 문서화**: `features/custom.py`에 새 피처 추가 시 로직과 근거 주석 필수
4. **전략 격리**: 각 전략은 `strategy/models/` 아래 별도 파일로 관리
5. **실계좌 주의**: `.env`의 `KIS_PAPER=false`는 실거래 모드 — 테스트 시 `true`로 변경

---

## 🔒 보안 규칙 (절대 준수)

### 1. 민감정보 취급
다음은 **절대로** 외부 전송/커밋/로그 출력 금지:
- `.env` 파일의 모든 값 (KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT_NO, BOT_TOKEN, CHAT_ID, GEMINI_API_KEY)
- `.mcp.json` 내용
- `token_cache.json`, `ledger.json`
- 실제 계좌번호, 주문번호, 체결 내역

### 2. AI 호출 시 (Gemini MCP 등)
- **프롬프트에 `.env` 값, 실제 API 응답, 계좌정보 포함 금지**
- 코드 생성 요청 시에는 **요구사항만** 설명. 실제 키/토큰 예시도 쓰지 말 것
- 에러 로그 공유 시 민감정보 마스킹 필수

### 3. 커밋 규칙
- `git add .` / `git add -A` **금지** — 반드시 파일 경로 명시
- 커밋 전 `git status`로 stage된 파일 확인
- `.gitignore`에 등재된 파일이 stage되었는지 매번 확인
- 토큰/키 패턴 (`sk-`, `ghp_`, `eyJ`, `BOT_TOKEN=값`) grep 검증 권장

### 4. 로그/에러 출력
- `print()`, 로거로 `.env` 값을 출력하지 말 것
- 예외 메시지에 토큰이 포함되지 않도록 try/except에서 필터링
- Telegram/Slack 알림 메시지에도 민감정보 포함 금지

### 5. 실거래 모드 안전장치
- `KIS_PAPER=false` 전환은 **유저가 수동으로만** — 코드/스크립트가 자동 변경 금지
- 실거래 모드에서는 주문 실행 전 반드시 컨펌 단계 필요
- 주문 금액 상한선 하드코딩 (예: 1회 $5,000) 권장

### 6. 의존성 관리
- 제3자 패키지 추가 시 출처/신뢰도 확인
- skill 설치 시 Socket/Snyk 보안 리포트 검토
- 실행 가능한 스크립트가 포함된 skill은 내용 리뷰 후 실행

---

## 주요 명령어

```bash
# 백테스팅 대시보드 실행
streamlit run backtest/app.py

# 특정 종목 데이터 수집
python data/fetcher.py --ticker AAPL --period 2y

# Telegram 알림 테스트
python notification/telegram.py --test

# 의존성 설치
pip install -r requirements.txt
```

---

## 환경 변수 (.env)

```
# 한국투자증권 KIS API
KIS_APP_KEY=...
KIS_APP_SECRET=...
KIS_ACCOUNT_NO=...
KIS_ACCOUNT_PROD_CD=01
KIS_PAPER=true   # 테스트 시 true, 실거래 시 false

# Telegram
BOT_TOKEN=...
CHAT_ID=...

# Google Gemini
GEMINI_API_KEY=...
```

---

## 분석 워크플로우

```
데이터 수집 (data/fetcher.py)
    ↓
피처 생성 (features/technical.py, custom.py)
    ↓
전략 시그널 생성 (strategy/models/*.py)
    ↓
백테스팅 검증 (backtest/engine.py)
    ↓
성과 측정 vs SPY (backtest/metrics.py)
    ↓
Streamlit 시각화 (backtest/app.py)
    ↓
실시간 알림 (notification/telegram.py)
```

---

## Skills 활용 가이드

새로운 분석 스킬이 필요하면:
```bash
# 스킬 검색
npx skills search stock-analysis

# 스킬 설치
npx skills add <repo-url> --skill <skill-name>
```

설치된 스킬 기반 전략은 `strategy/models/` 아래에 래핑하여 관리.
