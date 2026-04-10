import pandas as pd
import ta


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """이동평균: SMA(20, 50, 200), EMA(12, 26)"""
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
    return df


def add_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """RSI(14)"""
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD (12, 26, 9) + Signal + Histogram"""
    macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    return df


def add_bollinger(df: pd.DataFrame) -> pd.DataFrame:
    """볼린저밴드 (20, 2) - Upper, Middle, Lower"""
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Middle'] = bollinger.bollinger_mavg()
    df['BB_Lower'] = bollinger.bollinger_lband()
    return df


def add_atr(df: pd.DataFrame) -> pd.DataFrame:
    """ATR(14)"""
    df['ATR_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    return df


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """거래량 이동평균: Volume SMA(20)"""
    df['Volume_SMA_20'] = ta.trend.sma_indicator(df['Volume'], window=20)
    return df


def add_stochastic(df: pd.DataFrame) -> pd.DataFrame:
    """Stochastic Oscillator (14, 3, 3)"""
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """모든 기술적 지표를 한번에 추가"""
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_atr(df)
    df = add_volume_indicators(df)
    df = add_stochastic(df)
    return df
