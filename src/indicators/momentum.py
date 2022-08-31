import pandas_ta as pta
from pandas_ta import verify_series


def rsi(close, length: int = None):
    return pta.rsi(close=close, length=length)


def mom(close, length: int = None):
    return pta.mom(close=close, length=length)


def apo(close, fast=None, slow=None):
    return pta.apo(close=close, fast=fast, slow=slow)


def rapo(close):
    """
    Relative Absolute Price Oscillator (APO), which creates relative APO for better comparison of distinct signals
    that might have magnitude different scale
    """
    tmp = pta.apo(close=close)
    tmp_max = tmp.max()
    return tmp.divide(tmp_max)


def willr(row, high_cols, low_cols, close_cols, length=None):
    high = row[high_cols]
    low = row[low_cols]
    close = row[close_cols]
    return _my_willr(high, low, close, length)


def _my_willr(high, low, close, length=None, talib=None, offset=None, **kwargs):
    """Indicator: William's Percent R (WILLR)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 14
    min_periods = int(kwargs["min_periods"]) if "min_periods" in kwargs and kwargs["min_periods"] is not None else length
    _length = max(length, min_periods)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)

    if high is None or low is None or close is None: return

    # Calculate Result
    new_axis = [i for i in range(close.size)]
    close = close.set_axis(new_axis)
    lowest_low = low.rolling(length, min_periods=min_periods).min()
    lowest_low = lowest_low.set_axis(new_axis)
    highest_high = high.rolling(length, min_periods=min_periods).max()
    highest_high = highest_high.set_axis(new_axis)
    willr = 100 * ((close - lowest_low) / (highest_high - lowest_low) - 1)
    return willr
