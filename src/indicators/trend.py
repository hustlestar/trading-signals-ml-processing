import pandas as pd
import pandas_ta as pta

def sma(close, length: int = 10):
    return pta.sma(close, length=length)


def ema(close, length: int = 10):
    return pta.ema(close, length=length)


def dema(close, length: int = 10):
    return pta.dema(close, length=length)


def rema(close, length: int = 10):
    """Relative EMA which is in the equal range of 0 to 1"""
    return pta.ema(close, length=length) / close


def vwap(row, high_cols, low_cols, close_cols, volume_cols):
    high = row[high_cols]
    low = row[low_cols]
    close = row[close_cols]
    volume = row[volume_cols]
    return pta.vwap(high, low, close, volume)


def wcp(row, high_cols, low_cols, close_cols):
    high = row[high_cols]
    low = row[low_cols]
    close = row[close_cols]
    return pta.wcp(high, low, close)