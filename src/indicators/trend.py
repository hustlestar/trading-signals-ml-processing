import pandas_ta as pta


def sma(close, length: int = 10):
    return pta.sma(close, length=length)


def ema(close, length: int = 10):
    return pta.ema(close, length=length)


def dema(close, length: int = 10):
    return pta.dema(close, length=length)