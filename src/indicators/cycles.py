import pandas_ta as pta


def ebsw(close, length: int):
    return pta.ebsw(close, length=length)
