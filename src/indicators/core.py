import logging
from functools import partial
from typing import Callable

import pandas as pd

import indicators.momentum as im
import indicators.trend as it
import pandas_ta as pta


def add_all_indicators_to_df(df):
    current_hourly_bars_closes = get_current_hourly_bars_cols(df, "close")
    current_hourly_bars_highs = get_current_hourly_bars_cols(df, "high")
    current_hourly_bars_lows = get_current_hourly_bars_cols(df, "low")
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(it.ema, length=20),
                             prefix="EMA_20_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=10)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(it.ema, length=40),
                             prefix="EMA_40_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(it.dema, length=30),
                             prefix="DEMA_30_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(it.dema, length=15),
                             prefix="DEMA_15_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=10)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(im.rsi, length=30),
                             prefix="RSI_30_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(im.rsi, length=15),
                             prefix="RSI_15_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=10)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(im.rapo),
                             prefix="RAPO_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=10)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_highs + current_hourly_bars_lows + current_hourly_bars_closes],
                             func=partial(im.willr, length=30, high_cols=current_hourly_bars_highs, low_cols=current_hourly_bars_lows, close_cols=current_hourly_bars_closes),
                             prefix="WILLR_30_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_highs + current_hourly_bars_lows + current_hourly_bars_closes],
                             func=partial(im.willr, length=15, high_cols=current_hourly_bars_highs, low_cols=current_hourly_bars_lows, close_cols=current_hourly_bars_closes),
                             prefix="WILLR_15_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=10)
    return df


def add_indicator_to_df(base_df: pd.DataFrame, subset_df: pd.DataFrame, func: Callable, prefix: str, postfix: str = "", number_of_cols_to_add: int = 10):
    """
    Takes pandas DataFrame, function of indicator, prefix, postfix and number of cols to add and creates
    columns associated with this indicator. Works with COLUMN based time-series features,
    where each column is some part of time-series data
    :param base_df: dataframe to which add new columns
    :param subset_df: pandas DataFrame part to which apply function
    :param func: function which creates TA indicator columns
    :param prefix: prefix for the new columns
    :param postfix: postfix fot the new columns (optional)
    :param number_of_cols_to_add: number of the last columns to add (optional)
    :return: new pandas DataFrame with TA indicator columns
    """
    logging.info(f"Adding indicator {prefix} to df {base_df.shape}")
    new_df = subset_df.apply(func, axis=1).iloc[:, -1 * number_of_cols_to_add:]

    def form_col_name(index):
        return f"{prefix if index > 0 else prefix.replace('MINUS', 'LATEST')}" + (f"_{index}" if index > 0 else "") + (f"_{postfix}" if postfix else postfix)

    new_df.columns = [form_col_name(k) for k in reversed(range(number_of_cols_to_add))]
    return pd.concat([base_df, new_df], axis=1)


def get_current_hourly_bars_cols(df, postfix):
    current_hourly_bars_cols_list = [col for col in df if col.startswith('current_hour_bars') and col.endswith(postfix)]
    current_hourly_bars_cols_list = sorted(current_hourly_bars_cols_list, reverse=True)
    current_hourly_bars_cols_list.append(f"latest_hour_{postfix}")

    assert len(current_hourly_bars_cols_list) == 49
    assert current_hourly_bars_cols_list[-2] == f'current_hour_bars_01_{postfix}'
    return current_hourly_bars_cols_list


if __name__ == '__main__':
    df = pd.DataFrame({})
    current_hourly_bars_closes = get_current_hourly_bars_cols(df, "close")
    current_hourly_bars_highs = get_current_hourly_bars_cols(df, "high")
    current_hourly_bars_lows = get_current_hourly_bars_cols(df, "low")
    current_hourly_bars_volumes = get_current_hourly_bars_cols(df, "low")

    # trend
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(pta.alma, length=20),
                             prefix="ALMA_20_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(pta.fwma, length=20),
                             prefix="FWMA_20_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(pta.hma, length=20),
                             prefix="HMA_20_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=pta.hwma,
                             prefix="HWMA_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(pta.jma, length=20),
                             prefix="JMA_20_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(pta.kama, length=20),
                             prefix="KAMA_20_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(pta.linreg, length=20, slope=True),
                             prefix="LINREG_20_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(pta.mcgd, length=20),
                             prefix="MCGD_20_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(pta.midpoint, length=20),
                             prefix="MIDPOINT_20_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(pta.midprice, length=20, min_periods=10),
                             prefix="MIDPPRICE_20_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(pta.pwma, length=20),
                             prefix="PWMA_20_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(pta.rma, length=20),
                             prefix="RMA_20_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(pta.sinwma, length=20),
                             prefix="SINWMA_20_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(it.rema, length=20),
                             prefix="REMA_20_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(pta.rma, length=20),
                             prefix="RMA_20_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(pta.ssf, length=20),
                             prefix="SSF_20_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(pta.t3, length=20),
                             prefix="T3_20_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(pta.tema, length=20),
                             prefix="TEMA_20_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(pta.trima, length=20),
                             prefix="TRIMA_20_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(pta.vidya, length=20),
                             prefix="VIDYA_20_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    # try use pure from pandas_ta and partial to pass hlcv
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_highs + current_hourly_bars_lows + current_hourly_bars_closes + current_hourly_bars_volumes],
                             func=partial(it.vwap, high_cols=current_hourly_bars_highs, low_cols=current_hourly_bars_lows,
                                          close_cols=current_hourly_bars_closes, volume_cols=current_hourly_bars_volumes),
                             prefix="VWAP_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=10)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_highs + current_hourly_bars_lows + current_hourly_bars_closes],
                             func=partial(it.wcp, high_cols=current_hourly_bars_highs, low_cols=current_hourly_bars_lows, close_cols=current_hourly_bars_closes),
                             prefix="WCP_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    df = add_indicator_to_df(base_df=df,
                             subset_df=df[current_hourly_bars_closes],
                             func=partial(pta.zlma, length=20),
                             prefix="ZLMA_MINUS",
                             postfix="HOUR",
                             number_of_cols_to_add=5)
    # momentum
