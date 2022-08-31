from typing import Callable

import pandas as pd


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
    new_df = subset_df.apply(func, axis=1).iloc[:, -1 * number_of_cols_to_add:]

    def form_col_name(index):
        return f"{prefix}_{index}" + (f"_{postfix}" if postfix else postfix)

    new_df.columns = [form_col_name(k) for k in reversed(range(number_of_cols_to_add))]
    return pd.concat([base_df, new_df], axis=1)
