import logging
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import SELECTED_FEATURES
from indicators.core import add_all_indicators_to_df

HOURS_TILL_ELIGIBLE = 30


class DataPreprocessor:
    def __init__(self, raw_flat_data=None, is_train=False):
        self.raw_flat_data = raw_flat_data
        self.is_train = is_train

    def provide_ready_df(self, df=None) -> pd.DataFrame:
        logging.info(f"Starting data preprocessing, training mode - {self.is_train}")
        if df is not None:
            raw_df = df
        else:
            raw_flat_data = self.raw_flat_data
            if self.is_train:
                raw_flat_data = DataPreprocessor.remove_partial_data(raw_flat_data)

            raw_df = pd.DataFrame(data=raw_flat_data)

        df = raw_df.reindex(sorted(raw_df.columns), axis=1)

        if self.is_train:
            df = DataPreprocessor.remove_corrupt_data(df)
            # remove semi missing data 2022-05-11 to 2022-05-24
            df = DataPreprocessor.remove_date_range(df,
                                                    datetime.fromisoformat("2022-05-11"),
                                                    datetime.fromisoformat("2022-05-24")
                                                    )
            # remove semi missing data 2022-06-30 16:30 to 2022-07-02 11:17
            df = DataPreprocessor.remove_date_range(df,
                                                    datetime.fromisoformat("2022-06-30 16:30"),
                                                    datetime.fromisoformat("2022-07-02 11:17"))
            # removing all notifications which are recent, because data is not complete for them
            df = DataPreprocessor.remove_most_recent_data(df)

            df = DataPreprocessor.add_regression_label_columns(df)

        df = DataPreprocessor.drop_categorical_features(df)

        if self.is_train:
            df = df.drop_duplicates(keep='first')

        DataPreprocessor.add_higher_high_col(df)
        DataPreprocessor.add_higher_high_col(df, 5)
        DataPreprocessor.add_higher_high_col(df, 10)
        DataPreprocessor.add_higher_high_col(df, 20)
        DataPreprocessor.add_higher_high_col(df, 40)

        DataPreprocessor.replace_negative_volumes(df)

        df = add_all_indicators_to_df(df)

        df = DataPreprocessor.drop_minutely_bar_cols(df, is_train=self.is_train)
        df = DataPreprocessor.drop_hourly_bar_ohl_cols(df, is_train=self.is_train)
        df = DataPreprocessor.drop_btc_stats_map_ol_cols(df, is_train=self.is_train)
        df = DataPreprocessor.drop_history_stats_map_ohl_cols(df, is_train=self.is_train)
        df = DataPreprocessor.drop_highly_correlated_features(df)
        df = DataPreprocessor.drop_highly_missing_features(df)

        if not self.is_train:
            df = DataPreprocessor.add_missing_columns(df)
        # df = DataPreprocessor.add_missing_columns(df)
        if not self.is_train:
            df = df.drop(['LABEL_UP_RETURN', 'LABEL_DOWN_RETURN'], axis=1, errors='ignore')
        # Initial try, with filling df with 0
        df = df.fillna(0)

        DataPreprocessor.add_current_hour_volume(df)

        history_vol_cols = DataPreprocessor.add_current_hour_volume_to_historical_volumes_coef(df)

        if self.is_train:
            df = next(DataPreprocessor.remove_outliers(df, history_vol_cols))
            # from scipy import stats
            # df_vol_coef_clean = df_vol[(np.abs(stats.zscore(df_vol)) < 4).all(axis=1)]

            # removing those which don't have -3 hours of data
            df = DataPreprocessor.remove_rows_with_less_than_3_hours(df)
        else:
            next(history_vol_cols)

        DataPreprocessor.add_change_since_1_2_3_hours_back(df)

        self.add_1_2_3_h_bars_vol_to_history_vol_coef(df)
        df = DataPreprocessor.clean_dataset(df)

        logging.info("Finished dataframe preparation")
        logging.info(f"Result DataFrame shape is {df.shape} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        df = df.reindex(sorted(df.columns), axis=1)
        return df

    @staticmethod
    def select_only_required_features(df):
        return df.loc[:, df.columns.isin(SELECTED_FEATURES)]

    @staticmethod
    def remove_rows_with_less_than_3_hours(df):
        df = df[
            (df['current_hour_bars_01_close'] != 0) &
            (df['current_hour_bars_02_close'] != 0) &
            (df['current_hour_bars_03_close'] != 0)
            ]
        return df

    @staticmethod
    def remove_most_recent_data(df):
        logging.info(f"Filtering training data label is recent")
        df = df[df.notification_date < (datetime.now() - timedelta(hours=HOURS_TILL_ELIGIBLE))]
        return df

    @staticmethod
    def remove_date_range(df, from_, to_):
        logging.info(f"Removing date range from {from_} to {to_} from data")
        df = df[(df.notification_date < from_) | (df.notification_date > to_)]
        return df

    @staticmethod
    def remove_partial_data(raw_flat_data):
        # clean data set, remove rows that include not all data
        logging.info(f"Filtering incomplete training data where btc and history is missing")
        raw_flat_data = [r for r in raw_flat_data if r['id'] > 1162]
        return raw_flat_data

    @staticmethod
    def add_missing_columns(df):
        df = df.reindex(columns=SELECTED_FEATURES)
        return df

    @staticmethod
    def add_regression_label_columns(df):
        logging.info("Adding regression label columns")
        df['LABEL_UP_RETURN'] = (df['highest_since_notified'] - df['price']) / df['price'] * 100
        df['LABEL_DOWN_RETURN'] = (df['lowest_since_notified'] - df['price']) / df['price'] * 100
        return df

    @staticmethod
    def drop_categorical_features(df):
        logging.info("Dropping categorical features")
        date_cols = [c for c in df.columns if str(c).endswith('days_time') or str(c).endswith('hours_time')]
        # Dropping categorical features
        return df.drop(['btc_stats_modifiedAt',
                        'current_previouslyAccessed_24HoursBackMinute',
                        'btc_stats_ticker',
                        'current_ticker',
                        'current_xinfo_ticker',
                        'filter_name',
                        'history_modifiedAt',
                        'history_ticker',
                        'ticker',
                        'lowest_since_notified',
                        'notification_date',
                        'highest_since_notified',
                        'id',
                        'current_hourlyBarsTime',
                        *date_cols
                        ], axis=1, errors='ignore')

    @staticmethod
    def replace_negative_volumes(df):
        # if volume is negative - replace with zero
        logging.info("Replacing negative volumes in df")
        for c in df.columns:
            if c.endswith('_volume') or c.endswith('_VOLUME'):
                df[c] = df[c].apply(lambda x: x if x > 0 else 0)

    @staticmethod
    def add_current_hour_volume(df):
        logging.info("Adding current hour volume")
        curr_vols = [c for c in df.columns if str(c).startswith('current_min') and str(c).endswith('volume')]
        df['CURRENT_HOUR_VOLUME'] = df[curr_vols].sum(axis=1)

    # current over -1 hour bar comparison
    @staticmethod
    def add_higher_high_col(df, shift=1, is_train=False):
        logging.info(f"Adding higher high columns to df of shape {df.shape} with shift of {shift}")
        current_hourly_bars_cols = [col for col in df if col.startswith('current_hour_bars') and col.endswith('close')]

        current_hourly_bars_cols = sorted(current_hourly_bars_cols, reverse=True)
        current_hourly_bars_cols.append("latest_hour_close")
        if is_train:
            assert len(current_hourly_bars_cols) == 49
            assert current_hourly_bars_cols[-2] == 'current_hour_bars_01_close'

        print(f"Current hourly bars cols: {len(current_hourly_bars_cols)}")
        DIGIT_PATTERN = r'\d+'
        higher_high_cols = []
        for current, previous in zip(current_hourly_bars_cols[shift:], current_hourly_bars_cols):
            current_numbers = re.findall(DIGIT_PATTERN, current)
            current_number = f'HOUR_{current_numbers[0]}_CLOSE' if current_numbers else current
            previous_numbers = re.findall(DIGIT_PATTERN, previous)
            previous_number = previous_numbers[0] if previous_numbers else previous
            is_higher_col = f'{current_number}_HIGHER_THAN_{previous_number}'.upper()
            df[is_higher_col] = df[current] > df[previous]
            df[is_higher_col] = df[is_higher_col].astype(int)
            higher_high_cols.append(is_higher_col)
        return higher_high_cols

    @staticmethod
    def add_current_hour_volume_to_historical_volumes_coef(df):
        logging.info("Adding current hour volume to historical volumes coefficent")
        history_vol_cols = []
        history_stats_vol_cols = [c for c in df.columns if str(c).startswith('history_statsMap_-') and str(c).endswith('avg1HourVolume')]
        print(history_stats_vol_cols)
        for k in history_stats_vol_cols:
            day = k.lstrip('history_statsMap_-').rstrip('avg1HourVolume')
            col_name = f'CURRENT_H_VOL_TO_{day}AVG'.upper()
            logging.info(f"Adding column {col_name} to df")
            history_vol_cols.append(col_name)
            df[col_name] = df['latest_hour_volume'] / df[k]
        while True:
            yield history_vol_cols

    @staticmethod
    def add_change_since_1_2_3_hours_back(df):
        latest_close_cols = ['CHANGE_SINCE_01_HOUR_BARS', 'CHANGE_SINCE_02_HOUR_BARS', 'CHANGE_SINCE_03_HOUR_BARS']
        df['CHANGE_SINCE_01_HOUR_BARS'] = (df['price'] - df['current_hour_bars_01_close']) / df['current_hour_bars_01_close'] * 100
        df['CHANGE_SINCE_02_HOUR_BARS'] = (df['price'] - df['current_hour_bars_02_close']) / df['current_hour_bars_02_close'] * 100
        df['CHANGE_SINCE_03_HOUR_BARS'] = (df['price'] - df['current_hour_bars_03_close']) / df['current_hour_bars_03_close'] * 100
        df_change_since_previous = df[[*latest_close_cols, 'LABEL_UP_RETURN', 'LABEL_DOWN_RETURN']]
        while True:
            yield df_change_since_previous

    @staticmethod
    def add_1_2_3_h_bars_vol_to_history_vol_coef(df):
        df['_01_H_BARS_VOL_TO_28D_AVG_H_VOL'] = df['current_hour_bars_01_volume'] / df['history_statsMap_-28_days_avg1HourVolume']
        df['_02_H_BARS_VOL_TO_28D_AVG_H_VOL'] = df['current_hour_bars_02_volume'] / df['history_statsMap_-28_days_avg1HourVolume']
        df['_03_H_BARS_VOL_TO_28D_AVG_H_VOL'] = df['current_hour_bars_03_volume'] / df['history_statsMap_-28_days_avg1HourVolume']

    @staticmethod
    def remove_outliers(df, history_vol_cols, part=0.01):
        # cleaning outliers in data
        df_vol = df[[*next(history_vol_cols), 'LABEL_UP_RETURN', 'LABEL_DOWN_RETURN']]
        x = df_vol.drop(['LABEL_UP_RETURN', 'LABEL_DOWN_RETURN'], 1)
        # cleaning outliers in data
        df_vol_coef_clean = df[((x > x.quantile(part)) & (x < x.quantile(1 - part))).all(1)]
        while True:
            yield df_vol_coef_clean

    @staticmethod
    def drop_columns_having_same_values(df):
        # drop columns with duplicated values, leave first one
        df = df.loc[:, ~df.apply(lambda y: y.duplicated(), axis=1).all()].copy()
        return df

    @staticmethod
    def remove_corrupt_data(df):
        # data in this range is broken due to the upstream bug
        return df[(df["id"] < 13865) | (df["id"] > 16787)]

    @staticmethod
    def drop_minutely_bar_cols(df, is_train=False):
        current_min_bars_cols = [col for col in df if col.startswith('current_min_bars')]
        if is_train:
            # should be 60 * 5
            assert len(current_min_bars_cols) == 60 * 5
        df = df.drop(current_min_bars_cols, axis=1)
        return df

    @staticmethod
    def drop_hourly_bar_ohl_cols(df, is_train=False):
        current_hourly_bars_partly_cols = [col for col in df if col.startswith('current_hour_bars') and not col.endswith('close') and not col.endswith('volume')]
        if is_train:
            # should be 48 * 3
            assert len(current_hourly_bars_partly_cols) == 48 * 3
        df = df.drop(current_hourly_bars_partly_cols, axis=1)
        return df

    @staticmethod
    def drop_history_stats_map_ohl_cols(df, is_train=False):
        history_stats_map__bars_partly_cols = [col for col in df if col.startswith('history_statsMap')
                                               and not col.endswith('close')
                                               and not col.endswith('Volume')
                                               and not col.endswith('changeRate')
                                               ]
        if is_train:
            # should be 48 * 3
            assert len(history_stats_map__bars_partly_cols) == 30
        df = df.drop(history_stats_map__bars_partly_cols, axis=1)
        return df

    @staticmethod
    def drop_btc_stats_map_ol_cols(df, is_train=False):
        btc_stats_stats_map_bars_partly_cols = [col for col in df if col.startswith('btc_stats_statsMap_')
                                                and not col.endswith('high')
                                                and not col.endswith('close')
                                                and not col.endswith('Volume')
                                                and not col.endswith('changeRate')
                                                ]
        if is_train:
            # should be 48 * 3
            assert len(btc_stats_stats_map_bars_partly_cols) == 22
        df = df.drop(btc_stats_stats_map_bars_partly_cols, axis=1)
        return df

    @staticmethod
    def drop_highly_correlated_features(df):
        highly_correlated_cols = [
            'btc_stats_statsMap_-10_days_close',
            'btc_stats_statsMap_-12_hours_close',
            'btc_stats_statsMap_-14_days_close',
            'btc_stats_statsMap_-20_days_close',
            'btc_stats_statsMap_-24_hours_close',
            'btc_stats_statsMap_-3_days_close',
            'btc_stats_statsMap_-5_days_close',
            'btc_stats_statsMap_-6_hours_close',
            'btc_stats_statsMap_-7_days_close',
            'history_statsMap_-10_days_close',
            'history_statsMap_-12_hours_close',
            'history_statsMap_-14_days_close',
            'history_statsMap_-20_days_close',
            'history_statsMap_-24_hours_close',
            'history_statsMap_-3_days_close',
            'history_statsMap_-5_days_close',
            'history_statsMap_-6_hours_close',
            'history_statsMap_-7_days_close',
            'current_lastMinutelyBar_close',
            'current_previousMinutelyBar_close',
            'current_previousMinutelyBar_high',
            'current_previousMinutelyBar_low',
            'current_previousMinutelyBar_open',
            'current_previousMinutelyBar_volume'
        ]
        df = df.drop(highly_correlated_cols, axis=1)
        return df

    @staticmethod
    def drop_highly_missing_features(df):
        sparse_features = ['current_hour_bars_30_close',
                           'current_hour_bars_30_volume',
                           'current_hour_bars_31_close',
                           'current_hour_bars_31_volume',
                           'current_hour_bars_32_close',
                           'current_hour_bars_32_volume',
                           'current_hour_bars_33_close',
                           'current_hour_bars_33_volume',
                           'current_hour_bars_34_close',
                           'current_hour_bars_34_volume',
                           'current_hour_bars_35_close',
                           'current_hour_bars_35_volume',
                           'current_hour_bars_36_close',
                           'current_hour_bars_36_volume',
                           'current_hour_bars_37_close',
                           'current_hour_bars_37_volume',
                           'current_hour_bars_38_close',
                           'current_hour_bars_38_volume',
                           'current_hour_bars_39_close',
                           'current_hour_bars_39_volume',
                           'current_hour_bars_40_close',
                           'current_hour_bars_40_volume',
                           'current_hour_bars_41_close',
                           'current_hour_bars_41_volume',
                           'current_hour_bars_42_close',
                           'current_hour_bars_42_volume',
                           'current_hour_bars_43_close',
                           'current_hour_bars_43_volume',
                           'current_hour_bars_44_close',
                           'current_hour_bars_44_volume',
                           'current_hour_bars_45_close',
                           'current_hour_bars_45_volume',
                           'current_hour_bars_46_close',
                           'current_hour_bars_46_volume',
                           'current_hour_bars_47_close',
                           'current_hour_bars_47_volume',
                           'current_hour_bars_48_close',
                           'current_hour_bars_48_volume'
                           ]
        df = df.drop(sparse_features, axis=1)
        return df

    @staticmethod
    def clean_dataset(df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)
