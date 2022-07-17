import logging
from datetime import datetime, timedelta

import pandas as pd

from config import SELECTED_FEATURES

HOURS_TILL_ELIGIBLE = 30


class DataPreprocessor:
    def __init__(self, raw_flat_data=None, is_train=False):
        self.raw_flat_data = raw_flat_data
        self.is_train = is_train

    def provide_ready_df(self, df=None) -> pd.DataFrame:
        logging.info(f"Starting data preprocessing, training mode - {self.is_train}")
        if df:
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
            df = DataPreprocessor.remove_date_range(
                df,
                datetime.fromisoformat("2022-05-11"),
                datetime.fromisoformat("2022-05-24")
            )
            # remove semi missing data 2022-06-30 16:30 to 2022-07-02 11:17
            df = DataPreprocessor.remove_date_range(
                df,
                datetime.fromisoformat("2022-06-30 16:30"),
                datetime.fromisoformat("2022-07-02 11:17"))
            # removing all notifications which are recent, because data is not complete for them
            df = DataPreprocessor.remove_most_recent_data(df)

            df = DataPreprocessor.add_regression_label_columns(df)

        df = DataPreprocessor.drop_categorical_features(df)

        if self.is_train:
            df = df.drop_duplicates(keep='first')

        # df = self.drop_columns_having_same_values(df)
        df = self.select_only_required_features(df)
        df = DataPreprocessor.add_missing_columns(df)
        if not self.is_train:
            df = df.drop(['label_up_return', 'label_down_return'], axis=1, errors='ignore')
        # Initial try, with filling df with 0
        df = df.fillna(0)

        DataPreprocessor.replace_negative_volumes(df)

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

        logging.info("Finished dataframe preparation")
        logging.info(f"Result DataFrame shape is {df.shape}")
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
        # for f in SELECTED_FEATURES:
        #     if f not in df.columns:
        #         df[f] = 0
        return df

    @staticmethod
    def add_regression_label_columns(df):
        logging.info("Adding regression label columns")
        df['label_up_return'] = (df['highest_since_notified'] - df['price']) / df['price'] * 100
        df['label_down_return'] = (df['lowest_since_notified'] - df['price']) / df['price'] * 100
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
            if c.endswith('_volume'):
                df[c] = df[c].apply(lambda x: x if x > 0 else 0)

    @staticmethod
    def add_current_hour_volume(df):
        logging.info("Adding current hour volume")
        curr_vols = [c for c in df.columns if str(c).startswith('current_min') and str(c).endswith('volume')]
        df['current_hour_volume'] = df[curr_vols].sum(axis=1)

    @staticmethod
    def add_current_hour_volume_to_historical_volumes_coef(df):
        logging.info("Adding current hour volume to historical volumes coefficent")
        history_vol_cols = []
        history_stats_vol_cols = [c for c in df.columns if str(c).startswith('history_statsMap_-') and str(c).endswith('avg1HourVolume')]
        print(history_stats_vol_cols)
        for k in history_stats_vol_cols:
            day = k.lstrip('history_statsMap_-').rstrip('avg1HourVolume')
            col_name = f'current_h_vol_to_{day}avg'
            logging.info(f"Adding column {col_name} to df")
            history_vol_cols.append(col_name)
            df[col_name] = df['current_hour_volume'] / df[k]
        while True:
            yield history_vol_cols

    @staticmethod
    def add_change_since_1_2_3_hours_back(df):
        latest_close_cols = ['change_since_01_hour_bars', 'change_since_02_hour_bars', 'change_since_03_hour_bars']
        df['change_since_01_hour_bars'] = (df['price'] - df['current_hour_bars_01_close']) / df['current_hour_bars_01_close'] * 100
        df['change_since_02_hour_bars'] = (df['price'] - df['current_hour_bars_02_close']) / df['current_hour_bars_02_close'] * 100
        df['change_since_03_hour_bars'] = (df['price'] - df['current_hour_bars_03_close']) / df['current_hour_bars_03_close'] * 100
        df_change_since_previous = df[[*latest_close_cols, 'label_up_return', 'label_down_return']]
        while True:
            yield df_change_since_previous

    @staticmethod
    def add_1_2_3_h_bars_vol_to_history_vol_coef(df):
        df['_01_h_bars_vol_to_28d_avg_h_vol'] = df['current_hour_bars_01_volume'] / df['history_statsMap_-28_days_avg1HourVolume']
        df['_02_h_bars_vol_to_28d_avg_h_vol'] = df['current_hour_bars_02_volume'] / df['history_statsMap_-28_days_avg1HourVolume']
        df['_03_h_bars_vol_to_28d_avg_h_vol'] = df['current_hour_bars_03_volume'] / df['history_statsMap_-28_days_avg1HourVolume']

    @staticmethod
    def remove_outliers(df, history_vol_cols, part=0.01):
        # cleaning outliers in data
        df_vol = df[[*next(history_vol_cols), 'label_up_return', 'label_down_return']]
        x = df_vol.drop(['label_up_return', 'label_down_return'], 1)
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
