import psycopg2
from dotenv import dotenv_values

conf = dotenv_values(".env")


def get_connection(other_conf=None, database=None):
    global conf
    conf = other_conf if other_conf else conf
    return psycopg2.connect(host=conf.get("JDBC_HOST"),
                            port=conf.get("JDBC_PORT"),
                            database=database if database else conf.get("JDBC_DATABASE"),
                            user=conf.get("JDBC_USER_NAME"),
                            password=conf.get("JDBC_PASSWORD"))


SELECTED_FEATURES = ['CURRENT_HOUR_VOLUME', 'CURRENT_H_VOL_TO_10_DAYS_AVG',
                     'CURRENT_H_VOL_TO_12_HOURS_AVG', 'CURRENT_H_VOL_TO_14_DAYS_AVG',
                     'CURRENT_H_VOL_TO_20_DAYS_AVG', 'CURRENT_H_VOL_TO_24_HOURS_AVG',
                     'CURRENT_H_VOL_TO_28_DAYS_AVG', 'CURRENT_H_VOL_TO_3_DAYS_AVG',
                     'CURRENT_H_VOL_TO_5_DAYS_AVG', 'CURRENT_H_VOL_TO_6_HOURS_AVG',
                     'CURRENT_H_VOL_TO_7_DAYS_AVG', 'DEMA_15_MINUS_0_HOUR',
                     'DEMA_15_MINUS_1_HOUR', 'DEMA_15_MINUS_2_HOUR',
                     'DEMA_15_MINUS_3_HOUR', 'DEMA_15_MINUS_4_HOUR',
                     'DEMA_15_MINUS_5_HOUR', 'DEMA_15_MINUS_6_HOUR',
                     'DEMA_15_MINUS_7_HOUR', 'DEMA_15_MINUS_8_HOUR',
                     'DEMA_15_MINUS_9_HOUR', 'DEMA_30_MINUS_0_HOUR',
                     'DEMA_30_MINUS_1_HOUR', 'DEMA_30_MINUS_2_HOUR',
                     'DEMA_30_MINUS_3_HOUR', 'DEMA_30_MINUS_4_HOUR',
                     'EMA_20_MINUS_0_HOUR', 'EMA_20_MINUS_1_HOUR',
                     'EMA_20_MINUS_2_HOUR', 'EMA_20_MINUS_3_HOUR',
                     'EMA_20_MINUS_4_HOUR', 'EMA_20_MINUS_5_HOUR',
                     'EMA_20_MINUS_6_HOUR', 'EMA_20_MINUS_7_HOUR',
                     'EMA_20_MINUS_8_HOUR', 'EMA_20_MINUS_9_HOUR',
                     'EMA_40_MINUS_0_HOUR', 'EMA_40_MINUS_1_HOUR',
                     'EMA_40_MINUS_2_HOUR', 'EMA_40_MINUS_3_HOUR',
                     'EMA_40_MINUS_4_HOUR', 'HOUR_01_CLOSE_HIGHER_THAN_02',
                     'HOUR_01_CLOSE_HIGHER_THAN_06', 'HOUR_01_CLOSE_HIGHER_THAN_11',
                     'HOUR_01_CLOSE_HIGHER_THAN_21', 'HOUR_01_CLOSE_HIGHER_THAN_41',
                     'HOUR_02_CLOSE_HIGHER_THAN_03', 'HOUR_02_CLOSE_HIGHER_THAN_07',
                     'HOUR_02_CLOSE_HIGHER_THAN_12', 'HOUR_02_CLOSE_HIGHER_THAN_22',
                     'HOUR_02_CLOSE_HIGHER_THAN_42', 'HOUR_03_CLOSE_HIGHER_THAN_04',
                     'HOUR_03_CLOSE_HIGHER_THAN_08', 'HOUR_03_CLOSE_HIGHER_THAN_13',
                     'HOUR_03_CLOSE_HIGHER_THAN_23', 'HOUR_03_CLOSE_HIGHER_THAN_43',
                     'HOUR_04_CLOSE_HIGHER_THAN_05', 'HOUR_04_CLOSE_HIGHER_THAN_09',
                     'HOUR_04_CLOSE_HIGHER_THAN_14', 'HOUR_04_CLOSE_HIGHER_THAN_24',
                     'HOUR_04_CLOSE_HIGHER_THAN_44', 'HOUR_05_CLOSE_HIGHER_THAN_06',
                     'HOUR_05_CLOSE_HIGHER_THAN_10', 'HOUR_05_CLOSE_HIGHER_THAN_15',
                     'HOUR_05_CLOSE_HIGHER_THAN_25', 'HOUR_05_CLOSE_HIGHER_THAN_45',
                     'HOUR_06_CLOSE_HIGHER_THAN_07', 'HOUR_06_CLOSE_HIGHER_THAN_11',
                     'HOUR_06_CLOSE_HIGHER_THAN_16', 'HOUR_06_CLOSE_HIGHER_THAN_26',
                     'HOUR_06_CLOSE_HIGHER_THAN_46', 'HOUR_07_CLOSE_HIGHER_THAN_08',
                     'HOUR_07_CLOSE_HIGHER_THAN_12', 'HOUR_07_CLOSE_HIGHER_THAN_17',
                     'HOUR_07_CLOSE_HIGHER_THAN_27', 'HOUR_07_CLOSE_HIGHER_THAN_47',
                     'HOUR_08_CLOSE_HIGHER_THAN_09', 'HOUR_08_CLOSE_HIGHER_THAN_13',
                     'HOUR_08_CLOSE_HIGHER_THAN_18', 'HOUR_08_CLOSE_HIGHER_THAN_28',
                     'HOUR_08_CLOSE_HIGHER_THAN_48', 'HOUR_09_CLOSE_HIGHER_THAN_10',
                     'HOUR_09_CLOSE_HIGHER_THAN_14', 'HOUR_09_CLOSE_HIGHER_THAN_19',
                     'HOUR_09_CLOSE_HIGHER_THAN_29', 'HOUR_10_CLOSE_HIGHER_THAN_11',
                     'HOUR_10_CLOSE_HIGHER_THAN_15', 'HOUR_10_CLOSE_HIGHER_THAN_20',
                     'HOUR_10_CLOSE_HIGHER_THAN_30', 'HOUR_11_CLOSE_HIGHER_THAN_12',
                     'HOUR_11_CLOSE_HIGHER_THAN_16', 'HOUR_11_CLOSE_HIGHER_THAN_21',
                     'HOUR_11_CLOSE_HIGHER_THAN_31', 'HOUR_12_CLOSE_HIGHER_THAN_13',
                     'HOUR_12_CLOSE_HIGHER_THAN_17', 'HOUR_12_CLOSE_HIGHER_THAN_22',
                     'HOUR_12_CLOSE_HIGHER_THAN_32', 'HOUR_13_CLOSE_HIGHER_THAN_14',
                     'HOUR_13_CLOSE_HIGHER_THAN_18', 'HOUR_13_CLOSE_HIGHER_THAN_23',
                     'HOUR_13_CLOSE_HIGHER_THAN_33', 'HOUR_14_CLOSE_HIGHER_THAN_15',
                     'HOUR_14_CLOSE_HIGHER_THAN_19', 'HOUR_14_CLOSE_HIGHER_THAN_24',
                     'HOUR_14_CLOSE_HIGHER_THAN_34', 'HOUR_15_CLOSE_HIGHER_THAN_16',
                     'HOUR_15_CLOSE_HIGHER_THAN_20', 'HOUR_15_CLOSE_HIGHER_THAN_25',
                     'HOUR_15_CLOSE_HIGHER_THAN_35', 'HOUR_16_CLOSE_HIGHER_THAN_17',
                     'HOUR_16_CLOSE_HIGHER_THAN_21', 'HOUR_16_CLOSE_HIGHER_THAN_26',
                     'HOUR_16_CLOSE_HIGHER_THAN_36', 'HOUR_17_CLOSE_HIGHER_THAN_18',
                     'HOUR_17_CLOSE_HIGHER_THAN_22', 'HOUR_17_CLOSE_HIGHER_THAN_27',
                     'HOUR_17_CLOSE_HIGHER_THAN_37', 'HOUR_18_CLOSE_HIGHER_THAN_19',
                     'HOUR_18_CLOSE_HIGHER_THAN_23', 'HOUR_18_CLOSE_HIGHER_THAN_28',
                     'HOUR_18_CLOSE_HIGHER_THAN_38', 'HOUR_19_CLOSE_HIGHER_THAN_20',
                     'HOUR_19_CLOSE_HIGHER_THAN_24', 'HOUR_19_CLOSE_HIGHER_THAN_29',
                     'HOUR_19_CLOSE_HIGHER_THAN_39', 'HOUR_20_CLOSE_HIGHER_THAN_21',
                     'HOUR_20_CLOSE_HIGHER_THAN_25', 'HOUR_20_CLOSE_HIGHER_THAN_30',
                     'HOUR_20_CLOSE_HIGHER_THAN_40', 'HOUR_21_CLOSE_HIGHER_THAN_22',
                     'HOUR_21_CLOSE_HIGHER_THAN_26', 'HOUR_21_CLOSE_HIGHER_THAN_31',
                     'HOUR_21_CLOSE_HIGHER_THAN_41', 'HOUR_22_CLOSE_HIGHER_THAN_23',
                     'HOUR_22_CLOSE_HIGHER_THAN_27', 'HOUR_22_CLOSE_HIGHER_THAN_32',
                     'HOUR_22_CLOSE_HIGHER_THAN_42', 'HOUR_23_CLOSE_HIGHER_THAN_24',
                     'HOUR_23_CLOSE_HIGHER_THAN_28', 'HOUR_23_CLOSE_HIGHER_THAN_33',
                     'HOUR_23_CLOSE_HIGHER_THAN_43', 'HOUR_24_CLOSE_HIGHER_THAN_25',
                     'HOUR_24_CLOSE_HIGHER_THAN_29', 'HOUR_24_CLOSE_HIGHER_THAN_34',
                     'HOUR_24_CLOSE_HIGHER_THAN_44', 'HOUR_25_CLOSE_HIGHER_THAN_26',
                     'HOUR_25_CLOSE_HIGHER_THAN_30', 'HOUR_25_CLOSE_HIGHER_THAN_35',
                     'HOUR_25_CLOSE_HIGHER_THAN_45', 'HOUR_26_CLOSE_HIGHER_THAN_27',
                     'HOUR_26_CLOSE_HIGHER_THAN_31', 'HOUR_26_CLOSE_HIGHER_THAN_36',
                     'HOUR_26_CLOSE_HIGHER_THAN_46', 'HOUR_27_CLOSE_HIGHER_THAN_28',
                     'HOUR_27_CLOSE_HIGHER_THAN_32', 'HOUR_27_CLOSE_HIGHER_THAN_37',
                     'HOUR_27_CLOSE_HIGHER_THAN_47', 'HOUR_28_CLOSE_HIGHER_THAN_29',
                     'HOUR_28_CLOSE_HIGHER_THAN_33', 'HOUR_28_CLOSE_HIGHER_THAN_38',
                     'HOUR_28_CLOSE_HIGHER_THAN_48', 'HOUR_29_CLOSE_HIGHER_THAN_30',
                     'HOUR_29_CLOSE_HIGHER_THAN_34', 'HOUR_29_CLOSE_HIGHER_THAN_39',
                     'HOUR_30_CLOSE_HIGHER_THAN_31', 'HOUR_30_CLOSE_HIGHER_THAN_35',
                     'HOUR_30_CLOSE_HIGHER_THAN_40', 'HOUR_31_CLOSE_HIGHER_THAN_32',
                     'HOUR_31_CLOSE_HIGHER_THAN_36', 'HOUR_31_CLOSE_HIGHER_THAN_41',
                     'HOUR_32_CLOSE_HIGHER_THAN_33', 'HOUR_32_CLOSE_HIGHER_THAN_37',
                     'HOUR_32_CLOSE_HIGHER_THAN_42', 'HOUR_33_CLOSE_HIGHER_THAN_34',
                     'HOUR_33_CLOSE_HIGHER_THAN_38', 'HOUR_33_CLOSE_HIGHER_THAN_43',
                     'HOUR_34_CLOSE_HIGHER_THAN_35', 'HOUR_34_CLOSE_HIGHER_THAN_39',
                     'HOUR_34_CLOSE_HIGHER_THAN_44', 'HOUR_35_CLOSE_HIGHER_THAN_36',
                     'HOUR_35_CLOSE_HIGHER_THAN_40', 'HOUR_35_CLOSE_HIGHER_THAN_45',
                     'HOUR_36_CLOSE_HIGHER_THAN_37', 'HOUR_36_CLOSE_HIGHER_THAN_41',
                     'HOUR_36_CLOSE_HIGHER_THAN_46', 'HOUR_37_CLOSE_HIGHER_THAN_38',
                     'HOUR_37_CLOSE_HIGHER_THAN_42', 'HOUR_37_CLOSE_HIGHER_THAN_47',
                     'HOUR_38_CLOSE_HIGHER_THAN_39', 'HOUR_38_CLOSE_HIGHER_THAN_43',
                     'HOUR_38_CLOSE_HIGHER_THAN_48', 'HOUR_39_CLOSE_HIGHER_THAN_40',
                     'HOUR_39_CLOSE_HIGHER_THAN_44', 'HOUR_40_CLOSE_HIGHER_THAN_41',
                     'HOUR_40_CLOSE_HIGHER_THAN_45', 'HOUR_41_CLOSE_HIGHER_THAN_42',
                     'HOUR_41_CLOSE_HIGHER_THAN_46', 'HOUR_42_CLOSE_HIGHER_THAN_43',
                     'HOUR_42_CLOSE_HIGHER_THAN_47', 'HOUR_43_CLOSE_HIGHER_THAN_44',
                     'HOUR_43_CLOSE_HIGHER_THAN_48', 'HOUR_44_CLOSE_HIGHER_THAN_45',
                     'HOUR_45_CLOSE_HIGHER_THAN_46', 'HOUR_46_CLOSE_HIGHER_THAN_47',
                     'HOUR_47_CLOSE_HIGHER_THAN_48', 'LATEST_HOUR_CLOSE_HIGHER_THAN_01',
                     'LATEST_HOUR_CLOSE_HIGHER_THAN_05',
                     'LATEST_HOUR_CLOSE_HIGHER_THAN_10',
                     'LATEST_HOUR_CLOSE_HIGHER_THAN_20',
                     'LATEST_HOUR_CLOSE_HIGHER_THAN_40', 'RAPO_MINUS_0_HOUR',
                     'RAPO_MINUS_1_HOUR', 'RAPO_MINUS_2_HOUR', 'RAPO_MINUS_3_HOUR',
                     'RAPO_MINUS_4_HOUR', 'RAPO_MINUS_5_HOUR', 'RAPO_MINUS_6_HOUR',
                     'RAPO_MINUS_7_HOUR', 'RAPO_MINUS_8_HOUR', 'RAPO_MINUS_9_HOUR',
                     'RSI_15_MINUS_0_HOUR', 'RSI_15_MINUS_1_HOUR',
                     'RSI_15_MINUS_2_HOUR', 'RSI_15_MINUS_3_HOUR',
                     'RSI_15_MINUS_4_HOUR', 'RSI_15_MINUS_5_HOUR',
                     'RSI_15_MINUS_6_HOUR', 'RSI_15_MINUS_7_HOUR',
                     'RSI_15_MINUS_8_HOUR', 'RSI_15_MINUS_9_HOUR',
                     'RSI_30_MINUS_0_HOUR', 'RSI_30_MINUS_1_HOUR',
                     'RSI_30_MINUS_2_HOUR', 'RSI_30_MINUS_3_HOUR',
                     'RSI_30_MINUS_4_HOUR', 'WILLR_15_MINUS_0_HOUR',
                     'WILLR_15_MINUS_1_HOUR', 'WILLR_15_MINUS_2_HOUR',
                     'WILLR_15_MINUS_3_HOUR', 'WILLR_15_MINUS_4_HOUR',
                     'WILLR_15_MINUS_5_HOUR', 'WILLR_15_MINUS_6_HOUR',
                     'WILLR_15_MINUS_7_HOUR', 'WILLR_15_MINUS_8_HOUR',
                     'WILLR_15_MINUS_9_HOUR', 'WILLR_30_MINUS_0_HOUR',
                     'WILLR_30_MINUS_1_HOUR', 'WILLR_30_MINUS_2_HOUR',
                     'WILLR_30_MINUS_3_HOUR', 'WILLR_30_MINUS_4_HOUR',
                     '_01_H_BARS_VOL_TO_28D_AVG_H_VOL',
                     '_02_H_BARS_VOL_TO_28D_AVG_H_VOL',
                     '_03_H_BARS_VOL_TO_28D_AVG_H_VOL',
                     'btc_stats_statsMap_-10_days_avg1HourVolume',
                     'btc_stats_statsMap_-10_days_changeRate',
                     'btc_stats_statsMap_-10_days_high',
                     'btc_stats_statsMap_-12_hours_avg1HourVolume',
                     'btc_stats_statsMap_-12_hours_changeRate',
                     'btc_stats_statsMap_-12_hours_high',
                     'btc_stats_statsMap_-14_days_avg1HourVolume',
                     'btc_stats_statsMap_-14_days_changeRate',
                     'btc_stats_statsMap_-14_days_high',
                     'btc_stats_statsMap_-20_days_avg1HourVolume',
                     'btc_stats_statsMap_-20_days_changeRate',
                     'btc_stats_statsMap_-20_days_high',
                     'btc_stats_statsMap_-24_hours_avg1HourVolume',
                     'btc_stats_statsMap_-24_hours_changeRate',
                     'btc_stats_statsMap_-24_hours_high',
                     'btc_stats_statsMap_-28_days_avg1HourVolume',
                     'btc_stats_statsMap_-28_days_changeRate',
                     'btc_stats_statsMap_-28_days_close',
                     'btc_stats_statsMap_-28_days_high',
                     'btc_stats_statsMap_-3_days_avg1HourVolume',
                     'btc_stats_statsMap_-3_days_changeRate',
                     'btc_stats_statsMap_-3_days_high',
                     'btc_stats_statsMap_-5_days_avg1HourVolume',
                     'btc_stats_statsMap_-5_days_changeRate',
                     'btc_stats_statsMap_-5_days_high',
                     'btc_stats_statsMap_-60_days_avg1HourVolume',
                     'btc_stats_statsMap_-60_days_changeRate',
                     'btc_stats_statsMap_-60_days_close',
                     'btc_stats_statsMap_-60_days_high',
                     'btc_stats_statsMap_-6_hours_avg1HourVolume',
                     'btc_stats_statsMap_-6_hours_changeRate',
                     'btc_stats_statsMap_-6_hours_high',
                     'btc_stats_statsMap_-7_days_avg1HourVolume',
                     'btc_stats_statsMap_-7_days_changeRate',
                     'btc_stats_statsMap_-7_days_high', 'current_close',
                     'current_currentHourlyBarVolume',
                     'current_currentMinutelyBarVolume', 'current_hour_bars_01_close',
                     'current_hour_bars_01_volume', 'current_hour_bars_02_close',
                     'current_hour_bars_02_volume', 'current_hour_bars_03_close',
                     'current_hour_bars_03_volume', 'current_hour_bars_04_close',
                     'current_hour_bars_04_volume', 'current_hour_bars_05_close',
                     'current_hour_bars_05_volume', 'current_hour_bars_06_close',
                     'current_hour_bars_06_volume', 'current_hour_bars_07_close',
                     'current_hour_bars_07_volume', 'current_hour_bars_08_close',
                     'current_hour_bars_08_volume', 'current_hour_bars_09_close',
                     'current_hour_bars_09_volume', 'current_hour_bars_10_close',
                     'current_hour_bars_10_volume', 'current_hour_bars_11_close',
                     'current_hour_bars_11_volume', 'current_hour_bars_12_close',
                     'current_hour_bars_12_volume', 'current_hour_bars_13_close',
                     'current_hour_bars_13_volume', 'current_hour_bars_14_close',
                     'current_hour_bars_14_volume', 'current_hour_bars_15_close',
                     'current_hour_bars_15_volume', 'current_hour_bars_16_close',
                     'current_hour_bars_16_volume', 'current_hour_bars_17_close',
                     'current_hour_bars_17_volume', 'current_hour_bars_18_close',
                     'current_hour_bars_18_volume', 'current_hour_bars_19_close',
                     'current_hour_bars_19_volume', 'current_hour_bars_20_close',
                     'current_hour_bars_20_volume', 'current_hour_bars_21_close',
                     'current_hour_bars_21_volume', 'current_hour_bars_22_close',
                     'current_hour_bars_22_volume', 'current_hour_bars_23_close',
                     'current_hour_bars_23_volume', 'current_hour_bars_24_close',
                     'current_hour_bars_24_volume', 'current_hour_bars_25_close',
                     'current_hour_bars_25_volume', 'current_hour_bars_26_close',
                     'current_hour_bars_26_volume', 'current_hour_bars_27_close',
                     'current_hour_bars_27_volume', 'current_hour_bars_28_close',
                     'current_hour_bars_28_volume', 'current_hour_bars_29_close',
                     'current_hour_bars_29_volume', 'current_lastMinutelyBar_high',
                     'current_lastMinutelyBar_low', 'current_lastMinutelyBar_open',
                     'current_lastMinutelyBar_volume', 'current_open',
                     'current_xinfo_ask', 'current_xinfo_averagePrice',
                     'current_xinfo_bid', 'current_xinfo_changePrice',
                     'current_xinfo_changeRate', 'current_xinfo_high',
                     'current_xinfo_last', 'current_xinfo_low', 'current_xinfo_volume',
                     'current_xinfo_volumeValue',
                     'history_statsMap_-10_days_avg1HourVolume',
                     'history_statsMap_-10_days_changeRate',
                     'history_statsMap_-12_hours_avg1HourVolume',
                     'history_statsMap_-12_hours_changeRate',
                     'history_statsMap_-14_days_avg1HourVolume',
                     'history_statsMap_-14_days_changeRate',
                     'history_statsMap_-20_days_avg1HourVolume',
                     'history_statsMap_-20_days_changeRate',
                     'history_statsMap_-24_hours_avg1HourVolume',
                     'history_statsMap_-24_hours_changeRate',
                     'history_statsMap_-28_days_avg1HourVolume',
                     'history_statsMap_-28_days_changeRate',
                     'history_statsMap_-28_days_close',
                     'history_statsMap_-3_days_avg1HourVolume',
                     'history_statsMap_-3_days_changeRate',
                     'history_statsMap_-5_days_avg1HourVolume',
                     'history_statsMap_-5_days_changeRate',
                     'history_statsMap_-6_hours_avg1HourVolume',
                     'history_statsMap_-6_hours_changeRate',
                     'history_statsMap_-7_days_avg1HourVolume',
                     'history_statsMap_-7_days_changeRate', 'latest_hour_close',
                     'latest_hour_high', 'latest_hour_low', 'latest_hour_open',
                     'latest_hour_volume', 'price', 'same_signal_in_last_24_hour',
                     'same_signal_in_last_24_hours', 'same_signal_in_last_3_hour',
                     'same_signal_profit_in_last_24_hours',
                     'same_signal_profit_in_last_3_hours']
