import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from bentos.service_v1 import RFCClassifierModelService
from config import get_connection
from data.db import execute_sql
from data.preprocessing import DataPreprocessor
from data.notifcation_preparation import flat_notifications_from_sql, prepare_dataset


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


class ModelFactory:
    def __init__(self):
        self.trained_models = {}

    @staticmethod
    def train_random_forest_classifier(x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
        print(list(x_train.columns))
        rfc = RandomForestClassifier()
        rfc.fit(x_train, y_train)
        return x_train, x_test, y_train, y_test, rfc

    @staticmethod
    def prepare_classification_label(df, threshold):
        col = 'label_up_return' if threshold > 0 else 'label_down_return'
        label_col_name = f'up_{threshold}_return' if threshold > 0 else f'down_{threshold * -1}_return'
        logging.info(f"Creating label column {label_col_name} with threshold of {threshold}")
        y = df[col]
        k = pd.DataFrame(y)
        conditions = [(k[col] >= threshold),
                      (k[col] < threshold)]
        y = pd.DataFrame(np.select(conditions, [1, 0]), columns=[f'label_{threshold}_{"up" if threshold > 0 else "down"}'])
        logging.info(y.value_counts())
        return label_col_name, y

    def prepare_models(self, df: pd.DataFrame, thresholds):
        label_cols = ['label_up_return', 'label_down_return']
        raw_x = df.drop(label_cols, axis=1)
        for thr in thresholds:
            logging.info(f"Training model for {thr} threshold")
            model_name, label_df = self.prepare_classification_label(df, thr)

            x, x_test, y, y_test, model = ModelFactory.train_random_forest_classifier(raw_x, label_df)
            # Make predictions for the test UP set
            y_predictions = model.predict(x_test)
            # View accuracy score
            logging.info(accuracy_score(y_test, y_predictions))
            self.trained_models[model_name] = model
            logging.info(f'Finished model training for {thr} threshold')


if __name__ == '__main__':
    conn = get_connection()
    notifications = execute_sql(conn, "select * from notifications order by id")
    raw_flat_data = prepare_dataset(flat_notifications_from_sql(notifications))
    data_preprocessor = DataPreprocessor(raw_flat_data, True)
    df = data_preprocessor.provide_ready_df()
    mf = ModelFactory()
    mf.prepare_models(df, [30, -10])
    #artifacts_list = [SklearnModelArtifact(k) for k in mf.trained_models.keys()]
    #model_callable = artifacts(artifacts_list)(RFCClassifierModelService)
    #model = model_callable()
    model = RFCClassifierModelService()
    for k, v in mf.trained_models.items():
        logging.info(f"Packing model {k}")
        model.pack(k, v)
    saved_path = model.save()
    logging.info(f'Saved model to {saved_path}')

    print("XXX")

# y_up = df['label_up_return']
# y_down = df['label_down_return']
#
# df_norm = min_max_scaler(df)
# x_norm = df_norm.drop(label_cols, axis=1)
# y_up_norm = df_norm['label_up_return']
# y_down_norm = df_norm['label_down_return']
#
# df_std = standard_scaler(df)
# x_std = df_std.drop(label_cols, axis=1)
# y_up_std = df_std['label_up_return']
# y_down_std = df_std['label_down_return']


# k = pd.DataFrame(y_down)
# down_conditions = [
#     (k['label_down_return'] >= -10),
#     (k['label_down_return'] < -10)
# ]
# y_down_class = pd.DataFrame(np.select(down_conditions, [1, 0]), columns=['label_10_down'])
# y_down_class.value_counts()
# x_down_class, x_down_test_class, y_down_class, y_down_test_class, down_random_forest_model = train_random_forest_classifier(x, y_down_class)
# # Make predictions for the test DOWN set
# y_down_predictions = down_random_forest_model.predict(x_down_test_class)
# # View accuracy score
# accuracy_score(y_down_test_class, y_down_predictions)
