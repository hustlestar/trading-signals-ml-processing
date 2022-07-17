import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

from bentos.service_v1 import SignalClassifierModelService
from config import get_connection
from data.db import execute_sql
from data.preprocessing import DataPreprocessor
from data.notifcation_preparation import flat_notifications_from_sql, prepare_dataset

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


class ModelFactory:
    def __init__(self, model_name=None):
        self.training_dict = {
            "RandomForestClassifier": ModelFactory.train_random_forest_classifier,
            "ExtraTreesClassifier": ModelFactory.train_extra_trees_classifier
        }
        self.trained_models = {}
        self.model_name = model_name

    @staticmethod
    def train_random_forest_classifier(x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
        print(list(x_train.columns))
        rfc = RandomForestClassifier()
        rfc.fit(x_train, y_train)
        return x_train, x_test, y_train, y_test, rfc

    @staticmethod
    def train_extra_trees_classifier(x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
        print(list(x_train.columns))
        pars = {"n_estimators": 500,
                "min_samples_split": 2,
                "min_samples_leaf": 2,
                "class_weight": 'balanced_subsample',
                "max_features": "sqrt"}
        model = ExtraTreesClassifier(**pars)
        model.fit(x_train, y_train)
        return x_train, x_test, y_train, y_test, model

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

    def train_model(self, x, y):
        if self.model_name:
            logging.info(f"Selected {self.model_name} algo")
            algo = self.training_dict.get(self.model_name)
        else:
            logging.info(f"Selected DEFAULT algo: train_random_forest_classifier")
            algo = ModelFactory.train_random_forest_classifier
        return algo(x, y)

    def prepare_models(self, df: pd.DataFrame, thresholds):
        label_cols = ['label_up_return', 'label_down_return']
        raw_x = df.drop(label_cols, axis=1)
        for thr in thresholds:
            logging.info(f"Training model for {thr} threshold")
            model_name, label_df = self.prepare_classification_label(df, thr)
            x, x_test, y, y_test, model = self.train_model(raw_x, label_df)
            # Make predictions for the test UP set
            y_predictions = model.predict(x_test)
            # View accuracy score
            logging.info(accuracy_score(y_test, y_predictions))
            acc = accuracy_score(y_test, y_predictions)
            logging.info(f"accuracy - {acc}")
            y_pred_proba = model.predict_proba(x_test)[::, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            logging.info(f"auc - {auc}")
            # View confusion matrix for test data and predictions
            matrix = confusion_matrix(y_test, y_predictions)
            logging.info(f"matrix - \n{matrix}")
            self.trained_models[model_name] = model
            logging.info(f'Finished model training for {thr} threshold')


if __name__ == '__main__':
    conn = get_connection()
    notifications = execute_sql(conn, "select * from notifications order by id")
    raw_flat_data = prepare_dataset(flat_notifications_from_sql(notifications))
    data_preprocessor = DataPreprocessor(raw_flat_data, True)
    df = data_preprocessor.provide_ready_df()
    mf = ModelFactory()
    mf.prepare_models(df, [80, 50, 20, -10])
    # artifacts_list = [SklearnModelArtifact(k) for k in mf.trained_models.keys()]
    # model_callable = artifacts(artifacts_list)(RFCClassifierModelService)
    # model = model_callable()
    service = SignalClassifierModelService()
    for k, v in mf.trained_models.items():
        logging.info(f"Packing model {k}")
        service.pack(k, v)
    saved_path = service.save()
    logging.info(f'Saved model to {saved_path}')

    logging.info("Finished Experiment")
