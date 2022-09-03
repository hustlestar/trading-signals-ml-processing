import logging
import os.path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

from bentos.service_v1 import SignalClassifierModelService
from bentos.service_v2 import BagOfClassifierModelsService
from config import get_connection
from data.db import execute_sql, batched_read_notification_sql
from data.preprocessing import DataPreprocessor
from data.notifcation_preparation import flat_notifications_from_sql, prepare_raw_dataset

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


class ModelFactory:
    def __init__(self, model_name=None):
        self.training_dict = {
            "RFC": ModelFactory.train_random_forest_classifier,
            "ETC": ModelFactory.train_extra_trees_classifier
        }
        self.trained_models = {}
        self.model_name = model_name

    @staticmethod
    def train_random_forest_classifier(x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        pars = {
            "n_estimators": 200,
            'min_samples_split': 2,
            'min_samples_leaf': 2,
            'max_samples': 0.8,
            'max_features': 0.8,
            "class_weight": "balanced_subsample",
            "max_depth": 100,
            "bootstrap": True
        }
        rfc = RandomForestClassifier(random_state=13, **pars)
        rfc.fit(x_train, y_train)
        return x_train, x_test, y_train, y_test, rfc

    @staticmethod
    def train_extra_trees_classifier(x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        pars = {
            "n_estimators": 200,
            "min_samples_split": 2,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "max_depth": 100,
            "max_features": "sqrt"
        }
        model = ExtraTreesClassifier(random_state=13, **pars)
        model.fit(x_train, y_train)
        return x_train, x_test, y_train, y_test, model

    @staticmethod
    def prepare_classification_label(df, threshold):
        col = 'LABEL_UP_RETURN' if threshold > 0 else 'LABEL_DOWN_RETURN'
        label_col_name = f'up_{threshold}_return'.upper() if threshold > 0 else f'down_{threshold * -1}_return'.upper()
        logging.info(f"Creating label column {label_col_name} with threshold of {threshold}")
        y = df[col]
        k = pd.DataFrame(y)
        conditions = [(k[col] >= threshold),
                      (k[col] < threshold)]
        y = pd.DataFrame(np.select(conditions, [1, 0]), columns=[f'label_{threshold}_{"up" if threshold > 0 else "down"}'.upper()])
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
        label_cols = ['LABEL_UP_RETURN', 'LABEL_DOWN_RETURN']
        raw_x = df.drop(label_cols, axis=1)
        for thr in thresholds:
            logging.info(f">>>>>>>>>>>>>>>>>>>Training model for {thr} threshold")
            model_name, label_df = self.prepare_classification_label(df, thr)
            x, x_test, y, y_test, model = self.train_model(raw_x, label_df)
            # Make predictions for the test UP set
            y_predictions = model.predict(x_test)
            # View accuracy score
            logging.info(accuracy_score(y_test, y_predictions))
            acc = accuracy_score(y_test, y_predictions)
            logging.info(f"accuracy - {acc}")
            logging.info(f"precision {precision_score(y_test, y_predictions)}")
            y_pred_proba = model.predict_proba(x_test)[::, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            logging.info(f"auc - {auc}")
            logging.warning(f"recall {recall_score(y_test, y_predictions)}")
            # View confusion matrix for test data and predictions
            matrix = confusion_matrix(y_test, y_predictions)
            logging.info(f"matrix - \n{matrix}")
            self.trained_models[model_name] = model
            logging.info(f'Finished model training for {thr} threshold')


def pack_models_to_service(mf: ModelFactory, service):
    for k, v in mf.trained_models.items():
        model_name = f"{mf.model_name}_{k}"
        logging.info(f"Packing model {model_name}")
        service.pack(model_name, v)


def load_raw_data(path):
    return pd.read_parquet(path)


def run_experiment(raw_flat_data, raw_df=None):
    data_preprocessor = DataPreprocessor(raw_flat_data, True)
    df = data_preprocessor.provide_ready_df(raw_df)
    # artifacts_list = [SklearnModelArtifact(k) for k in mf.trained_models.keys()]
    # model_callable = artifacts(artifacts_list)(RFCClassifierModelService)
    # model = model_callable()
    service = BagOfClassifierModelsService()
    mf = ModelFactory(model_name="ETC")
    mf.prepare_models(df, [80, 50, 20, -10])
    pack_models_to_service(mf, service)
    mf = ModelFactory(model_name="RFC")
    mf.prepare_models(df, [80, 50, 20, -10])
    pack_models_to_service(mf, service)
    saved_path = service.save()
    logging.info(f'Saved model to {saved_path}')
    return mf, service


def save_raw_df(path):
    conn = get_connection(database='dev_trading_bot')
    notifications = batched_read_notification_sql(conn, subset=0)
    raw_flat_data = prepare_raw_dataset(flat_notifications_from_sql(notifications))
    raw_df = pd.DataFrame(data=raw_flat_data)
    df = raw_df.reindex(sorted(raw_df.columns), axis=1)
    df.to_parquet(path)


if __name__ == '__main__':
    DEV_RAW_DF_PATH = os.path.join('data', 'DEV_20220901.parquet')
    PROD_RAW_DF_PATH = os.path.join('data', 'PROD_20220901.parquet')
    #save_raw_df(DEV_RAW_DF_PATH)
    dev = load_raw_data(DEV_RAW_DF_PATH)
    prod = load_raw_data(PROD_RAW_DF_PATH)
    df = pd.concat([dev, prod])
    mf, service = run_experiment(None, df)
    logging.info("Finished Experiment")
