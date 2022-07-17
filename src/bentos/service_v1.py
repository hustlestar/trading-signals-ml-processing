# bento_service.py

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput, JsonOutput
from bentoml.frameworks.sklearn import SklearnModelArtifact

from data.preprocessing import DataPreprocessor
from data.notifcation_preparation import prepare_dataset, flat_notifications_from_sql, flat_notifications_from_json


@env(infer_pip_packages=True)
@artifacts([
    SklearnModelArtifact('up_80_return'),
    SklearnModelArtifact('up_50_return'),
    SklearnModelArtifact('up_20_return'),
    SklearnModelArtifact('down_10_return')
])
class SignalClassifierModelService(BentoService):
    """
    A minimum prediction service exposing a Scikit-learn model
    """
    def __init__(self):
        super(SignalClassifierModelService, self).__init__()

    @api(input=JsonInput(), batch=False, output=JsonOutput())
    def predict(self, notification):
        """
        An inference API named `predict` with Dataframe input adapter, which codifies
        how HTTP requests or CSV files are converted to a pandas Dataframe object as the
        inference API function input
        """
        print(f"Input is notification {notification['id']}")
        print(f"\n{notification}")
        raw_flat_data = prepare_dataset(flat_notifications_from_json([notification]))
        data_preprocessor = DataPreprocessor(raw_flat_data, False)
        ready_df = data_preprocessor.provide_ready_df()
        print("Finished preprocessing for input")
        res = {}
        print(self.artifacts)
        print(dir(self.artifacts))
        for k, v in self.artifacts.items():
            print(f"Predicting using {k}")
            res[k] = v.get().predict(ready_df)
        print(f"Predictions result is {res}")
        return res
