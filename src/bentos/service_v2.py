# bento_service.py
import json

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput, JsonOutput
from bentoml.frameworks.sklearn import SklearnModelArtifact

from data.notifcation_preparation import prepare_raw_dataset, flat_notifications_from_json
from data.preprocessing import DataPreprocessor


@env(infer_pip_packages=True)
@artifacts([
    SklearnModelArtifact('ETC_UP_80_RETURN'),
    SklearnModelArtifact('ETC_UP_50_RETURN'),
    SklearnModelArtifact('ETC_UP_20_RETURN'),
    SklearnModelArtifact('ETC_DOWN_10_RETURN'),
    SklearnModelArtifact('RFC_UP_80_RETURN'),
    SklearnModelArtifact('RFC_UP_50_RETURN'),
    SklearnModelArtifact('RFC_UP_20_RETURN'),
    SklearnModelArtifact('RFC_DOWN_10_RETURN'),
])
class BagOfClassifierModelsService(BentoService):
    def __init__(self):
        super(BagOfClassifierModelsService, self).__init__()

    @api(input=JsonInput(), batch=False, output=JsonOutput())
    def predict(self, notification):
        """
        An inference API named `predict` with Dataframe input adapter, which codifies
        how HTTP requests or CSV files are converted to a pandas Dataframe object as the
        inference API function input
        """
        print(f"Input is notification {notification['id']}")
        res = {}
        try:
            raw_flat_data = prepare_raw_dataset(flat_notifications_from_json([notification]))
            data_preprocessor = DataPreprocessor(raw_flat_data, False)
            ready_df = data_preprocessor.provide_ready_df()
            print("Finished preprocessing for input")
            for k, v in self.artifacts.items():
                print(f"Predicting using {k}")
                res[k] = v.get().predict(ready_df)[0]
            print(f"Predictions result is \n{res}")
        except Exception as x:
            print(f"Exception during predict for input:\n {notification} \nexception:\n{x}")
            print(json.dumps(notification))
        return res
