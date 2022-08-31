# bento_service.py

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput, JsonOutput
from bentoml.frameworks.sklearn import SklearnModelArtifact

from data.notifcation_preparation import prepare_dataset, flat_notifications_from_json
from data.preprocessing import DataPreprocessor


@env(infer_pip_packages=True)
@artifacts([
    SklearnModelArtifact('etc_up_80_return'),
    SklearnModelArtifact('etc_up_50_return'),
    SklearnModelArtifact('etc_up_20_return'),
    SklearnModelArtifact('etc_down_10_return'),
    SklearnModelArtifact('rfc_up_80_return'),
    SklearnModelArtifact('rfc_up_50_return'),
    SklearnModelArtifact('rfc_up_20_return'),
    SklearnModelArtifact('rfc_down_10_return'),
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
        raw_flat_data = prepare_dataset(flat_notifications_from_json([notification]))
        data_preprocessor = DataPreprocessor(raw_flat_data, False)
        ready_df = data_preprocessor.provide_ready_df()
        print("Finished preprocessing for input")
        res = {}
        for k, v in self.artifacts.items():
            print(f"Predicting using {k}")
            try:
                res[k] = v.get().predict(ready_df)[0]
            except Exception as x:
                print(f"Exception during predict for {k}:\n{x}")
        print(f"Predictions result is \n{res}")
        return res
