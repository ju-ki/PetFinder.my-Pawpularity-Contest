from catboost import Pool, CatBoostRegressor
from base import BaseModel


class MyCatRegressorModel(BaseModel):

    """
    エラーが起こる場合はこちらを参照してください
    https://catboost.ai/docs/concepts/python-reference_catboostregressor.html

    """

    def __init__(self, model_params):
        self.model_params = model_params

    def build_model(self):
        model = CatBoostRegressor(**self.model_params)
        return model

    def fit(self, train_x, train_y, valid_x, valid_y):
        train_pool = Pool(train_x, train_y)
        valid_pool = Pool(valid_x, valid_y)
        self.model = self.build_model()
        self.model.fit(train_pool,
                       early_stopping_rounds=50,
                       plot=False,
                       use_best_model=True,
                       eval_set=[valid_pool],
                       verbose=False,
                       )
        return self.model

    def predict(self, model, valid_x):
        return model.predict(valid_x)
