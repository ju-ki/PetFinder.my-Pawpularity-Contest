from xgboost import XGBModel
from base import BaseModel


class MyXGBModel(BaseModel):
    def __init__(self, model_params):
        self.model_params = model_params

    def build_model(self):
        model = XGBModel(**self.model_params)
        return model

    def fit(self, train_x, train_y, valid_x, valid_y):
        self.model = self.build_model()
        self.model.fit(train_x, train_y,
                       eval_set=[(valid_x, valid_y)],
                       early_stopping_rounds=50,
                       verbose=False
                       )
        return self.model

    def predict(self, model, valid_x):
        return model.predict(valid_x)
