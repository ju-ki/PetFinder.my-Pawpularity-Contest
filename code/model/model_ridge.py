from sklearn.linear_model import Ridge
from base import BaseModel


class MyRidge(BaseModel):
    def __init__(self, model_params):
        self.model_params = model_params
        self.model = None

    def build_model(self):
        model = Ridge(**self.model_params)
        return model

    def fit(self, train_x, train_y, valid_x, valid_y):
        self.model = self.build_model()
        self.model.fit(train_x, train_y)

        return self.model

    def predict(self, model, valid_x):
        return model.predict(valid_x)
