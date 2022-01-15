import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMModel
from .base import BaseModel


class MyLGBMModel(BaseModel):

    # ref:https://qiita.com/tubo/items/f83a97f2488cc1f40088 tuboさんのベースラインから
    #   :https://signate.jp/competitions/402/discussions/lgbm-baseline-except-text-vs-include-text-lb07994-1　masatoさんのベースラインから
    def __init__(self, model_params, fit_params):
        self.model_params = model_params
        self.fit_params = fit_params
        self.model = None

    def build_model(self):
        self.model = LGBMModel(**self.model_params)
        return self.model

    def fit(self, train_x, train_y, valid_x=None, valid_y=None):
        self.model = self.build_model()
        self.model.fit(
            train_x, train_y,
            eval_set=[[valid_x, valid_y]],
            **self.fit_params
        )
        return self.model

    def predict(self, model, valid_x):
        preds = self.model.predict(valid_x)
        return preds

    def visualize_importance(self, models, train_feat_df):
        feature_importance_df = pd.DataFrame()
        num = 0
        for i, model in models.items():
            _df = pd.DataFrame()
            _df['feature_importance'] = model.feature_importances_
            _df['column'] = train_feat_df.columns
            _df['fold'] = num + 1
            feature_importance_df = pd.concat([feature_importance_df, _df],
                                              axis=0, ignore_index=True)
            num += 1

        order = feature_importance_df.groupby('column')\
            .sum()[['feature_importance']]\
            .sort_values('feature_importance', ascending=False).index[:50]

        fig, ax = plt.subplots(figsize=(8, max(6, len(order) * .25)))
        sns.boxenplot(data=feature_importance_df,
                      x='feature_importance',
                      y='column',
                      order=order,
                      ax=ax,
                      palette='viridis',
                      orient='h')
        ax.tick_params(axis='x', rotation=90)
        ax.set_title('Importance')
        ax.grid()
        return fig, feature_importance_df

    def visualize_feature_importance(self, train_x, train_y, cv, num=50):
        self.vis_model = self.build_model()
        feature_importance_df = pd.DataFrame()
        for cv_num, (tr_idx, va_idx) in enumerate(cv):
            tr_x, va_x = train_x.values[tr_idx], train_x.values[va_idx]
            tr_y, va_y = train_y.values[tr_idx], train_y.values[va_idx]
            self.vis_model.fit(tr_x, tr_y,
                               eval_set=[[va_x, va_y]],
                               **self.fit_params)
            _df = pd.DataFrame()
            _df["feature_importance"] = self.vis_model.feature_importances_
            _df["columns"] = train_x.columns
            _df["fold"] = cv_num + 1
            feature_importance_df = pd.concat(
                [feature_importance_df, _df], axis=0, ignore_index=True)
        order = feature_importance_df.groupby("columns").sum()[["feature_importance"]].sort_values(
            "feature_importance", ascending=False).index[:num]
        fig, ax = plt.subplots(figsize=(12, max(4, len(order) * .2)))
        sns.boxenplot(data=feature_importance_df, y="columns",
                      x="feature_importance", order=order, ax=ax, palette="viridis")
        fig.tight_layout()
        ax.grid()
        ax.set_title(f"feature_importance_TOP{num}")
        fig.tight_layout()
        plt.show()
        return fig, feature_importance_df
