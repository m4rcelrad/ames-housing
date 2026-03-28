import os
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_score


class ModelTrainer:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def build_pipeline(self, model, use_log_transform=True):
        if use_log_transform:
            regressor = TransformedTargetRegressor(
                regressor=model,
                func=np.log1p,
                inverse_func=np.expm1
            )
        else:
            regressor = model

        return Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("regressor", regressor)
        ])

    @staticmethod
    def evaluate_with_cv(pipeline, X, y, cv=5):
        scores = cross_val_score(
            pipeline, X, y,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1
        )

        rmse_scores = -scores
        return rmse_scores.mean(), rmse_scores.std()

    @staticmethod
    def fit(pipeline, X, y):
        pipeline.fit(X, y)
        return pipeline

    @staticmethod
    def save_model(pipeline, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(pipeline, path)
