import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import numpy as np

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

    def fit(self, pipeline, X, y):
        pipeline.fit(X, y)
        return pipeline

    def save_model(self, pipeline, path):
        joblib.dump(pipeline, path)

