from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class ModelFactory:
    @staticmethod
    def get_model(model_name, **kwargs):
        models = {
            "linear": LinearRegression,
            "ridge": Ridge,
            "lasso": Lasso,
            "random_forest": RandomForestRegressor,
            "gradient_boosting": GradientBoostingRegressor
        }

        if model_name not in models:
            raise ValueError(f"Model '{model_name}' is not supported.")

        return models[model_name](**kwargs)
