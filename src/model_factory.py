from enum import Enum
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class ModelType(Enum):
    LINEAR = "linear"
    RIDGE = "ridge"
    LASSO = "lasso"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"

class ModelFactory:
    DEFAULT_PARAMS = {
        ModelType.RIDGE: {"alpha": 1.0},
        ModelType.LASSO: {"alpha": 0.1},
        ModelType.RANDOM_FOREST: {"n_estimators": 100, "random_state": 42},
        ModelType.GRADIENT_BOOSTING: {"n_estimators": 100, "learning_rate": 0.1}
    }

    @staticmethod
    def get_model(model_type: ModelType, **kwargs):
        models = {
            ModelType.LINEAR: LinearRegression,
            ModelType.RIDGE: Ridge,
            ModelType.LASSO: Lasso,
            ModelType.RANDOM_FOREST: RandomForestRegressor,
            ModelType.GRADIENT_BOOSTING: GradientBoostingRegressor
        }

        if model_type not in models:
            raise ValueError(f"Model '{model_type}' is not supported.")

        params = ModelFactory.DEFAULT_PARAMS.get(model_type, {}).copy()
        params.update(kwargs)

        return models[model_type](**params)
